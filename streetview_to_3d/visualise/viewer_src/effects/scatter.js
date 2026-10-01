// The world is drawn as points; what is far off is only stored as triangles
// (land.ply, roads.ply, blocks.ply: postprocess/terrain.py), so they are
// scattered with points as they load. Each corner says how far apart its
// points go (the ply's "gap", terrain.point_gap); a triangle takes its
// closest corner's, rounded up to one of GAPS, and its points lie on a grid
// that far apart, fixed in the world -- the ground's on east/north, a
// slope's along it and up -- each jittered a little (JITTER, as the
// postprocess's own points are), so triangles side by side join without a
// seam. A point takes its colour from its triangle's corners.
//
// A wall (blocks.ply's facade: metres along, metres up from its foot, -1e4
// on a roof) is gridded on its facade instead: a whole number of points a
// bay and a floor (or of bays and floors a point), so where there are two
// or more points of each, some fall on its windows and some between --
// floors of windows at any distance points can show them (as postprocess/buildings.windows gives the near points: each
// pane a little lighter or darker, a pale sill under it, the wall darker at
// its foot); fewer, the wall's average.
//
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// the spacings points are drawn at, each a quarter over the last
export const GAPS = Array.from({ length: 28 }, (_, k) => 0.05 * 1.25 ** k);
export const JITTER = 0.3, // of the spacing, either way
  WALL_JITTER = 0.08, // on a wall's facade grid: its windows stay in rows
  FLAT = 0.7; // a triangle facing up this much is gridded on the ground, else along and up
export const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // of a bay
  WINDOW_V = [0.3, 0.8], // of a floor
  WINDOW = [0.12, [0.22, 0.43, 0.64]], // blue glass, lightly tinted by the wall
  WINDOW_MIX = 0.6, // a window over its wall this much: suggested, not printed
  PANE = 0.25, // each pane this much lighter or darker, at most
  SILL = [0.06, 0.12], // a sill this high (of a floor) under a window, this much lighter
  FOOT = [0.8, 3]; // a wall this dark at its foot, as it was by this high
const NO_FACADE = -1000;
const WINDOW_SHARE = (WINDOW_U[1] - WINDOW_U[0]) * (WINDOW_V[1] - WINDOW_V[0]);

// the smallest of GAPS at least gap
export function level(gap) {
  let k = 0;
  while (k < GAPS.length - 1 && GAPS[k] < gap * 0.999) k++;
  return k;
}

const loader = new PLYLoader();
loader.setCustomPropertyNameMapping({
  facade: ['facade_u', 'facade_v'],
  facadeLayout: ['facade_bay', 'facade_floor'],
  gap: ['gap'],
});

// A surface's ply (a buffer) as triangles in the viewer's frame (flip: y up, z south).
export function parseSurface(buffer, flip, name) {
  const geometry = loader.parse(buffer);
  if (!geometry.getAttribute('position')?.count || !geometry.index) {
    geometry.dispose();
    throw Error(`${name} has no triangles.`);
  }
  const layout = geometry.getAttribute('facadeLayout');
  if (layout && !layout.array.some((value) => Number.isFinite(value) && value > 0)) {
    geometry.deleteAttribute('facadeLayout');
  }
  geometry.applyMatrix4(flip);
  return geometry;
}

const hash = (i, j, k) => {
  const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453;
  return s - Math.floor(s);
};
const smooth = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};
const within = (x, a, b) => (x >= a && x <= b ? 1 : 0);

// about step, a whole number of them a size (a bay, a floor) or a whole number of sizes
const onGrid = (size, step) =>
  step < size ? size / Math.round(size / step) : size * Math.round(step / size);

// rgb (3 numbers, changed in place) of a wall at (u, v) on it: its window
// there if exact, else the wall's average of window and wall
export function facadeColour(rgb, u, v, exact = true, bay = BAY_M, floor = FLOOR_M) {
  const cu = u / bay,
    cv = v / floor;
  const qu = cu - Math.floor(cu),
    qv = cv - Math.floor(cv);
  const up = v >= 0.5 ? 1 : 0; // no window in the ground floor's first half metre
  const inside = within(qu, ...WINDOW_U) * within(qv, ...WINDOW_V);
  const sill = exact
    ? within(qu, WINDOW_U[0] - 0.03, WINDOW_U[1] + 0.03) *
      within(qv, WINDOW_V[0] - SILL[0], WINDOW_V[0])
    : 0;
  const frame = exact && inside && (qu < 0.34 || qu > 0.66 || qv < 0.34 || qv > 0.76);
  const pane = exact ? hash(Math.floor(cu), Math.floor(cv), 0) : 0.5;
  const w = (exact ? inside : WINDOW_SHARE) * up * WINDOW_MIX;
  const shade = 1 + (pane - 0.5) * 2 * PANE;
  const foot = FOOT[0] + (1 - FOOT[0]) * smooth(0, FOOT[1], v);
  for (let c = 0; c < 3; c++) {
    const wall = rgb[c] * (1 + SILL[1] * sill * up);
    const glass = frame ? rgb[c] * 0.78 : (rgb[c] * WINDOW[0] + WINDOW[1][c]) * shade;
    rgb[c] = (wall + (glass - wall) * w) * foot;
  }
  return rgb;
}

// Points over every triangle of geometry (parseSurface): a BufferGeometry of
// position, color and gap (each point's spacing, one of GAPS).
export function scatter(geometry) {
  const p = geometry.getAttribute('position').array,
    col = geometry.getAttribute('color')?.array,
    gapOf = geometry.getAttribute('gap'),
    facade = geometry.getAttribute('facade'),
    layout = geometry.getAttribute('facadeLayout'),
    index = geometry.index.array;
  const own = gapOf?.count && gapOf.array.some((g) => g > 0) ? gapOf.array : null;
  const wall = facade?.count ? facade.array : null;
  let pos = new Float32Array(3 << 16),
    rgb = new Float32Array(3 << 16),
    gaps = new Float32Array(1 << 16),
    n = 0;
  const more = (a) => {
    const b = new Float32Array(a.length * 2);
    b.set(a);
    return b;
  };
  const c3 = [0, 0, 0],
    at = [0, 0, 0],
    bt = [0, 0, 0],
    ct = [0, 0, 0];
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t],
      b = index[t + 1],
      c = index[t + 2];
    for (let k = 0; k < 3; k++) {
      at[k] = p[3 * a + k];
      bt[k] = p[3 * b + k];
      ct[k] = p[3 * c + k];
    }
    const e1 = [bt[0] - at[0], bt[1] - at[1], bt[2] - at[2]],
      e2 = [ct[0] - at[0], ct[1] - at[1], ct[2] - at[2]];
    const nx = e1[1] * e2[2] - e1[2] * e2[1],
      ny = e1[2] * e2[0] - e1[0] * e2[2],
      nz = e1[0] * e2[1] - e1[1] * e2[0];
    const len = Math.hypot(nx, ny, nz);
    if (len < 1e-9) continue;
    const g = own ? Math.min(own[a], own[b], own[c]) : 1;
    const step = GAPS[level(g)];
    const onWall =
      wall &&
      wall[2 * a + 1] > NO_FACADE &&
      wall[2 * b + 1] > NO_FACADE &&
      wall[2 * c + 1] > NO_FACADE;
    // the grid's two ways, how far apart on each, how jittered, where it starts
    let A, B, C, su, sv, jitter, v0, exact, seed;
    const bay = layout && layout.getX(a) > 0 ? layout.getX(a) : BAY_M;
    const floor = layout && layout.getY(a) > 0 ? layout.getY(a) : FLOOR_M;
    if (onWall) {
      [su, sv, jitter, v0, exact, seed] = [
        onGrid(bay, step),
        onGrid(floor, step),
        WALL_JITTER,
        0.5,
        Math.round(bay / step) > 1 && Math.round(floor / step) > 1,
        3,
      ];
      A = [wall[2 * a], wall[2 * a + 1]];
      B = [wall[2 * b], wall[2 * b + 1]];
      C = [wall[2 * c], wall[2 * c + 1]];
    } else {
      const flat = Math.abs(ny) >= FLAT * len;
      const h = Math.hypot(nx, nz) || 1,
        hx = flat ? 1 : -nz / h,
        hz = flat ? 0 : nx / h;
      const uv = (q) => (flat ? [q[0], q[2]] : [q[0] * hx + q[2] * hz, q[1]]);
      [su, sv, jitter, v0, exact, seed] = [
        step,
        step,
        JITTER,
        0,
        false,
        flat ? 1 : hx * hx > hz * hz ? 0 : 2,
      ];
      A = uv(at);
      B = uv(bt);
      C = uv(ct);
    }
    const det = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]);
    if (Math.abs(det) < 1e-12) continue;
    const spacing = GAPS[level(Math.max(su, sv))],
      k = level(spacing) + 97 * seed;
    const lo0 = Math.floor(Math.min(A[0], B[0], C[0]) / su - jitter),
      hi0 = Math.ceil(Math.max(A[0], B[0], C[0]) / su + jitter),
      lo1 = Math.floor(Math.min(A[1], B[1], C[1]) / sv - v0 - jitter),
      hi1 = Math.ceil(Math.max(A[1], B[1], C[1]) / sv - v0 + jitter);
    for (let i = lo0; i <= hi0; i++)
      for (let j = lo1; j <= hi1; j++) {
        // the jitter a grid point's own: the same in every triangle it falls near
        const x = (i + (hash(i, j, k) - 0.5) * 2 * jitter) * su,
          y = (j + v0 + (hash(j, i, k + 51) - 0.5) * 2 * jitter) * sv;
        const wb = ((x - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (y - A[1])) / det,
          wc = ((B[0] - A[0]) * (y - A[1]) - (x - A[0]) * (B[1] - A[1])) / det,
          wa = 1 - wb - wc;
        if (wa < 0 || wb < 0 || wc < 0) continue;
        if (n === gaps.length) {
          pos = more(pos);
          rgb = more(rgb);
          gaps = more(gaps);
        }
        for (let d = 0; d < 3; d++) {
          pos[3 * n + d] = wa * at[d] + wb * bt[d] + wc * ct[d];
          c3[d] = col ? wa * col[3 * a + d] + wb * col[3 * b + d] + wc * col[3 * c + d] : 0.6;
        }
        if (onWall) facadeColour(c3, x, y, exact, bay, floor);
        rgb.set(c3, 3 * n);
        gaps[n] = spacing;
        n++;
      }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos.slice(0, 3 * n), 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(rgb.slice(0, 3 * n), 3));
  out.setAttribute('gap', new THREE.Float32BufferAttribute(gaps.slice(0, n), 1));
  return out;
}
