// Triangle surfaces (roads.ply) scattered with points on a world-fixed jittered grid, spaced by
// the smallest corner's gap rounded up to GAPS, so neighbouring triangles join without seams.
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { hash } from '@viewer/effects/util';

// allowed point spacings (m), each 1.25x the last
export const GAPS = Array.from({ length: 28 }, (_, k) => 0.05 * 1.25 ** k);
export const JITTER = 0.3, // of the spacing, either way
  FLAT = 0.7; // normal.y above this: gridded east/north, else along-slope/up

// index of the smallest of GAPS at least gap
export function level(gap) {
  let k = 0;
  while (k < GAPS.length - 1 && GAPS[k] < gap * 0.999) k++;
  return k;
}

const loader = new PLYLoader();
loader.setCustomPropertyNameMapping({
  facade: ['facade_u', 'facade_v'],
  facadeLayout: ['facade_bay', 'facade_floor'],
  glass: ['glass_r', 'glass_g', 'glass_b'],
  gap: ['gap'],
});

// A surface ply (also blocks.ply's facade and glass) as triangles in the viewer frame (flip: y up, z south).
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

// Points over every triangle (parseSurface): a BufferGeometry of position, color and gap.
export function scatter(geometry) {
  const p = geometry.getAttribute('position').array,
    col = geometry.getAttribute('color')?.array,
    gapOf = geometry.getAttribute('gap'),
    index = geometry.index.array;
  const own = gapOf?.count && gapOf.array.some((g) => g > 0) ? gapOf.array : null;
  let pos = new Float32Array(3 << 16),
    rgb = new Float32Array(3 << 16),
    gaps = new Float32Array(1 << 16),
    n = 0;
  const more = (a) => {
    const b = new Float32Array(a.length * 2);
    b.set(a);
    return b;
  };
  const at = [0, 0, 0],
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
    const step = GAPS[level(own ? Math.min(own[a], own[b], own[c]) : 1)];
    // grid axes: east/north, or along-slope/up
    const flat = Math.abs(ny) >= FLAT * len;
    const h = Math.hypot(nx, nz) || 1,
      hx = flat ? 1 : -nz / h,
      hz = flat ? 0 : nx / h;
    const uv = (q) => (flat ? [q[0], q[2]] : [q[0] * hx + q[2] * hz, q[1]]);
    const seed = flat ? 1 : hx * hx > hz * hz ? 0 : 2;
    const A = uv(at),
      B = uv(bt),
      C = uv(ct);
    const det = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]);
    if (Math.abs(det) < 1e-12) continue;
    const k = level(step) + 97 * seed;
    const lo0 = Math.floor(Math.min(A[0], B[0], C[0]) / step - JITTER),
      hi0 = Math.ceil(Math.max(A[0], B[0], C[0]) / step + JITTER),
      lo1 = Math.floor(Math.min(A[1], B[1], C[1]) / step - JITTER),
      hi1 = Math.ceil(Math.max(A[1], B[1], C[1]) / step + JITTER);
    for (let i = lo0; i <= hi0; i++)
      for (let j = lo1; j <= hi1; j++) {
        // jitter depends only on the grid point, so it matches in neighbouring triangles
        const x = (i + (hash(i, j, k) - 0.5) * 2 * JITTER) * step,
          y = (j + (hash(j, i, k + 51) - 0.5) * 2 * JITTER) * step;
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
          rgb[3 * n + d] = col
            ? wa * col[3 * a + d] + wb * col[3 * b + d] + wc * col[3 * c + d]
            : 0.6;
        }
        gaps[n] = step;
        n++;
      }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(pos.slice(0, 3 * n), 3));
  out.setAttribute('color', new THREE.Float32BufferAttribute(rgb.slice(0, 3 * n), 3));
  out.setAttribute('gap', new THREE.Float32BufferAttribute(gaps.slice(0, n), 1));
  return out;
}
