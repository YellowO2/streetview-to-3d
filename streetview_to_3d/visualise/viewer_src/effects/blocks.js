import * as THREE from 'three';
import { PATCHES } from '@viewer/effects/patches';
import { haze } from '@viewer/effects/land';
import { SUN } from '@viewer/effects/water';
import { FLAT, JITTER } from '@viewer/effects/scatter';
import { tunable } from '@viewer/effects/tune-panel';

// The buildings built of points and nothing else, as a painter builds them of
// dabs: no surface under them, where the points end the building ends. Each
// one the GPU's own point -- a round dab, squashed as a disc lying on its face
// would look from here. Their colours are the buildings' own, never made up
// here: this only draws them -- dabs, light, haze.
//
// The far buildings (blocks.ply, postprocess/buildings.solid: triangles) are
// laid with points as they load (marks):
// - their walls and roofs: on a grid fixed on each (jittered as the world's
//   points are, JITTER; a wall's on its facade, metres along it and up from its
//   foot, a roof's on the ground or, steep, along it and up), SPACE of the
//   building's points' spacing apart (its gap);
// - their structure: along every edge where faces meet at a crease or a face
//   ends (corners, eaves, ridges, gables; not its foot), smaller (EDGE_DAB);
// - their windows: nothing of their own -- part of the wall, they are the
//   wall's points that fall on them, in the windows' colour its facade gives
//   (blocks.ply's glass), where points are close enough to tell them (two to
//   a bay and a floor).
// The buildings DA3 reaches (buildings.ply, postprocess/buildings.points) are
// points already, met to DA3's -- moved onto its walls, left out where it has
// them, pulled toward it and its colour beside it.
//
// Each dab ROUND of its spacing across, lit: a face toward the sun warm and
// lighter, away from it cool and darker (light); a little lighter or darker
// its own way (vary), its edge uneven. Which lies over which is each dab's
// own, fixed in the world (lifted off its face by its own share of LAYER),
// never how it is seen, so nothing flickers. Scattered as a floating point
// is, but still (scatter): each a little off its place -- in front of or
// behind its face, along it (SCATTER of its spacing at most, a spacing
// counted no more than SCATTER_M, so far off big dabs scatter as little as
// near ones) -- bigger or smaller, its own way, fixed.
//
// Near DA3's own points (buildings.ply's near: within postprocess/
// buildings.BLEND_M, their colour already mixed toward DA3's there), a dab
// turns into one of them as it comes up to them: facing the eye, DA3's
// points' size, floating and pulsing as they do (points.js), lit as they
// are -- not at all.

export const KNOBS = {
  size: [1.1, 0.5, 2.5], // a dab's size, of what it was made at
  scatter: [0.5, 0, 1], // each dab off its place, bigger or smaller: still, as a point floats
  light: [0.5, 0, 1], // how far apart in the sun and in the shade
  vary: [0.11, 0, 0.4], // each dab, how much lighter or darker
};
const SPACE = 1.2, // the far buildings' points this much of their spacing apart
  ROUND = 1.6, // a dab this across, of its spacing
  EDGE_DAB = 0.9, // an edge's dab, of its spacing
  LAYER = 0.04, // a dab lifted off its face up to this much of its spacing
  SCATTER = [0.35, 0.25], // a dab off its place at most this much of its spacing: off its face, along it
  SCATTER_M = 0.4, // its spacing counted as no more than this: far off, sparse dabs scatter as near ones
  SWELL = 0.15, // bigger or smaller by at most this much
  CREASE = Math.cos((30 * Math.PI) / 180); // faces meeting at more than 30 degrees: an edge
const NO_FACADE = -1000;
// a facade's rhythm, when blocks.ply gives none (postprocess/buildings.facade_layout)
export const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // a window, of a bay
  WINDOW_V = [0.3, 0.8]; // and of a floor
const SUNLIT = [1.12, 1.02, 0.86], // the sun's warmth on a face
  SHADED = [0.5, 0.56, 0.78]; // the shade's cool, the sky's blue in it
const EDGE = 1; // postprocess/buildings.Blocks.kind's

const knobs = tunable('Buildings', KNOBS);
const f = (x) => x.toFixed(4);
const v3 = (v) => `vec3(${v.map(f).join(', ')})`;
const sun = new THREE.Vector3(...SUN).normalize().toArray();

const vertexShader = `
  uniform float ${Object.keys(KNOBS).join(', ')}, halfHeight;
  uniform float pointM, styleTime, styleFloat, styleLook, stylePointScale; // DA3's points' look (points.js)
  attribute vec3 facing, tint;
  attribute float dab, near; // its size (m); how near DA3's points: 1 drawn as they are
  varying vec3 colour, world;
  varying vec2 dir;
  varying float seed, flatness;
  float h1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
  void main() {
    vec3 centre = position;
    seed = fract(sin(dot(centre, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * 100.;
    vec3 eye = cameraPosition - centre;
    // a face's side toward the eye: its outside, the buildings being closed
    vec3 n = dot(facing, eye) < 0. ? -facing : facing;
    float t = near; // turning into DA3's points: still scatter giving way to their float
    // scattered, still: off its place, bigger or smaller, its own way
    vec4 r = vec4(h1(seed * 5.3), h1(seed * 6.7), h1(seed * 8.9), h1(seed * 10.1)) * 2. - 1.;
    float spacing = dab / ${f(ROUND)};
    float loose = scatter * min(1., ${f(SCATTER_M)} / spacing) * (1. - t); // the same metres however big
    vec3 a = normalize(abs(n.y) < .9 ? cross(n, vec3(0., 1., 0.)) : cross(n, vec3(1., 0., 0.))), b = cross(n, a);
    vec3 off = (n * r.x * ${f(SCATTER[0])} + (a * r.y + b * r.z) * ${f(SCATTER[1])}) * spacing * loose;
    // near DA3, floating and pulsing as its points do
    float phase = h1(seed * 11.3) * 6.2832;
    vec3 drift = vec3(sin(styleTime * .55 + phase) * .45, sin(styleTime * .8 + phase) * .65,
      cos(styleTime * .5 + phase) * .45) * styleLook * .004 * styleFloat * t;
    world = centre + off + drift + n * h1(seed * 7.1) * ${f(LAYER)} * spacing * (1. - t);
    float d = mix(dab * size * (1. + r.z * ${f(SWELL)} * loose),
      pointM * stylePointScale * (1. + sin(styleTime * .8 + phase) * .14 * styleFloat), t);
    vec4 mv = viewMatrix * vec4(world, 1.);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = d * projectionMatrix[1][1] * halfHeight / -mv.z;
    // a disc lying on its face, as seen: squashed along its facing on the screen, the more edge-on
    vec3 nv = normalize((viewMatrix * vec4(n, 0.)).xyz);
    flatness = mix(max(abs(dot(nv, normalize(-mv.xyz))), .25), 1., t);
    dir = length(nv.xy) > 1e-4 ? normalize(nv.xy) : vec2(1., 0.);
    // its colour, its face lit or in shade, a little lighter or darker its own way --
    // near DA3, as DA3's points are: as they are
    float sunlit = smoothstep(-.05, .25, dot(n, ${v3(sun)}));
    colour = tint * mix(vec3(1.), mix(${v3(SHADED)}, ${v3(SUNLIT)}, sunlit), light * (1. - t))
      * (1. + (h1(seed * 3.3) - .5) * 2. * vary * (1. - t));
  }`;

const fragmentShader = `
  uniform float haze;
  varying vec3 colour, world;
  varying vec2 dir;
  varying float seed, flatness;
  ${PATCHES}
  void main() {
    // round, squashed as seen, its edge uneven
    vec2 q = vec2(gl_PointCoord.x, 1. - gl_PointCoord.y) * 2. - 1.;
    float u = dot(q, dir) / flatness, v = dot(q, vec2(-dir.y, dir.x));
    if (u * u + v * v > 1. - .3 * vnoise(vec2(u * 3. + seed * 17., v * 2.))) discard;
    gl_FragColor = vec4(hazed(colour, length(world - cameraPosition), haze), 1.);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

// (a, b, c): where (x, y) is in a triangle on its own two ways (uv, det), if in it
function within({ uv: [ua, ub, uc], det }, x, y) {
  const wb = ((x - ua[0]) * (uc[1] - ua[1]) - (uc[0] - ua[0]) * (y - ua[1])) / det,
    wc = ((ub[0] - ua[0]) * (y - ua[1]) - (x - ua[0]) * (ub[1] - ua[1])) / det;
  return wb >= -1e-6 && wc >= -1e-6 && wb + wc <= 1 + 1e-6 ? [1 - wb - wc, wb, wc] : null;
}

const hash = (i, j, k) => {
  const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453;
  return s - Math.floor(s);
};

// The points laid over the buildings' triangles (parseSurface: the viewer's
// frame): { centre, facing, tint, dab } per point (dab: its size, metres).
export function marks(geometry) {
  const p = geometry.getAttribute('position'),
    col = geometry.getAttribute('color'),
    gapOf = geometry.getAttribute('gap'),
    facade = geometry.getAttribute('facade'),
    layout = geometry.getAttribute('facadeLayout'),
    glassOf = geometry.getAttribute('glass')?.count ? geometry.getAttribute('glass') : null,
    index = geometry.index.array;
  const out = { centre: [], facing: [], tint: [], dab: [] };
  const add = (c, n, t, dab) => {
    out.centre.push(...c);
    out.facing.push(...n);
    out.tint.push(...t);
    out.dab.push(dab);
  };
  const V = (i) => new THREE.Vector3(p.getX(i), p.getY(i), p.getZ(i));
  const C = (i) => (col ? [col.getX(i), col.getY(i), col.getZ(i)] : [0.6, 0.6, 0.6]);
  const at = (a, b, c, wa, wb, wc) => [0, 1, 2].map((d) => wa * a[d] + wb * b[d] + wc * c[d]);
  const edges = new Map();
  const key = (v) =>
    v
      .toArray()
      .map((x) => Math.round(x * 1000))
      .join(',');
  for (let t = 0; t < index.length; t += 3) {
    const [a, b, c] = [index[t], index[t + 1], index[t + 2]];
    const [A, B, Cv] = [V(a), V(b), V(c)];
    const n = new THREE.Vector3().subVectors(B, A).cross(new THREE.Vector3().subVectors(Cv, A));
    if (n.length() < 1e-9) continue;
    n.normalize();
    const gap = gapOf && gapOf.getX(a) > 0 ? gapOf.getX(a) : 1;
    const s = gap * SPACE;
    const wall = facade && [a, b, c].every((i) => facade.getY(i) > NO_FACADE / 2);
    // the grid's two ways on the face
    let uv;
    if (wall) {
      uv = (i) => [facade.getX(i), facade.getY(i)];
    } else if (Math.abs(n.y) >= FLAT) {
      uv = (i) => [p.getX(i), p.getZ(i)];
    } else {
      const h = Math.hypot(n.x, n.z);
      uv = (i) => [(p.getX(i) * -n.z) / h + (p.getZ(i) * n.x) / h, p.getY(i)];
    }
    const [ua, ub, uc] = [uv(a), uv(b), uv(c)];
    const det = (ub[0] - ua[0]) * (uc[1] - ua[1]) - (uc[0] - ua[0]) * (ub[1] - ua[1]);
    if (Math.abs(det) < 1e-12) continue;
    const k = Math.round(s * 1000) + (wall ? 3 : Math.abs(n.y) >= FLAT ? 1 : 2) * 97;
    const tint = [C(a), C(b), C(c)],
      glass = glassOf && [a, b, c].map((i) => [glassOf.getX(i), glassOf.getY(i), glassOf.getZ(i)]);
    const lo0 = Math.floor(Math.min(ua[0], ub[0], uc[0]) / s - 1),
      hi0 = Math.ceil(Math.max(ua[0], ub[0], uc[0]) / s + 1),
      lo1 = Math.floor(Math.min(ua[1], ub[1], uc[1]) / s - 1),
      hi1 = Math.ceil(Math.max(ua[1], ub[1], uc[1]) / s + 1);
    const inside = (x, y) => within({ uv: [ua, ub, uc], det }, x, y);
    // a wall's point falling on a window, where points are close enough to tell them
    const bay = layout && layout.getX(a) > 0 ? layout.getX(a) : BAY_M,
      floor = layout && layout.getY(a) > 0 ? layout.getY(a) : FLOOR_M;
    const windows = wall && glass && s * 2 <= Math.min(bay, floor);
    const onWindow = (x, y) => {
      const qx = x / bay - Math.floor(x / bay),
        qy = y / floor - Math.floor(y / floor);
      return (
        y >= 0.5 && qx >= WINDOW_U[0] && qx <= WINDOW_U[1] && qy >= WINDOW_V[0] && qy <= WINDOW_V[1]
      );
    };
    for (let i = lo0; i <= hi0; i++)
      for (let j = lo1; j <= hi1; j++) {
        const x = (i + 0.5 + (hash(i, j, k) - 0.5) * 2 * JITTER) * s,
          y = (j + 0.5 + (hash(j, i, k + 51) - 0.5) * 2 * JITTER) * s;
        const w = inside(x, y);
        if (!w) continue;
        add(
          at(A.toArray(), B.toArray(), Cv.toArray(), ...w),
          n.toArray(),
          at(...(windows && onWindow(x, y) ? glass : tint), ...w),
          s * ROUND,
        );
      }
    // its edges, for the structure's points
    for (const [u, v, o] of [
      [a, b, c],
      [b, c, a],
      [c, a, b],
    ]) {
      const ends = [key(V(u)), key(V(v))].sort().join('|');
      const e = edges.get(ends) || { u, v, o, faces: [], s, tint: C(u) };
      e.faces.push(n.clone());
      edges.set(ends, e);
    }
  }
  for (const { u, v, o, faces, s, tint } of edges.values()) {
    const [U, W] = [V(u), V(v)];
    if (faces.length === 2 && Math.abs(faces[0].dot(faces[1])) > CREASE) continue; // flat across it
    if (faces.length === 1 && Math.abs(U.y - W.y) < 0.05 && V(o).y > U.y + 0.05) continue; // its foot
    const len = U.distanceTo(W);
    if (len < 1e-3) continue;
    const k = Math.max(1, Math.round(len / (s * EDGE_DAB)));
    for (let i = 0; i < k; i++)
      add(
        U.clone()
          .lerp(W, (i + 0.5) / k)
          .toArray(),
        faces[0].toArray(),
        tint,
        s * EDGE_DAB,
      );
  }
  return out;
}

// The buildings' triangles (parseSurface: the viewer's frame) as their points, drawn.
export function blockPoints(geometry) {
  const made = marks(geometry);
  geometry.dispose();
  return pointsOf(made);
}

// The buildings DA3 reaches (buildings.ply: points, each its facing, kind --
// an edge's smaller -- and how near DA3's), drawn.
export function buildingPoints(geometry) {
  const p = geometry.getAttribute('position'),
    gapOf = geometry.getAttribute('gap'),
    kindOf = geometry.getAttribute('kind'),
    nearOf = geometry.getAttribute('near');
  const count = p.count;
  const col = geometry.getAttribute('color');
  const made = {
    centre: p.array.slice(0, 3 * count),
    facing: geometry.getAttribute('normal').array.slice(0, 3 * count),
    tint: col ? col.array.slice(0, 3 * count) : new Float32Array(3 * count).fill(0.6),
    dab: new Float32Array(count),
    near: nearOf?.count ? nearOf.array.slice(0, count) : null,
  };
  for (let i = 0; i < count; i++) {
    const s = gapOf && gapOf.getX(i) > 0 ? gapOf.getX(i) : 1;
    made.dab[i] = s * (kindOf && Math.round(kindOf.getX(i)) === EDGE ? EDGE_DAB : ROUND);
  }
  geometry.dispose();
  return pointsOf(made);
}

// the uniforms DA3's points move by (points.js), so points near them move alike
const pointStyle = () => ({
  styleDensity: { value: 1 },
  stylePointScale: { value: 1 },
  styleRound: { value: 0 },
  styleTime: { value: 0 },
  styleFloat: { value: 0 },
  styleScan: { value: 0 },
  styleRadius: { value: 1 },
  styleLook: { value: 1 },
  styleCenter: { value: new THREE.Vector3() },
  styleReveal: { value: 1 },
});

// points ({ centre, facing, tint, dab, near? }) as the GPU's points: a few
// bytes each (its facing, colour and nearness as bytes)
function pointsOf(made) {
  const count = made.dab.length;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(made.centre, 3));
  const facing = new Int8Array(3 * count),
    tint = new Uint8Array(3 * count),
    near = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    for (let d = 0; d < 3; d++) {
      facing[3 * i + d] = Math.round(made.facing[3 * i + d] * 127);
      tint[3 * i + d] = Math.round(Math.min(Math.max(made.tint[3 * i + d], 0), 1) * 255);
    }
    near[i] = Math.round((made.near ? made.near[i] : 0) * 255); // far off: none near DA3
  }
  g.setAttribute('facing', new THREE.BufferAttribute(facing, 3, true));
  g.setAttribute('tint', new THREE.BufferAttribute(tint, 3, true));
  g.setAttribute('dab', new THREE.Float32BufferAttribute(made.dab, 1));
  g.setAttribute('near', new THREE.BufferAttribute(near, 1, true));
  const halfHeight = { value: 1 };
  const points = new THREE.Points(
    g,
    new THREE.ShaderMaterial({
      // DA3's points' look, set as theirs are (app.js setPointSize; controller.js, as points.js has them)
      uniforms: {
        ...knobs,
        haze,
        halfHeight,
        pointM: { value: 0.1 },
        ...pointStyle(),
      },
      vertexShader,
      fragmentShader,
    }),
  );
  const size = new THREE.Vector2();
  points.onBeforeRender = (renderer) => {
    renderer.getDrawingBufferSize(size);
    halfHeight.value = size.y / 2;
  };
  points.frustumCulled = false;
  points.userData.pointStyle = true; // moved as DA3's points are (controller.js)
  return points;
}
