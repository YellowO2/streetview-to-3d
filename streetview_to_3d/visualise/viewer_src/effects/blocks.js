import * as THREE from 'three';
import { PATCHES } from '@viewer/effects/patches';
import { haze } from '@viewer/effects/land';
import { SUN } from '@viewer/effects/water';
import { FLAT, JITTER } from '@viewer/effects/scatter';
import { tunable } from '@viewer/effects/tune-panel';

// The buildings (blocks.ply, postprocess/buildings.solid: triangles, each
// building its colour as it is) built of brush strokes and nothing else, as
// a painter builds them: no surface under them, where the strokes end the
// building ends. Their colours are the buildings' own, never made up here:
// this only draws them -- strokes, light, haze.
//
// - Its walls and roofs: strokes on a grid fixed on each (jittered as the
//   world's points are, JITTER; a wall's on its facade, metres along it and
//   up from its foot, a roof's on the ground or, steep, along it and up),
//   SPACE of the building's points' spacing apart (its gap); each a flat
//   brush mark lying on its face, across a wall, down a roof's slope.
// - Its structure: strokes along every edge where faces meet at a crease or a
//   face ends (corners, eaves, ridges, gables; not its foot), its colour.
// - Its windows: nothing of their own -- part of the wall, they are the
//   wall's strokes that fall on them, in the windows' colour its facade
//   gives (blocks.ply's glass), where strokes are close enough to tell them
//   (two to a bay and a floor).
//
// Painted by light and shade: a face toward the sun warm and lighter, away
// from it cool and darker (light); each stroke a little lighter or darker
// (vary), bristled along its length, its edge uneven -- or, round, more a
// point than a stroke: a round
// dab ROUND across, as the scene's points are. Which lies over which is
// each stroke's own, fixed in the world (lifted off its face by its own
// share of LAYER), never how it is seen, so nothing flickers. Scattered as a
// floating point is, but still (scatter): each a little off its place -- in
// front of or behind its face, along it (SCATTER of its spacing at most, a
// spacing counted no more than SCATTER_M, so far off big strokes scatter as
// little as near ones) -- tipped out of it, bigger or smaller, its own way,
// fixed.
//
// Near DA3's own points (the ply's near: within postprocess/buildings.BLEND_M,
// their colour already mixed toward DA3's there), a stroke turns into one of
// them as it comes up to them: round, facing the eye, DA3's points' size,
// floating and pulsing as they do (points.js), lit as they are -- not at all.

export const KNOBS = {
  size: [1.1, 0.5, 2.5], // a stroke's size, of what it was made at
  round: [1, 0, 1], // 0 brush strokes, 1 round dabs, as points
  scatter: [0.5, 0, 1], // each stroke off its place, tipped, bigger or smaller: still, as a point floats
  light: [0.5, 0, 1], // how far apart in the sun and in the shade
  vary: [0.11, 0, 0.4], // each stroke, how much lighter or darker
};
const SPACE = 1.2, // strokes this much of the points' spacing apart
  LONG = 2.2, // a stroke this long, of that
  WIDE = 1.2, // and this wide
  EDGE_WIDE = 0.45, // a structure's stroke this wide
  ROUND = 1.6, // a round dab this across, of the strokes' spacing
  LAYER = 0.04, // a stroke lifted off its face up to this much of its spacing
  SCATTER = [0.35, 0.25], // a stroke off its place at most this much of its spacing: off its face, along it
  SCATTER_M = 0.4, // its spacing counted as no more than this: far off, sparse strokes scatter as near ones
  TIP = 0.5, // tipped out of its face at most this much (of a right angle's tangent)
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
const SURFACE = 0,
  EDGE = 1; // as postprocess/buildings.Blocks.kind

const knobs = tunable('Buildings', KNOBS);
const f = (x) => x.toFixed(4);
const v3 = (v) => `vec3(${v.map(f).join(', ')})`;
const sun = new THREE.Vector3(...SUN).normalize().toArray();

const vertexShader = `
  uniform float ${Object.keys(KNOBS).join(', ')};
  attribute vec3 centre, facing, along, tint;
  attribute vec4 shape; // length, width (m), kind, seed
  attribute float near; // how near DA3's points: 1 drawn as they are
  uniform float pointM, styleTime, styleFloat, styleLook, stylePointScale; // DA3's points' look (points.js)
  varying vec3 colour, world;
  varying vec2 mark;
  varying float seed, shapeOf;
  float h1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
  void main() {
    float kind = shape.z;
    seed = shape.w;
    mark = position.xy;
    vec3 eye = cameraPosition - centre;
    // a face's side toward the eye: its outside, the buildings being closed
    vec3 n = dot(facing, eye) < 0. ? -facing : facing;
    float t = near; // turning into DA3's points: still scatter giving way to their float
    vec3 across = kind == ${f(EDGE)} ? normalize(cross(along, eye)) : normalize(cross(n, along));
    // scattered, still: off its place, tipped, bigger or smaller, its own way
    vec4 r = vec4(h1(seed * 5.3), h1(seed * 6.7), h1(seed * 8.9), h1(seed * 10.1)) * 2. - 1.;
    float spacing = shape.y / (kind == ${f(EDGE)} ? ${f(EDGE_WIDE)} : ${f(WIDE)});
    float loose = scatter * min(1., ${f(SCATTER_M)} / spacing) * (1. - t); // the same metres however big
    vec3 off = (n * r.x * ${f(SCATTER[0])} + (along * r.y + across * r.z) * ${f(SCATTER[1])}) * spacing * loose;
    vec3 way = normalize(along + n * r.w * ${f(TIP)} * loose),
      side = normalize(across + n * r.y * ${f(TIP)} * loose);
    // near DA3, facing the eye as its points do
    vec3 right = vec3(viewMatrix[0][0], viewMatrix[1][0], viewMatrix[2][0]),
      up = vec3(viewMatrix[0][1], viewMatrix[1][1], viewMatrix[2][1]);
    way = normalize(mix(way, right, t));
    side = normalize(mix(side, up, t));
    // its size: a stroke's, or round a dab's -- near DA3, its points' (pulsing as they do)
    float rounder = mix(round, 1., t);
    vec2 dims = shape.xy;
    if (kind != ${f(EDGE)}) dims = mix(dims, vec2(shape.y * ${f(ROUND / WIDE)}), rounder) * size;
    if (kind == ${f(EDGE)}) dims = mix(dims, vec2(shape.y * 2.), rounder) * size;
    dims *= 1. + r.z * ${f(SWELL)} * loose;
    float phase = h1(seed * 11.3) * 6.2832;
    dims = mix(dims, vec2(pointM * stylePointScale * (1. + sin(styleTime * .8 + phase) * .14 * styleFloat)), t);
    float lift = h1(seed * 7.1) * ${f(LAYER)} * shape.y + (kind == ${f(EDGE)} ? .05 * shape.y : 0.);
    // and floating as they do
    vec3 drift = vec3(sin(styleTime * .55 + phase) * .45, sin(styleTime * .8 + phase) * .65,
      cos(styleTime * .5 + phase) * .45) * styleLook * .004 * styleFloat * t;
    world = centre + off + drift + (way * position.x * dims.x + side * position.y * dims.y) * .5
      + (kind == ${f(EDGE)} ? normalize(eye) : n) * lift * (1. - t);
    shapeOf = rounder;
    // its colour, its face lit or in shade, a little lighter or darker its own way --
    // near DA3, as DA3's points are: as they are
    float sunlit = smoothstep(-.05, .25, dot(n, ${v3(sun)}));
    colour = tint * mix(vec3(1.), mix(${v3(SHADED)}, ${v3(SUNLIT)}, sunlit), light * (1. - t))
      * (1. + (h1(seed * 3.3) - .5) * 2. * vary * (1. - t));
    gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.);
  }`;

const fragmentShader = `
  uniform float haze;
  varying vec3 colour, world;
  varying vec2 mark;
  varying float seed, shapeOf;
  ${PATCHES}
  void main() {
    // a brush's mark: square-ended along, round across -- or a round dab; its edge ragged
    float e = mix(pow(abs(mark.x), 4.) + mark.y * mark.y, dot(mark, mark), shapeOf);
    if (e > 1. - .3 * vnoise(vec2(mark.x * 3. + seed * 17., mark.y * 2.))) discard;
    // bristles: streaks along it, fainter on a dab
    vec3 c = colour * (1. + (.2 * vnoise(vec2(mark.x * 1.5, mark.y * 6.) + seed * 31.) - .1) * (1. - .6 * shapeOf));
    gl_FragColor = vec4(hazed(c, length(world - cameraPosition), haze), 1.);
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

// The strokes building the buildings' triangles (parseSurface: the viewer's
// frame): { centre, facing, along, tint, shape } per stroke (see the shader).
export function strokes(geometry) {
  const p = geometry.getAttribute('position'),
    col = geometry.getAttribute('color'),
    gapOf = geometry.getAttribute('gap'),
    facade = geometry.getAttribute('facade'),
    layout = geometry.getAttribute('facadeLayout'),
    glassOf = geometry.getAttribute('glass')?.count ? geometry.getAttribute('glass') : null,
    index = geometry.index.array;
  const out = { centre: [], facing: [], along: [], tint: [], shape: [] };
  const add = (c, n, a, t, len, wide, kind) => {
    out.centre.push(...c);
    out.facing.push(...n);
    out.along.push(...a);
    out.tint.push(...t);
    out.shape.push(len, wide, kind, hash(c[0], c[1], c[2]) * 100);
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
    // the grid's two ways on the face, and the stroke's way
    let uv, along;
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
    if (wall) {
      // along the wall: where its facade's u grows, v the same
      const e1 = new THREE.Vector3().subVectors(B, A),
        e2 = new THREE.Vector3().subVectors(Cv, A);
      along = e1
        .multiplyScalar(uc[1] - ua[1])
        .sub(e2.multiplyScalar(ub[1] - ua[1]))
        .multiplyScalar(1 / det)
        .normalize();
    } else {
      // down the slope; a flat roof's east-west
      const down = new THREE.Vector3(n.x, 0, n.z);
      along =
        down.length() > 0.05
          ? down.sub(n.clone().multiplyScalar(down.dot(n))).normalize()
          : new THREE.Vector3(1, 0, 0);
    }
    const k = Math.round(s * 1000) + (wall ? 3 : Math.abs(n.y) >= FLAT ? 1 : 2) * 97;
    const tint = [C(a), C(b), C(c)],
      glass = glassOf && [a, b, c].map((i) => [glassOf.getX(i), glassOf.getY(i), glassOf.getZ(i)]);
    const lo0 = Math.floor(Math.min(ua[0], ub[0], uc[0]) / s - 1),
      hi0 = Math.ceil(Math.max(ua[0], ub[0], uc[0]) / s + 1),
      lo1 = Math.floor(Math.min(ua[1], ub[1], uc[1]) / s - 1),
      hi1 = Math.ceil(Math.max(ua[1], ub[1], uc[1]) / s + 1);
    const inside = (x, y) => within({ uv: [ua, ub, uc], det }, x, y);
    // a wall's stroke falling on a window, where strokes are close enough to tell them
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
          along.toArray(),
          at(...(windows && onWindow(x, y) ? glass : tint), ...w),
          s * LONG,
          s * WIDE,
          SURFACE,
        );
      }
    // its edges, for the structure's strokes
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
    const dir = new THREE.Vector3().subVectors(W, U).normalize();
    const k = Math.max(1, Math.round(len / s));
    for (let i = 0; i < k; i++) {
      const c = U.clone().lerp(W, (i + 0.5) / k);
      add(
        c.toArray(),
        faces[0].toArray(),
        dir.toArray(),
        tint,
        (len / k) * 1.6,
        s * EDGE_WIDE,
        EDGE,
      );
    }
  }
  return out;
}

// The buildings DA3 reaches (buildings.ply, postprocess/buildings.points:
// their points already met to DA3's -- moved onto its walls, left out where
// it has them, pulled toward it and its colour beside it): a stroke on each
// point, as the triangles' are laid, its facing, way, kind (SURFACE, EDGE)
// and colour (its windows' too) the point's own, as long and wide as its
// spacing (gap) says.
export function pointStrokes(geometry) {
  const p = geometry.getAttribute('position'),
    n = geometry.getAttribute('normal'),
    a = geometry.getAttribute('along'),
    col = geometry.getAttribute('color'),
    gapOf = geometry.getAttribute('gap'),
    kindOf = geometry.getAttribute('kind');
  const count = p.count;
  const nearOf = geometry.getAttribute('near');
  const made = {
    near: nearOf?.count ? nearOf.array.slice(0, count) : new Float32Array(count),
    centre: p.array.slice(0, 3 * count),
    facing: n.array.slice(0, 3 * count),
    along: a.array.slice(0, 3 * count),
    tint: col ? col.array.slice(0, 3 * count) : new Float32Array(3 * count).fill(0.6),
    shape: new Float32Array(4 * count),
  };
  for (let i = 0; i < count; i++) {
    const s = gapOf && gapOf.getX(i) > 0 ? gapOf.getX(i) : 1,
      kind = kindOf ? Math.round(kindOf.getX(i)) : SURFACE;
    made.shape.set(
      kind === EDGE ? [s * 1.6, s * EDGE_WIDE, EDGE, 0] : [s * LONG, s * WIDE, SURFACE, 0],
      4 * i,
    );
    made.shape[4 * i + 3] = hash(p.getX(i), p.getY(i), p.getZ(i)) * 100;
  }
  geometry.dispose();
  return strokeMesh(made);
}

// The buildings' triangles (parseSurface: the viewer's frame) as their strokes, drawn.
export function blocksStrokes(geometry) {
  const made = strokes(geometry);
  geometry.dispose();
  return strokeMesh(made);
}

// the uniforms DA3's points move by (points.js), so strokes near them move alike
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

// strokes ({ centre, facing, along, tint, shape, near? }) as one instanced mesh
function strokeMesh(made) {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3),
  );
  g.setIndex([0, 1, 2, 0, 2, 3]);
  made.near ??= new Float32Array(made.shape.length / 4); // far off: none near DA3
  for (const [name, size] of [
    ['near', 1],
    ['centre', 3],
    ['facing', 3],
    ['along', 3],
    ['tint', 3],
    ['shape', 4],
  ])
    g.setAttribute(name, new THREE.InstancedBufferAttribute(new Float32Array(made[name]), size));
  g.instanceCount = made.shape.length / 4;
  const blocks = new THREE.Mesh(
    g,
    new THREE.ShaderMaterial({
      // DA3's points' look, set as theirs are (app.js setPointSize; controller.js, as points.js has them)
      uniforms: { ...knobs, haze, pointM: { value: 0.1 }, ...pointStyle() },
      side: THREE.DoubleSide,
      vertexShader,
      fragmentShader,
    }),
  );
  blocks.frustumCulled = false;
  blocks.userData.pointStyle = true; // moved as DA3's points are (controller.js)
  return blocks;
}
