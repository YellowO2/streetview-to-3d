import * as THREE from 'three';
import { haze, HAZED } from '@viewer/style/haze';
import { FLAT, GAPS, JITTER, level } from '@viewer/world/scatter';
import { tunable } from '@viewer/ui/tune-panel';
import { styleUniforms } from '@viewer/style/points';
import { worldUniforms, WORLD_VERTEX, WORLD_FRAGMENT, cutPoint } from '@viewer/style/world-points';
import { SUN, f, v3, hash } from '@viewer/util';

// Far buildings, land and moving things drawn as lit round dabs; colours come from the data.
// Near DA3 (`near` 1) a dab turns into a DA3 point: eye-facing, its size and float, unlit.

const KNOBS = {
  size: [1.1, 0.5, 2.5], // dab size, of its built size
  scatter: [0.5, 0, 1], // fixed per-dab offset and size jitter
  light: [0.5, 0, 1], // sunlit vs shaded contrast
  vary: [0.11, 0, 0.4], // per-dab lightness variation
};
const SPACE = 1.2, // far-building dab spacing, of the point gap
  ROUND = 1.6, // dab diameter, of its spacing
  EDGE_DAB = 0.9, // edge dab, of its spacing
  LAYER = 0.04, // most a dab lifts off its face, of its spacing (fixed draw order, no flicker)
  SCATTER = 0.35, // most a dab sits off its face, of its spacing
  SCATTER_M = 0.4, // spacing cap for scatter, so big far dabs scatter no more than near ones
  SWELL = 0.15, // most a dab grows or shrinks
  ROUGH = 0.3, // edge roughness, of its radius squared
  CREASE = Math.cos((30 * Math.PI) / 180); // faces meeting at over 30 degrees make an edge
const NO_FACADE = -1000;
// facade rhythm when blocks.ply gives none (postprocess/buildings.facade_layout)
const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // window span, of a bay
  WINDOW_V = [0.3, 0.8]; // and of a floor
const SUNLIT = [1.12, 1.02, 0.86],
  SHADED = [0.5, 0.56, 0.78];
const EDGE = 1; // postprocess/buildings.Blocks kind for an edge point

const knobs = tunable('Buildings', KNOBS);
const sun = new THREE.Vector3(...SUN).normalize().toArray();

const vertexShader = `
  #include <fog_pars_vertex>
  ${WORLD_VERTEX}
  uniform float ${Object.keys(KNOBS).join(', ')}, halfHeight;
  uniform float pointM, styleTime, styleFloat, styleLook, stylePointScale; // DA3's point look (points.js)
  attribute vec3 facing, tint;
  attribute float dab, near; // size (m); 1 = drawn as a DA3 point
  #ifdef OWN_SEED
  attribute float grain; // per-dab seed that moves with it (traffic.js)
  #endif
  varying vec3 colour, world;
  varying float seed;
  float h1(float x) { return fract(sin(x * 12.9898) * 43758.5453); }
  void main() {
    vec3 centre = position;
    #ifdef OWN_SEED
    seed = grain;
    #else
    seed = fract(sin(dot(centre, vec3(12.9898, 78.233, 37.719))) * 43758.5453) * 100.;
    #endif
    vec3 eye = cameraPosition - centre;
    // the face's side toward the eye: buildings are closed
    vec3 n = dot(facing, eye) < 0. ? -facing : facing;
    float t = near;
    vec4 r = vec4(h1(seed * 5.3), h1(seed * 6.7), h1(seed * 8.9), h1(seed * 10.1)) * 2. - 1.;
    float spacing = dab / ${f(ROUND)};
    float loose = scatter * min(1., ${f(SCATTER_M)} / spacing) * (1. - t);
    vec3 off = n * r.x * ${f(SCATTER)} * spacing * loose;
    float phase = h1(seed * 11.3) * 6.2832;
    vec3 drift = vec3(sin(styleTime * .55 + phase) * .45, sin(styleTime * .8 + phase) * .65,
      cos(styleTime * .5 + phase) * .45) * styleLook * .004 * styleFloat * t;
    world = centre + off + drift + n * h1(seed * 7.1) * ${f(LAYER)} * spacing * (1. - t);
    float d = mix(dab * size * (1. + r.z * ${f(SWELL)} * loose),
      pointM * stylePointScale * (1. + sin(styleTime * .8 + phase) * .14 * styleFloat), t);
    float demoIn;
    world += demoed(centre, h1(seed * 13.7), demoIn) - centre;
    if (shotAway(centre)) demoIn = 0.;
    vec4 mv = viewMatrix * vec4(world, 1.);
    gl_Position = demoIn < .5 ? vec4(2., 2., 2., 1.) : projectionMatrix * mv;
    vec4 mvPosition = mv;
    #include <fog_vertex>
    gl_PointSize = demoIn < .5 ? 0. : worldSize(d * projectionMatrix[1][1] * halfHeight / -mv.z, seed * .01, fract(seed * .37));
    if (gl_PointSize == 0.) gl_Position = vec4(2., 2., 2., 1.);
    // lit by face toward the sun, varied per dab; near DA3, unlit
    float sunlit = smoothstep(-.05, .25, dot(n, ${v3(sun)}));
    colour = tint * mix(vec3(1.), mix(${v3(SHADED)}, ${v3(SUNLIT)}, sunlit), light * (1. - t))
      * (1. + (h1(seed * 3.3) - .5) * 2. * vary * (1. - t));
  }`;

const fragmentShader = `
  #include <fog_pars_fragment>
  uniform float haze;
  varying vec3 colour, world;
  varying float seed;
  ${HAZED}
  ${WORLD_FRAGMENT}
  void main() {
    float u = gl_PointCoord.x * 2. - 1., v = 1. - gl_PointCoord.y * 2.;
    ${cutPoint('seed * .01', `u * u + v * v > 1. - ${f(ROUGH)} * vnoise(vec2(u * 3. + seed * 17., v * 2.))`)}
    gl_FragColor = vec4(hazed(colour * pointShade, length(world - cameraPosition), haze), 1.);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    #include <fog_fragment>
  }`;

// barycentric weights of (x, y) in a 2D triangle, or null if outside
function within({ uv: [ua, ub, uc], det }, x, y) {
  const wb = ((x - ua[0]) * (uc[1] - ua[1]) - (uc[0] - ua[0]) * (y - ua[1])) / det,
    wc = ((ub[0] - ua[0]) * (y - ua[1]) - (x - ua[0]) * (ub[1] - ua[1])) / det;
  return wb >= -1e-6 && wc >= -1e-6 && wb + wc <= 1 + 1e-6 ? [1 - wb - wc, wb, wc] : null;
}

// Dabs over triangles (viewer frame): { centre, facing, tint, dab (size, m), near }.
// edges: add crease strokes; levels: snap each triangle's gap to GAPS so neighbours share a grid
// (the land); smooth: interpolate corner normals; space, jitter, round: of the spacing.
export function marks(
  geometry,
  {
    edges: structure = true,
    levels = false,
    smooth = false,
    space = SPACE,
    jitter = JITTER,
    round = ROUND,
  } = {},
) {
  const p = geometry.getAttribute('position'),
    col = geometry.getAttribute('color'),
    gapOf = geometry.getAttribute('gap'),
    facade = geometry.getAttribute('facade'),
    layout = geometry.getAttribute('facadeLayout'),
    glassOf = geometry.getAttribute('glass')?.count ? geometry.getAttribute('glass') : null,
    normal = smooth ? geometry.getAttribute('normal') : null,
    index = geometry.index.array;
  const nearOf = geometry.getAttribute('near')?.count ? geometry.getAttribute('near') : null;
  const out = { centre: [], facing: [], tint: [], dab: [], near: nearOf ? [] : null };
  const add = (c, n, t, dab, near = 0) => {
    out.centre.push(...c);
    out.facing.push(...n);
    out.tint.push(...t);
    out.dab.push(dab);
    out.near?.push(near);
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
    const corners =
      normal && [a, b, c].map((i) => [normal.getX(i), normal.getY(i), normal.getZ(i)]);
    const facingAt = (w) =>
      corners ? new THREE.Vector3(...at(...corners, ...w)).normalize().toArray() : n.toArray();
    const wall = facade && [a, b, c].every((i) => facade.getY(i) > NO_FACADE / 2);
    // grid axes: facade, ground (east/north) or along-slope/up; dabs stretched to cover slopes
    let uv,
      stretch = 1;
    if (wall) {
      uv = (i) => [facade.getX(i), facade.getY(i)];
    } else if (Math.abs(n.y) >= FLAT) {
      uv = (i) => [p.getX(i), p.getZ(i)];
      stretch = 1 / Math.abs(n.y);
    } else {
      const h = Math.hypot(n.x, n.z);
      uv = (i) => [(p.getX(i) * -n.z) / h + (p.getZ(i) * n.x) / h, p.getY(i)];
      stretch = 1 / h;
    }
    const gap = levels
      ? GAPS[level(Math.min(gapOf.getX(a), gapOf.getX(b), gapOf.getX(c)) / Math.sqrt(stretch))]
      : gapOf && gapOf.getX(a) > 0
        ? gapOf.getX(a)
        : 1;
    const s = gap * space;
    const [ua, ub, uc] = [uv(a), uv(b), uv(c)];
    const det = (ub[0] - ua[0]) * (uc[1] - ua[1]) - (uc[0] - ua[0]) * (ub[1] - ua[1]);
    if (Math.abs(det) < 1e-12) continue;
    const k = Math.round(s * 1000) + (wall ? 3 : Math.abs(n.y) >= FLAT ? 1 : 2) * 97;
    const tint = [C(a), C(b), C(c)],
      glass = glassOf && [a, b, c].map((i) => [glassOf.getX(i), glassOf.getY(i), glassOf.getZ(i)]);
    // only cells whose jittered point can land in the triangle
    const from = (lo) => Math.ceil(lo / s - 0.5 - jitter - 1e-9),
      to = (hi) => Math.floor(hi / s - 0.5 + jitter + 1e-9);
    const lo0 = from(Math.min(ua[0], ub[0], uc[0])),
      hi0 = to(Math.max(ua[0], ub[0], uc[0])),
      lo1 = from(Math.min(ua[1], ub[1], uc[1])),
      hi1 = to(Math.max(ua[1], ub[1], uc[1]));
    const inside = (x, y) => within({ uv: [ua, ub, uc], det }, x, y);
    // wall dabs on a window take the glass colour, if dabs are dense enough to show windows
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
        const x = (i + 0.5 + (hash(i, j, k) - 0.5) * 2 * jitter) * s,
          y = (j + 0.5 + (hash(j, i, k + 51) - 0.5) * 2 * jitter) * s;
        const w = inside(x, y);
        if (!w) continue;
        add(
          at(A.toArray(), B.toArray(), Cv.toArray(), ...w),
          facingAt(w),
          at(...(windows && onWindow(x, y) ? glass : tint), ...w),
          s * round * stretch,
          nearOf ? w[0] * nearOf.getX(a) + w[1] * nearOf.getX(b) + w[2] * nearOf.getX(c) : 0,
        );
      }
    // collect edges for crease strokes
    if (!structure) continue;
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
    if (faces.length === 2 && Math.abs(faces[0].dot(faces[1])) > CREASE) continue; // not a crease
    if (faces.length === 1 && Math.abs(U.y - W.y) < 0.05 && V(o).y > U.y + 0.05) continue; // building foot
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

// Dab diameter (of its spacing) that leaves no gaps on a grid jittered by jitter,
// allowing for the roughest edge (ROUGH) and smallest swell (SWELL) at default knobs.
export function covering(jitter) {
  const furthest = Math.SQRT1_2 * (1 + 2 * jitter),
    least = Math.sqrt(1 - ROUGH) * (1 - SWELL * KNOBS.scatter[0]);
  return (2 * furthest) / (least * KNOBS.size[0]);
}

// blocks.ply triangles (viewer frame) as drawn dabs
export function blockPoints(geometry) {
  const made = marks(geometry);
  geometry.dispose();
  return pointsOf(made);
}

// buildings.ply points (with normal, kind and near) as drawn dabs
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

// Dabs ({ centre, facing, tint, dab, near?, grain? }) as a THREE.Points with packed attributes.
// fog: apply the scene fog (the land). grain: per-dab seeds for moving things; else seeded by position.
export function pointsOf(made, { fog = false } = {}) {
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
    near[i] = Math.round((made.near ? made.near[i] : 0) * 255);
  }
  g.setAttribute('facing', new THREE.BufferAttribute(facing, 3, true));
  g.setAttribute('tint', new THREE.BufferAttribute(tint, 3, true));
  g.setAttribute('dab', new THREE.Float32BufferAttribute(made.dab, 1));
  g.setAttribute('near', new THREE.BufferAttribute(near, 1, true));
  if (made.grain) g.setAttribute('grain', new THREE.Float32BufferAttribute(made.grain, 1));
  const halfHeight = { value: 1 };
  const points = new THREE.Points(
    g,
    new THREE.ShaderMaterial({
      // pointM and style* are set as for DA3's points (app.js setPointSize, controller.js)
      uniforms: {
        ...knobs,
        haze,
        halfHeight,
        pointM: { value: 0.1 },
        ...styleUniforms(),
        ...worldUniforms,
        ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
      },
      fog,
      defines: made.grain ? { OWN_SEED: '' } : {},
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
  points.userData.pointStyle = true; // styled like DA3's points (controller.js)
  return points;
}
