import * as THREE from 'three';
import { Reflector } from 'three/addons/objects/Reflector.js';
import { GAPS, JITTER, level } from '@viewer/effects/scatter';
import { demo, DEMO } from '@viewer/effects/demo';
import { shot, SHOT } from '@viewer/effects/shot';

// The scene's water (postprocess/water.py, water.json: each body a flat
// shape at its level, and a grid of metres from dry land) as points, coloured
// the way water is: a mirror of the world, as much as the angle it is seen at
// says (Fresnel) -- looking down near by, mostly the water's own colour,
// turquoise in the shallows, blue out deep; looking across, far off, mostly
// what stands over it and the sky. Its waves tilt each point, which turns
// its share of mirror and shifts what it mirrors.
//
// The mirror is three.js's Reflector: each level's water, never drawn itself,
// draws the scene from under the water into a picture (RES of the screen)
// just before its points, which read it where they are on screen. The points
// lie on a grid fixed in the world, as far apart as the world's points are
// there (gapOf), each drawn SIZE of that across, in metres.
//
// Painted: each dab a flat stroke, as flat as the water is seen (flatness at
// least); what it mirrors from up to smear (of the picture) above or below
// it, the more the rougher the water there -- reflections streaked
// downwards, as painters draw them; what it mirrors a little darker and
// bluer (reflected) than what stands there; each dab its own a little
// lighter or darker (vary), a little bigger or smaller (spread).
export const RES = 0.5, // the mirror's picture, of the screen's pixels
  SUN = [3, 5, 4]; // as the viewport's light
// the look, as tuned by eye
export const KNOBS = {
  size: 1.83, // a point across, of its spacing
  r0: 0.16, // the share mirrored seen straight down (water's 0.02, Schlick's; more, as games do)
  edge: 1.06, // ... rising to all of it edge on this steeply (Schlick's 5)
  distort: 0.025, // what a point mirrors shifts this far (of the picture) a unit of tilt
  flatness: 0.05,
  smear: 0.0107,
  reflectedR: 0.82,
  reflectedG: 0.88,
  reflectedB: 0.96,
  vary: 0.023,
  spread: 0,
  deepM: 15.5, // all of deep this far from dry land
  glintSharp: 400, // the sun on it: how sharp
  glint: 1.5, // ... how bright
  rough: 1, // the waves' steepness, of WAVES'
  pace: 0.35, // their speed, of deep water's
};
export const COLOURS = { shallow: '#4fc4b0', deep: '#135a80', glintColour: '#fff4d8' };
// waves: their heading (radians from east), length and steepness (the most their slope tilts)
export const WAVES = [
  [0.4, 11, 0.05],
  [1.9, 6.3, 0.045],
  [-1.0, 3.1, 0.04],
  [2.7, 1.7, 0.03],
];
const BLOCK_M = 20, // points spaced alike a block
  TILE_M = 200; // points drawn together, skipped together out of view

const time = { value: 0 };
export function tickWater(dt) {
  time.value += Math.min(dt, 0.1);
}

// the look, one uniform each, shared by every water
const knobs = {
  ...Object.fromEntries(Object.entries(KNOBS).map(([k, v]) => [k, { value: v }])),
  ...Object.fromEntries(
    Object.entries(COLOURS).map(([k, hex]) => [k, { value: new THREE.Color(hex) }]),
  ),
};

const hash = (i, j, k) => {
  const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453;
  return s - Math.floor(s);
};

// metres from dry land at (east, north), bilinear on the shore's grid
function shoreAt(shore, e, n) {
  if (!shore) return KNOBS.deepM;
  const { lo, cell, size, metres } = shore;
  const x = Math.min(Math.max((e - lo) / cell - 0.5, 0), size - 1.001),
    y = Math.min(Math.max((n - lo) / cell - 0.5, 0), size - 1.001);
  const i = Math.floor(x),
    j = Math.floor(y),
    fx = x - i,
    fy = y - j;
  const at = (a, b) => metres[b * size + a];
  return (
    (at(i, j) * (1 - fx) + at(i + 1, j) * fx) * (1 - fy) +
    (at(i, j + 1) * (1 - fx) + at(i + 1, j + 1) * fx) * fy
  );
}

// does the segment (a, b) cross the edge (p, q)?
function crosses(ax, ay, bx, by, [px, py], [qx, qy]) {
  const side = (x0, y0, x1, y1, x, y) => (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
  return (
    side(ax, ay, bx, by, px, py) * side(ax, ay, bx, by, qx, qy) < 0 &&
    side(px, py, qx, qy, ax, ay) * side(px, py, qx, qy, bx, by) < 0
  );
}

// The points on rings (outlines and holes, even-odd), (east, north): a grid
// a block at its spacing, jittered as the world's points are. Whether a point
// is in: its block's middle's answer, flipped by each edge between the two.
export function waterGrid(rings, gapOf) {
  const edges = rings.flatMap((r) => r.map((p, i) => [p, r[(i + 1) % r.length]]));
  const lo = [Infinity, Infinity],
    hi = [-Infinity, -Infinity];
  for (const r of rings)
    for (const [e, n] of r) {
      lo[0] = Math.min(lo[0], e);
      lo[1] = Math.min(lo[1], n);
      hi[0] = Math.max(hi[0], e);
      hi[1] = Math.max(hi[1], n);
    }
  const b0 = lo.map((v) => Math.floor(v / BLOCK_M)),
    b1 = hi.map((v) => Math.floor(v / BLOCK_M));
  const local = new Map(); // a block's edges
  for (const edge of edges) {
    const [[pe, pn], [qe, qn]] = edge;
    for (
      let i = Math.floor(Math.min(pe, qe) / BLOCK_M);
      i <= Math.floor(Math.max(pe, qe) / BLOCK_M);
      i++
    )
      for (
        let j = Math.floor(Math.min(pn, qn) / BLOCK_M);
        j <= Math.floor(Math.max(pn, qn) / BLOCK_M);
        j++
      ) {
        const key = `${i},${j}`;
        if (!local.has(key)) local.set(key, []);
        local.get(key).push(edge);
      }
  }
  const out = { east: [], north: [], gap: [] };
  for (let bi = b0[0]; bi <= b1[0]; bi++)
    for (let bj = b0[1]; bj <= b1[1]; bj++) {
      const ce = (bi + 0.5) * BLOCK_M,
        cn = (bj + 0.5) * BLOCK_M;
      // the middle: a ray east, counted against every edge
      let middle = false;
      for (const [[pe, pn], [qe, qn]] of edges)
        if (pn > cn !== qn > cn && ce < ((qe - pe) * (cn - pn)) / (qn - pn) + pe) middle = !middle;
      const near = local.get(`${bi},${bj}`) || [];
      if (!middle && !near.length) continue;
      const k = level(gapOf(ce, cn)),
        step = GAPS[k];
      for (let i = Math.ceil((bi * BLOCK_M) / step); i * step < (bi + 1) * BLOCK_M; i++)
        for (let j = Math.ceil((bj * BLOCK_M) / step); j * step < (bj + 1) * BLOCK_M; j++) {
          const e = (i + (hash(i, j, k) - 0.5) * 2 * JITTER) * step,
            n = (j + (hash(j, i, k + 51) - 0.5) * 2 * JITTER) * step;
          let inside = middle;
          for (const edge of near) if (crosses(ce, cn, e, n, ...edge)) inside = !inside;
          if (!inside) continue;
          out.east.push(e);
          out.north.push(n);
          out.gap.push(step);
        }
    }
  return out;
}

const vec = (v) => v.map((x) => x.toFixed(4)).join(',');

function pointsMaterial(mirror, matrix, halfHeight) {
  const sun = new THREE.Vector3(...SUN).normalize().toArray();
  const waves = WAVES.map(([heading, length, steep]) => {
    const k = (2 * Math.PI) / length,
      speed = Math.sqrt(9.81 / k); // deep water's: longer waves go faster
    return `s += vec2(${vec([Math.cos(heading), Math.sin(heading)])}) * ${steep.toFixed(4)} * rough *
      cos(dot(p, vec2(${vec([Math.cos(heading) * k, Math.sin(heading) * k])})) - time * pace * ${(k * speed).toFixed(4)});`;
  }).join('\n');
  const steepest = WAVES.reduce((a, w) => a + w[2], 0).toFixed(4);
  return new THREE.ShaderMaterial({
    uniforms: {
      mirror: { value: mirror },
      matrix: { value: matrix },
      halfHeight,
      time,
      ...demo,
      ...shot,
      ...knobs,
      ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
    },
    fog: true, // the scene's haze: far water fades into the sky as the land does
    vertexShader: `
      #include <fog_pars_vertex>
      ${DEMO}
      ${SHOT}
      uniform sampler2D mirror; uniform mat4 matrix; uniform float halfHeight, time;
      uniform float ${Object.keys(KNOBS).join(', ')};
      uniform vec3 ${Object.keys(COLOURS).join(', ')};
      attribute vec2 water; // metres from dry land, spacing
      varying vec3 colour;
      varying float squash;
      void main() {
        float dab = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
        vec4 world = modelMatrix * vec4(position, 1.);
        // its slope (east, north), the waves' summed
        vec2 p = position.xy, s = vec2(0.);
        ${waves}
        vec3 n = normalize(vec3(-s.x, 1., s.y)), v = normalize(cameraPosition - world.xyz);
        float mirrored = r0 + (1. - r0) * pow(1. - max(dot(n, v), 0.), edge);
        vec4 at = matrix * vec4(position, 1.); // the Reflector's: from its own frame, as its points are
        float choppy = clamp(length(s) / ${steepest}, 0., 1.);
        vec2 shift = n.xz * distort + vec2(0., (fract(dab * 7.13) - .5) * 2. * smear * (.3 + .7 * choppy));
        vec3 seen = texture2D(mirror, at.xy / at.w + shift).rgb * vec3(reflectedR, reflectedG, reflectedB);
        vec3 own = mix(shallow, deep, smoothstep(0., deepM, water.x));
        colour = mix(own, seen, mirrored)
          + glintColour * pow(max(dot(reflect(-v, n), vec3(${vec(sun)})), 0.), glintSharp) * glint;
        colour *= 1. + (dab - .5) * 2. * vary;
        squash = max(abs(v.y), flatness);
        // the demos: moved as every point is (demo.js)
        float demoIn;
        if (shotAway(world.xyz)) demoIn = 0.; // shot away (shot.js)
        else world.xyz = demoed(world.xyz, dab, 1., demoIn);
        vec4 mv = viewMatrix * world, mvPosition = mv;
        gl_Position = demoIn < .5 ? vec4(2., 2., 2., 1.) : projectionMatrix * mv;
        #include <fog_vertex>
        gl_PointSize = size * (1. + (fract(dab * 3.71) - .5) * 2. * spread) * water.y
          * projectionMatrix[1][1] * halfHeight / -mv.z;
      }`,
    fragmentShader: `
      #include <fog_pars_fragment>
      varying vec3 colour;
      varying float squash;
      void main() {
        vec2 q = gl_PointCoord - .5;
        if (length(vec2(q.x, q.y / squash)) > .5) discard; // a flat dab, as flat as the water is seen
        gl_FragColor = vec4(colour, 1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }`,
  });
}

let mirroring = false; // one mirror at a time: in another's picture, a water is its old self

// One level's water: a Reflector over its shape (east, north), its points its children,
// so they are left out of its own picture.
function waterAt(level, rings, shore, gapOf) {
  const shapes = rings.outer.map((outer, i) => {
    const s = new THREE.Shape(outer.map(([e, n]) => new THREE.Vector2(e, n)));
    s.holes = rings.holes[i].map((h) => new THREE.Path(h.map(([e, n]) => new THREE.Vector2(e, n))));
    return s;
  });
  const geometry = new THREE.ShapeGeometry(shapes);
  const water = new Reflector(geometry, {
    textureWidth: 1,
    textureHeight: 1,
    clipBias: 0.003,
    multisample: 0,
  });
  const picture = water.getRenderTarget(),
    { tDiffuse, textureMatrix } = water.material.uniforms;
  water.material.dispose();
  water.material = new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false });
  // the shape lies in its own east/north: up turns to the viewer's y, north to -z
  water.rotateX(-Math.PI / 2);
  water.position.y = level;
  water.renderOrder = -10; // its picture drawn before its points
  water.name = 'Water';

  const halfHeight = { value: 1 },
    size = new THREE.Vector2(),
    draw = water.onBeforeRender;
  water.onBeforeRender = (renderer, scene, camera) => {
    renderer.getDrawingBufferSize(size);
    halfHeight.value = size.y / 2;
    if (mirroring) return;
    picture.setSize(Math.max(1, Math.round(size.x * RES)), Math.max(1, Math.round(size.y * RES)));
    mirroring = true;
    try {
      draw.call(water, renderer, scene, camera);
    } finally {
      mirroring = false;
    }
  };
  const all = [...rings.outer, ...rings.holes.flat()];
  const grid = waterGrid(all, gapOf);
  const material = pointsMaterial(tDiffuse.value, textureMatrix.value, halfHeight);
  const tiles = new Map();
  grid.east.forEach((e, i) => {
    const key = `${Math.floor(e / TILE_M)},${Math.floor(grid.north[i] / TILE_M)}`;
    if (!tiles.has(key)) tiles.set(key, []);
    tiles.get(key).push(i);
  });
  for (const of of tiles.values()) {
    const position = new Float32Array(3 * of.length),
      paint = new Float32Array(2 * of.length);
    of.forEach((i, j) => {
      const e = grid.east[i],
        n = grid.north[i];
      position.set([e, n, 0], 3 * j);
      paint.set([shoreAt(shore, e, n), grid.gap[i]], 2 * j);
    });
    const part = new THREE.BufferGeometry();
    part.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
    part.setAttribute('water', new THREE.Float32BufferAttribute(paint, 2));
    part.computeBoundingSphere();
    const points = new THREE.Points(part, material);
    points.userData.ownMotion = true; // the style's floating leaves it be
    water.add(points);
  }
  water.userData.points = grid.east.length;
  return water;
}

// water.json's bodies, one Reflector a level (its points under it), in the
// viewer's frame; gapOf(east, north): the world's points' spacing there.
export function waterSurfaces(data, gapOf) {
  const byLevel = new Map();
  for (const s of data?.surfaces || []) {
    if (!Number.isFinite(s.level) || !(s.outer?.length >= 3)) continue;
    if (!byLevel.has(s.level)) byLevel.set(s.level, { outer: [], holes: [] });
    byLevel.get(s.level).outer.push(s.outer);
    byLevel.get(s.level).holes.push((s.holes || []).filter((h) => h.length >= 3));
  }
  return [...byLevel].map(([level, rings]) => waterAt(level, rings, data.shore, gapOf));
}
