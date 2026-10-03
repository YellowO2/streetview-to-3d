import * as THREE from 'three';
import { Reflector } from 'three/addons/objects/Reflector.js';
import { GAPS, JITTER, level } from '@viewer/effects/scatter';
import {
  worldUniforms,
  WORLD_VERTEX,
  WORLD_FRAGMENT,
  cutPoint,
} from '@viewer/effects/world-points';
import { SUN, f, v3, hash } from '@viewer/effects/util';

// Water (water.json) as points on a world-fixed grid, Fresnel-mixed between the water colour
// (turquoise shallows, blue deep) and a Reflector's mirror image, tilted by summed waves.
// Painted: flat dabs squashed by view angle, reflections streaked vertically, a sun glint.
const RES = 0.5; // mirror picture resolution, of the screen's
// tuned by eye
const KNOBS = {
  size: 1.83, // point diameter, of its spacing
  r0: 0.16, // mirrored share looking straight down (Schlick r0, raised as games do)
  edge: 1.06, // Schlick exponent
  distort: 0.025, // reflection shift per unit of wave tilt (picture uv)
  flatness: 0.05, // flattest a dab gets edge-on
  smear: 0.0107, // vertical reflection streak (picture uv)
  reflectedR: 0.82,
  reflectedG: 0.88,
  reflectedB: 0.96,
  vary: 0.023, // per-dab lightness variation
  spread: 0, // per-dab size variation
  deepM: 15.5, // fully deep this far from shore
  glintSharp: 400,
  glint: 1.5,
  rough: 1, // wave steepness, of WAVES'
  pace: 0.35, // wave speed, of deep water's
};
const COLOURS = { shallow: '#4fc4b0', deep: '#135a80', glintColour: '#fff4d8' };
// [heading (radians from east), length (m), steepness (max slope)]
const WAVES = [
  [0.4, 11, 0.05],
  [1.9, 6.3, 0.045],
  [-1.0, 3.1, 0.04],
  [2.7, 1.7, 0.03],
];
const BLOCK_M = 20, // grid block sharing one spacing
  TILE_M = 200; // draw/cull tile

const time = { value: 0 };
export function tickWater(dt) {
  time.value += Math.min(dt, 0.1);
}

// one shared uniform per knob and colour
const knobs = {
  ...Object.fromEntries(Object.entries(KNOBS).map(([k, v]) => [k, { value: v }])),
  ...Object.fromEntries(
    Object.entries(COLOURS).map(([k, hex]) => [k, { value: new THREE.Color(hex) }]),
  ),
};

// metres from shore at (east, north), bilinear on the shore grid
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

// whether segment (a, b) crosses edge (p, q)
function crosses(ax, ay, bx, by, [px, py], [qx, qy]) {
  const side = (x0, y0, x1, y1, x, y) => (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0);
  return (
    side(ax, ay, bx, by, px, py) * side(ax, ay, bx, by, qx, qy) < 0 &&
    side(px, py, qx, qy, ax, ay) * side(px, py, qx, qy, bx, by) < 0
  );
}

// Jittered grid points inside rings (outlines and holes, even-odd) in (east, north), spaced per block.
// Inside test: the block centre's ray cast, flipped by each edge between it and the point.
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
  const local = new Map(); // edges per block
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
      // block centre: ray cast east against every edge
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

function pointsMaterial(mirror, matrix, halfHeight) {
  const sun = new THREE.Vector3(...SUN).normalize().toArray();
  const waves = WAVES.map(([heading, length, steep]) => {
    const k = (2 * Math.PI) / length,
      speed = Math.sqrt(9.81 / k); // deep water's: longer waves go faster
    return `s += vec2(${f(Math.cos(heading))}, ${f(Math.sin(heading))}) * ${f(steep)} * rough *
      cos(dot(p, vec2(${f(Math.cos(heading) * k)}, ${f(Math.sin(heading) * k)})) - time * pace * ${f(k * speed)});`;
  }).join('\n');
  const steepest = f(WAVES.reduce((a, w) => a + w[2], 0));
  return new THREE.ShaderMaterial({
    uniforms: {
      mirror: { value: mirror },
      matrix: { value: matrix },
      halfHeight,
      time,
      ...worldUniforms,
      ...knobs,
      ...THREE.UniformsUtils.clone(THREE.UniformsLib.fog),
    },
    fog: true,
    vertexShader: `
      #include <fog_pars_vertex>
      ${WORLD_VERTEX}
      uniform sampler2D mirror; uniform mat4 matrix; uniform float halfHeight, time;
      uniform float ${Object.keys(KNOBS).join(', ')};
      uniform vec3 ${Object.keys(COLOURS).join(', ')};
      attribute vec2 water; // metres from shore, spacing
      varying vec3 colour;
      varying float squash, glyphSeed;
      void main() {
        float dab = fract(sin(dot(position.xy, vec2(12.9898, 78.233))) * 43758.5453);
        glyphSeed = dab;
        vec4 world = modelMatrix * vec4(position, 1.);
        // summed wave slope (east, north)
        vec2 p = position.xy, s = vec2(0.);
        ${waves}
        vec3 n = normalize(vec3(-s.x, 1., s.y)), v = normalize(cameraPosition - world.xyz);
        float mirrored = r0 + (1. - r0) * pow(1. - max(dot(n, v), 0.), edge);
        vec4 at = matrix * vec4(position, 1.); // Reflector texture matrix, in its local frame
        float choppy = clamp(length(s) / ${steepest}, 0., 1.);
        vec2 shift = n.xz * distort + vec2(0., (fract(dab * 7.13) - .5) * 2. * smear * (.3 + .7 * choppy));
        vec3 seen = texture2D(mirror, at.xy / at.w + shift).rgb * vec3(reflectedR, reflectedG, reflectedB);
        vec3 own = mix(shallow, deep, smoothstep(0., deepM, water.x));
        colour = mix(own, seen, mirrored)
          + glintColour * pow(max(dot(reflect(-v, n), ${v3(sun)}), 0.), glintSharp) * glint;
        colour *= 1. + (dab - .5) * 2. * vary;
        squash = max(abs(v.y), flatness);
        float demoIn;
        if (shotAway(world.xyz)) demoIn = 0.;
        else world.xyz = demoed(world.xyz, dab, demoIn);
        vec4 mv = viewMatrix * world, mvPosition = mv;
        gl_Position = demoIn < .5 ? vec4(2., 2., 2., 1.) : projectionMatrix * mv;
        #include <fog_vertex>
        gl_PointSize = worldSize(size * (1. + (fract(dab * 3.71) - .5) * 2. * spread) * water.y
          * projectionMatrix[1][1] * halfHeight / -mv.z, dab, dab);
        if (gl_PointSize == 0.) gl_Position = vec4(2., 2., 2., 1.);
      }`,
    fragmentShader: `
      #include <fog_pars_fragment>
      varying vec3 colour;
      varying float squash, glyphSeed;
      ${WORLD_FRAGMENT}
      void main() {
        vec2 q = gl_PointCoord - .5;
        ${cutPoint('glyphSeed', 'length(vec2(q.x, q.y / squash)) > .5')}
        gl_FragColor = vec4(colour * pointShade, 1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }`,
  });
}

let mirroring = false; // one mirror render at a time; nested waters keep their last picture

// One level's water: an invisible Reflector over its shape, its points as children so
// they stay out of its own mirror picture.
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
  // shape is in east/north: rotate so up is y and north is -z
  water.rotateX(-Math.PI / 2);
  water.position.y = level;
  water.renderOrder = -10; // mirror picture before its points
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
    points.userData.ownMotion = true; // no style float
    water.add(points);
  }
  water.userData.points = grid.east.length;
  return water;
}

// water.json bodies as one Reflector per level (viewer frame); gapOf(east, north): point spacing there.
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
