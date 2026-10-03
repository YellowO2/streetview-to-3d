import * as THREE from 'three';
import { SUN, f, v3 } from '@viewer/util';

// Paint-style clouds as soft point dabs on a noise-shaped density field (no raymarching).
// The field tiles every FIELD_M around the camera; dabs near the moving surface show, drift
// with WIND, and are sorted back to front because depth can't separate them near the far plane.
const COVERAGE = 0.45; // share of the sky that is cloud
const LOOK = {
  size: 1, // dab size, of its built size
  light: 0.7, // lit vs shaded contrast
  soft: 0.7, // edge softness: 0 sharp, 1 blended into the sky
  life: 2.5, // surface motion speed
};
const FIELD_M = 12000,
  FADE_M = [3000, 5600], // clear to, gone by
  STEP_M = 45, // field grid; one dab per cell near the surface
  BASE_M = 1000, // flat cloud base height
  HIGH_M = 800, // tallest tops above the base
  WEATHER_M = 1500, // coverage noise scale
  BILLOW_M = [300, 110], // surface noise scales
  BILLOW = [0.25, 90], // surface noise strength: sides (of the pattern), top (m)
  SIDE_M = 600, // metres inside per unit of pattern depth
  BAND_M = [-50, 105], // dabs from this far outside the surface to this far in
  SHELL_M = 55, // dabs shown this deep below the current surface
  MOVE_M = 45, // surface motion amplitude
  DAB = 2, // dab width, of the grid
  SMEAR = 1.4, // dab width over height
  WIND = [8, -3], // m/s east, south
  SORT_S = 0.5, // re-sort at least this often
  SORT_M = 30, // or when the camera moves this far
  BUDGET_MS = 6; // dab building time per frame
const LIT = [1.0, 0.98, 0.94],
  SHADE = [0.6, 0.67, 0.8],
  HORIZON = [0.78, 0.94, 1.0], // paint sky (environment.js), low
  ZENITH = [0.13, 0.55, 1.0]; // and high

const sun = new THREE.Vector3(...SUN).normalize().toArray();

// integer hash in [0, 1)
function lattice(i, j, k) {
  let h = Math.imul(i, 0x27d4eb2d) ^ Math.imul(j, 0x165667b1) ^ Math.imul(k, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
// value noise in [0, 1], cell ~size, tiling every FIELD_M east and north
function noise(x, y, z, size, seed) {
  const n = Math.round(FIELD_M / size),
    cell = FIELD_M / n; // whole cells per field
  const fx = x / cell,
    fy = y / cell + seed * 7919,
    fz = z / cell;
  const i = Math.floor(fx),
    j = Math.floor(fy),
    k = Math.floor(fz);
  let u = fx - i,
    v = fy - j,
    w = fz - k;
  u = u * u * (3 - 2 * u);
  v = v * v * (3 - 2 * v);
  w = w * w * (3 - 2 * w);
  const i0 = ((i % n) + n) % n,
    i1 = (i0 + 1) % n,
    k0 = ((k % n) + n) % n,
    k1 = (k0 + 1) % n;
  const a = lattice(i0, j, k0) + (lattice(i1, j, k0) - lattice(i0, j, k0)) * u,
    b = lattice(i0, j + 1, k0) + (lattice(i1, j + 1, k0) - lattice(i0, j + 1, k0)) * u,
    c = lattice(i0, j, k1) + (lattice(i1, j, k1) - lattice(i0, j, k1)) * u,
    d = lattice(i0, j + 1, k1) + (lattice(i1, j + 1, k1) - lattice(i0, j + 1, k1)) * u;
  const near = a + (b - a) * v;
  return near + (c + (d - c) * v - near) * w;
}
// coverage pattern, [0, 1]
const weather = (x, z) =>
  0.6 * noise(x, 0, z, WEATHER_M, 1) +
  0.3 * noise(x, 0, z, WEATHER_M / 2, 2) +
  0.1 * noise(x, 0, z, WEATHER_M / 4, 3);
// surface noise, [-1, 1]
const billow = (x, y, z) =>
  2 * (0.7 * noise(x, y, z, BILLOW_M[0], 4) + 0.3 * noise(x, y, z, BILLOW_M[1], 5)) - 1;

// metres inside a cloud at (x, y, z); negative outside
export function inside(x, y, z, coverage) {
  const threshold = 0.72 - 0.3 * coverage;
  const deep = (weather(x, z) - threshold) / (1 - threshold);
  if (deep < -0.4 || y < BASE_M - 200) return -1e3;
  const b = billow(x, y, z);
  const side = (deep + BILLOW[0] * b) * SIDE_M,
    top = BASE_M + HIGH_M * Math.max(deep, 0) ** 0.7 + BILLOW[1] * b - y;
  return Math.min(side, top, y - BASE_M);
}

// Builds the dabs one grid row per yield; returns { position, facing (outward normal, depth) }.
export function* cloudRows(coverage = COVERAGE) {
  const n = Math.round(FIELD_M / STEP_M),
    layers = Math.ceil((HIGH_M + BILLOW[1] - BAND_M[0]) / STEP_M) + 1;
  const position = [],
    facing = [];
  for (let i = 0; i < n; i++, yield)
    for (let k = 0; k < n; k++) {
      const x = (i + 0.5) * STEP_M - FIELD_M / 2,
        z = (k + 0.5) * STEP_M - FIELD_M / 2;
      const threshold = 0.72 - 0.3 * coverage,
        deep = (weather(x, z) - threshold) / (1 - threshold);
      // skip columns no cloud can reach, whatever the noise (with margin for jitter)
      if ((deep + BILLOW[0]) * SIDE_M < BAND_M[0] - 0.05 * SIDE_M) continue;
      // highest possible top
      const roof =
        BASE_M + HIGH_M * Math.max(deep + 0.15, 0) ** 0.7 + BILLOW[1] - BAND_M[0] + STEP_M;
      // solid interior with no surface to dab
      const solid = deep - 0.15 - BILLOW[0] > BAND_M[1] / SIDE_M,
        hollow = [
          BASE_M + BAND_M[1] + STEP_M,
          BASE_M + HIGH_M * Math.max(deep - 0.15, 0) ** 0.7 - BILLOW[1] - BAND_M[1] - STEP_M,
        ];
      for (let l = 0; l < layers; l++) {
        // jitter off the grid
        const jx = (lattice(i, l, k) - 0.5) * STEP_M * 0.8,
          jy = (lattice(k, i + 1, l) - 0.5) * STEP_M * 0.8,
          jz = (lattice(l + 2, k, i) - 0.5) * STEP_M * 0.8;
        const px = x + jx,
          py = BASE_M + BAND_M[0] + l * STEP_M + jy,
          pz = z + jz;
        if (py > roof) break;
        if (solid && py > hollow[0] && py < hollow[1]) continue;
        const d = inside(px, py, pz, coverage);
        if (d < BAND_M[0] || d > BAND_M[1]) continue;
        // outward normal: down the gradient
        const e = 15;
        const g = new THREE.Vector3(
          d - inside(px + e, py, pz, coverage),
          d - inside(px, py + e, pz, coverage),
          d - inside(px, py, pz + e, coverage),
        );
        if (g.lengthSq() < 1e-9) g.set(0, 1, 0);
        g.normalize();
        position.push(px, py, pz);
        facing.push(g.x, g.y, g.z, d);
      }
    }
  return { position, facing };
}
const vertexShader = `
  ${Object.entries(LOOK)
    .map(([k, v]) => `const float ${k} = ${f(v)};`)
    .join('\n')}
  uniform float time, halfHeight;
  uniform vec2 drift; // wind offset (east, south), wrapped to FIELD_M
  attribute vec4 facing; // outward normal; depth inside when built
  varying vec3 colour;
  varying float alpha;
  const float FIELD = ${f(FIELD_M)}, TAU = 6.2832;
  void main() {
    // current surface: slow waves through the field (tiling with it)
    vec3 q = position * TAU / FIELD;
    float t = time * life;
    float move = .6 * sin(7. * q.x + 4. * q.z + position.y / 260. + t * .015)
      * sin(5. * q.z - 3. * q.x + t * .011)
      + .4 * sin(23. * q.x - 17. * q.z + position.y / 90. - t * .04);
    float deep = facing.w + ${f(MOVE_M)} * move;
    float here = smoothstep(0., 15., deep) * (1. - smoothstep(${f(SHELL_M - 15)}, ${f(SHELL_M)}, deep));
    // drift with the wind, wrapped around the camera
    vec2 xz = position.xz + drift;
    xz = cameraPosition.xz + mod(xz - cameraPosition.xz + FIELD * .5, FIELD) - FIELD * .5;
    vec3 p = vec3(xz.x, position.y, xz.y);
    vec4 mv = viewMatrix * vec4(p, 1.);
    gl_Position = projectionMatrix * mv;
    vec3 n = facing.xyz, toEye = normalize(cameraPosition - p);
    // silhouette edge: surface turning away from the eye
    float rim = (1. - smoothstep(.1, .7, dot(n, toEye))) * soft;
    float away = length(p.xz - cameraPosition.xz);
    gl_PointSize = ${f(DAB * STEP_M)} * size * (1. + .8 * rim) * here * step(-.5, dot(n, toEye))
      * (1. - smoothstep(${f(FADE_M[0])}, ${f(FADE_M[1])}, away))
      * projectionMatrix[1][1] * halfHeight / -mv.z;
    // lit toward the sun and higher up; shaded underneath
    float lit = clamp(.5 + .5 * dot(n, ${v3(sun)}), 0., 1.);
    float high = clamp((position.y - ${f(BASE_M)}) / ${f(HIGH_M * 0.7)}, 0., 1.);
    float shade = mix(.5, mix(lit, high, .35), light) * (n.y < -.6 ? .75 : 1.);
    float seed = fract(sin(dot(position, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
    colour = mix(${v3(SHADE)}, ${v3(LIT)}, shade) * (1. + (seed - .5) * .06);
    // edges and far clouds blend into the sky behind
    vec3 sky = mix(${v3(HORIZON)}, ${v3(ZENITH)}, smoothstep(0., .9, -toEye.y));
    colour = mix(colour, sky, .5 * rim);
    colour = mix(colour, ${v3(HORIZON)}, .7 * smoothstep(${f(FADE_M[0] * 0.4)}, ${f(FADE_M[1])}, away));
    alpha = mix(.85, .2, rim);
  }`;

const fragmentShader = `
  varying vec3 colour;
  varying float alpha;
  void main() {
    vec2 q = gl_PointCoord * 2. - 1.;
    q.y *= ${f(SMEAR)};
    float r = dot(q, q);
    if (r > 1.) discard;
    gl_FragColor = vec4(colour, alpha * (1. - smoothstep(.15, 1., r)));
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

const wrap = (x) => x - FIELD_M * Math.floor(x / FIELD_M + 0.5);

export function createClouds(parent) {
  const geometry = new THREE.BufferGeometry();
  const drift = { value: new THREE.Vector2() },
    halfHeight = { value: 1 };
  const material = new THREE.ShaderMaterial({
    uniforms: { time: { value: 0 }, halfHeight, drift },
    vertexShader,
    fragmentShader,
    transparent: true, // drawn after the world, so anything in front hides them
    depthWrite: false,
  });
  const clouds = new THREE.Points(geometry, material);
  clouds.name = 'Clouds';
  clouds.frustumCulled = false;
  clouds.visible = false;
  clouds.renderOrder = 10;
  const size = new THREE.Vector2();
  clouds.onBeforeRender = (renderer) => {
    renderer.getDrawingBufferSize(size);
    halfHeight.value = size.y / 2;
  };
  parent.add(clouds);

  // dabs are built over several frames when first shown
  let made = false,
    count = 0,
    order,
    keys,
    next;
  let building = null; // row generator in progress
  function make() {
    building ??= cloudRows();
    const until = performance.now() + BUDGET_MS;
    let step;
    do step = building.next();
    while (!step.done && performance.now() < until);
    if (!step.done) return false;
    const field = step.value;
    building = null;
    count = field.position.length / 3;
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(field.position, 3));
    geometry.setAttribute('facing', new THREE.Float32BufferAttribute(field.facing, 4));
    order = new Uint32Array(count).map((_, i) => i);
    keys = new Float32Array(count);
    next = new Uint32Array(count);
    geometry.setIndex(new THREE.BufferAttribute(order, 1));
    made = true;
    return true;
  }

  // sort back to front from the camera
  const sortedAt = new THREE.Vector3(Infinity, 0, 0),
    buckets = new Uint32Array(4097);
  let sortedTime = -Infinity;
  function sort(camera) {
    const p = geometry.getAttribute('position').array,
      d = drift.value,
      c = camera.position;
    let lo = Infinity,
      hi = -Infinity;
    for (let i = 0; i < count; i++) {
      const x = wrap(p[3 * i] + d.x - c.x),
        y = p[3 * i + 1] - c.y,
        z = wrap(p[3 * i + 2] + d.y - c.z);
      keys[i] = -(x * x + y * y + z * z); // farthest first
      lo = Math.min(lo, keys[i]);
      hi = Math.max(hi, keys[i]);
    }
    // approximate counting sort into 4096 bands
    buckets.fill(0);
    const scale = 4095 / Math.max(hi - lo, 1e-6);
    for (let i = 0; i < count; i++) buckets[1 + Math.floor((keys[i] - lo) * scale)]++;
    for (let b = 1; b < buckets.length; b++) buckets[b] += buckets[b - 1];
    for (let i = 0; i < count; i++) next[buckets[Math.floor((keys[i] - lo) * scale)]++] = i;
    order.set(next);
    geometry.index.needsUpdate = true;
  }

  return {
    update(visible, time, camera) {
      clouds.visible = visible && !!camera && camera.far > FIELD_M;
      if (!clouds.visible) return;
      if (!made && make()) sortedTime = -Infinity;
      clouds.visible = made;
      if (!clouds.visible) return;
      material.uniforms.time.value = time;
      drift.value.set(wrap(WIND[0] * time), wrap(WIND[1] * time));
      if (time - sortedTime > SORT_S || camera.position.distanceTo(sortedAt) > SORT_M) {
        sort(camera);
        sortedTime = time;
        sortedAt.copy(camera.position);
      }
    },
    dispose() {
      clouds.removeFromParent();
      geometry.dispose();
      material.dispose();
    },
  };
}
