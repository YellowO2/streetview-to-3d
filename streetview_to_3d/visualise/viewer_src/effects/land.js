import * as THREE from 'three';
import { GAPS } from '@viewer/effects/scatter';
import { tunable } from '@viewer/effects/tune-panel';
import { SUN } from '@viewer/effects/water';

// The land (land.ply, postprocess/terrain.py: triangles, its colours the
// satellite's and the panos') as one surface, painted: cut into patches
// fixed in the world, each one flat colour, a dab of paint. The patches
// fit together whole -- no gaps, nothing overlapping, nothing sticking out
// -- each the part of the land nearer its own middle (jittered on a grid)
// than any other's, its edges a little rough (ROUGH), as a brush leaves them.
//
// A patch is as wide as the world's points are spaced there (gapOf: size of
// it, never under MIN_GAP: near by the scene's own ground stands over it),
// one of GAPS; on a slope, which holds more land than its map, its grid finer
// as it faces up (down to FINE), so it stays as wide on the land; and longer
// round the slope than up it, the steeper the longer (stroke) -- as painters
// brush a hill.
//
// Its colour the land's at its middle -- the land's own colour there and
// its light -- found from the land's change across the triangle the pixel
// is in. Painted by light: lighter where it faces the sun, darker where it
// turns away (light, in steps if steps), seen through the air -- the further
// off, the more of the low sky's colour (haze, half of it by HAZE_M); broad
// patches (PATCH_M across) a little lighter or darker, half as much warmer
// or cooler (patches); each patch its own a little lighter or darker (vary).
// Pushed back a little in depth (polygonOffset), so what lies on it -- roads
// 15 cm up, the scene's ground 10 cm up -- wins even a kilometre off.
export const MIN_GAP = 0.3;
// the look's knobs: [value, lowest, highest] (a page opened with ?tune shows them: tune-panel.js)
export const KNOBS = {
  size: [1.15, 0.5, 6], // a patch across, of the points' spacing
  stroke: [1, 0, 3], // on a slope, a patch this much longer round it, of as steep as a wall
  light: [0.3, 0, 1.5], // facing the sun: 0 the colour as stored, 1 all of it lit
  steps: [0, 0, 8], // the light in this many steps (0: smooth)
  haze: [0.5, 0, 1], // the most of the sky's colour, far off
  patches: [0.08, 0, 0.4], // broad patches, how different
  vary: [0.1, 0, 0.4], // each patch, how different
};
// the rest, set
const FIXED = {
  HAZE_M: 800,
  PATCH_M: 60,
  FINE: 0.4, // a slope's grid finer as it faces up, down to this (as steep as 66 degrees)
  ROUGH: 0.15, // a patch's edge wanders this much of its width
};
const HAZE = new THREE.Color('#c8dcea'); // the low sky

const knobs = tunable('Land', KNOBS);
const vec = (v) => v.map((x) => x.toFixed(4)).join(',');
const sun = new THREE.Vector3(...SUN).normalize();
const g0 = GAPS[0].toFixed(4);

const vertexShader = `
  attribute float gap;
  varying vec3 colour, world, n;
  varying float spacing;
  void main() {
    colour = color;
    spacing = gap;
    n = normalize(mat3(modelMatrix) * normal);
    vec4 w = modelMatrix * vec4(position, 1.);
    world = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;

const fragmentShader = `
  uniform float ${Object.keys(KNOBS).join(', ')};
  ${Object.entries(FIXED)
    .map(([k, v]) => `const float ${k} = ${v.toFixed(4)};`)
    .join('\n')}
  varying vec3 colour, world, n;
  varying float spacing;
  float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  vec2 hash22(vec2 p) { return vec2(hash2(p), hash2(p + 19.19)); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3. - 2. * f);
    return mix(mix(hash2(i), hash2(i + vec2(1., 0.)), f.x),
               mix(hash2(i + vec2(0., 1.)), hash2(i + vec2(1., 1.)), f.x), f.y);
  }
  void main() {
    vec3 up = normalize(n);
    // the grid's step, one of GAPS: finer on a slope, as it faces up
    float want = max(spacing * size * max(up.y, FINE), ${g0});
    float k = ceil(log(want / ${g0}) / log(1.25) - .001);
    float grid = ${g0} * pow(1.25, k);
    // on the map, round the slope and up it
    float steep = length(up.xz);
    vec2 around = steep > 1e-3 ? vec2(-up.z, up.x) / steep : vec2(1., 0.),
      rise = vec2(-around.y, around.x);
    float longer = 1. + stroke * steep;
    // its patch: the nearest middle (round the slope counted shorter, so
    // patches are longer that way), the edge a little rough
    vec2 p = world.xz / grid;
    p += (vec2(vnoise(p * 2.3), vnoise(p * 2.3 + 7.1)) - .5) * 2. * ROUGH;
    vec2 cell = floor(p), best = cell;
    float nearest = 1e9;
    for (int i = -2; i <= 2; i++)
      for (int j = -2; j <= 2; j++) {
        vec2 c = cell + vec2(float(i), float(j));
        vec2 d = c + hash22(c + k * 7.31) - p;
        float a = dot(d, around) / longer, b = dot(d, rise);
        if (a * a + b * b < nearest) { nearest = a * a + b * b; best = c; }
      }
    vec2 middle = (best + hash22(best + k * 7.31)) * grid;
    // the land there, its colour and its facing: from how they change over
    // this triangle (on the map, a pixel's step east, north)
    vec2 ex = dFdx(world.xz), ey = dFdy(world.xz);
    float det = ex.x * ey.y - ex.y * ey.x;
    vec2 d = middle - world.xz;
    vec2 s = abs(det) > 1e-12 ? vec2(ey.y * d.x - ey.x * d.y, -ex.y * d.x + ex.x * d.y) / det : vec2(0.);
    vec3 base = clamp(colour + dFdx(colour) * s.x + dFdy(colour) * s.y, 0., 1.),
      facing = normalize(up + dFdx(up) * s.x + dFdy(up) * s.y);
    // painted: lit, patched, its own shade, hazed
    float lit = max(dot(facing, vec3(${vec(sun.toArray())})), 0.) / ${sun.y.toFixed(4)}; // 1 on flat ground
    if (steps > .5) lit = floor(lit * steps + .5) / steps;
    vec3 c = base * mix(1., lit, light);
    vec2 q = middle / PATCH_M;
    c *= 1. + (vnoise(q) - .5) * 2. * patches;
    float t = (vnoise(q * .7 + 17.3) - .5) * patches;
    c *= vec3(1. + t, 1., 1. - t);
    c *= 1. + (hash2(best + k * 3.17) - .5) * 2. * vary;
    float away = length(vec3(middle.x, world.y, middle.y) - cameraPosition);
    gl_FragColor = vec4(mix(c, vec3(${vec(HAZE.toArray())}), haze * (1. - exp2(-away / HAZE_M))), 1.);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;

// The land's triangles (the viewer's frame) as its painted surface; gapOf(x,
// z): the world's points' spacing there.
export function landSurface(geometry, gapOf) {
  geometry.computeVertexNormals();
  // a triangle wound either way: every normal up
  const nor = geometry.getAttribute('normal');
  for (let i = 0; i < nor.count; i++)
    if (nor.getY(i) < 0) nor.setXYZ(i, -nor.getX(i), -nor.getY(i), -nor.getZ(i));
  const p = geometry.getAttribute('position');
  const gap = new Float32Array(p.count).map((_, i) =>
    Math.max(MIN_GAP, gapOf(p.getX(i), p.getZ(i))),
  );
  geometry.setAttribute('gap', new THREE.Float32BufferAttribute(gap, 1));
  if (!geometry.getAttribute('color'))
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(new Float32Array(3 * p.count).fill(0.5), 3),
    );
  return new THREE.Mesh(
    geometry,
    new THREE.ShaderMaterial({
      uniforms: { ...knobs },
      vertexColors: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 4,
      vertexShader,
      fragmentShader,
    }),
  );
}
