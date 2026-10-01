import * as THREE from 'three';

// A surface painted in patches, each a dab of paint (land.js, blocks.js): the
// surface cut into the parts nearer each of points on a grid fixed on it
// (each jittered from its cell's centre) than any other's, each one flat
// colour -- they fit together whole, no gaps, nothing overlapping, nothing
// sticking out -- seen through the air: the further off, the more of the low
// sky's colour (half of haze by HAZE_M).
export const HAZE_M = 800;
const HAZE = new THREE.Color('#c8dcea'); // the low sky

// GLSL, for a fragment shader
export const PATCHES = `
  float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  vec2 hash22(vec2 p) { return vec2(hash2(p), hash2(p + 19.19)); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3. - 2. * f);
    return mix(mix(hash2(i), hash2(i + vec2(1., 0.)), f.x),
               mix(hash2(i + vec2(0., 1.)), hash2(i + vec2(1., 1.)), f.x), f.y);
  }
  // cell c's middle (cells 1 across): off its centre up to jitter either way (.5: anywhere in it)
  vec2 middleOf(vec2 c, float seed, float jitter) {
    return c + .5 + (hash22(c + seed * 7.31) - .5) * 2. * jitter;
  }
  // the cell whose middle is nearest p (in cells), along a counted longer
  // times shorter (patches that much longer that way), the edges wandering
  // rough of a cell, as a brush leaves them
  vec2 patchAt(vec2 p, vec2 a, float longer, float seed, float jitter, float rough) {
    p += (vec2(vnoise(p * 2.3), vnoise(p * 2.3 + 7.1)) - .5) * 2. * rough;
    vec2 b = vec2(-a.y, a.x), cell = floor(p), best = cell;
    float nearest = 1e9;
    for (int i = -2; i <= 2; i++)
      for (int j = -2; j <= 2; j++) {
        vec2 c = cell + vec2(float(i), float(j));
        vec2 d = middleOf(c, seed, jitter) - p;
        float x = dot(d, a) / longer, y = dot(d, b);
        if (x * x + y * y < nearest) { nearest = x * x + y * y; best = c; }
      }
    return best;
  }
  // c seen from away metres off
  vec3 hazed(vec3 c, float away, float haze) {
    return mix(c, vec3(${HAZE.toArray()
      .map((x) => x.toFixed(4))
      .join(',')}), haze * (1. - exp2(-away / ${HAZE_M.toFixed(1)})));
  }
`;
