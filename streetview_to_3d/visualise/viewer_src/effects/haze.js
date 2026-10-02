import * as THREE from 'three';
import { tunable } from '@viewer/effects/tune-panel';

// The world's points (blocks.js: buildings', the land's; the bird's) seen
// through the air: the further off, the more of the low sky's colour (half
// of haze by HAZE_M); and the noise their uneven edges are cut by.
export const HAZE_M = 800;
const HAZE = new THREE.Color('#c8dcea'); // the low sky
// how much of it, at most, far off (a page opened with ?tune shows it: tune-panel.js)
export const { haze } = tunable('Haze', { haze: [0.5, 0, 1] });

// GLSL, for a fragment shader
export const HAZED = `
  float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3. - 2. * f);
    return mix(mix(hash2(i), hash2(i + vec2(1., 0.)), f.x),
               mix(hash2(i + vec2(0., 1.)), hash2(i + vec2(1., 1.)), f.x), f.y);
  }
  // c seen from away metres off
  vec3 hazed(vec3 c, float away, float haze) {
    return mix(c, vec3(${HAZE.toArray()
      .map((x) => x.toFixed(4))
      .join(',')}), haze * (1. - exp2(-away / ${HAZE_M.toFixed(1)})));
  }
`;
