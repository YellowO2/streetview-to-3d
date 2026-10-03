import * as THREE from 'three';
import { tunable } from '@viewer/ui/tune-panel';
import { v3 } from '@viewer/util';

// Aerial haze for the far dabs (blocks.js, bird-paint.js): low-sky colour, half strength by HAZE_M.
const HAZE_M = 800;
const HAZE = new THREE.Color('#c8dcea'); // low sky
// most haze far off (adjustable with ?tune, tune-panel.js)
export const { haze } = tunable('Haze', { haze: [0.5, 0, 1] });

// GLSL, fragment: value noise (vnoise) and hazed(colour, metres away, haze)
export const HAZED = `
  float hash2(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
  float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3. - 2. * f);
    return mix(mix(hash2(i), hash2(i + vec2(1., 0.)), f.x),
               mix(hash2(i + vec2(0., 1.)), hash2(i + vec2(1., 1.)), f.x), f.y);
  }
  vec3 hazed(vec3 c, float away, float haze) {
    return mix(c, ${v3(HAZE.toArray())}, haze * (1. - exp2(-away / ${HAZE_M.toFixed(1)})));
  }
`;
