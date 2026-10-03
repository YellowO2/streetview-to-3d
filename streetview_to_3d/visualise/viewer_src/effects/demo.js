import * as THREE from 'three';
import { f } from '@viewer/effects/util';

// Demos that move every world point by one rule from where it stands, ending back in place.
// rise: a flat disk spreading from the middle, then rising, middle first.
// gentle: everything swung part way round the middle and back, slightly lifted and scattered.
const NEAR_M = 20, // the middle this wide goes first
  FAR_M = 1000, // all of it by this far (terrain.RADIUS_M)
  SPREAD_S = 1.5, // rise: disk reaches FAR_M
  RISE_AT = 1, // rise: middle starts rising
  RISE_LAG = 2.5, // rise: edge starts this much later
  RISE_S = 1.8, // rise: each place rises in this long
  GENTLE_S = 5, // gentle: total length
  SWING_S = 3, // gentle: each place out and back
  SWING = Math.PI / 3, // gentle: most turn
  STAGGER = 0.6, // gentle: most per-point start delay (s)
  GENTLE_LIFT_M = 0.5, // gentle: most lift
  SCATTER = 0.2; // gentle: most extra turn per point (radians)
export const DEMOS = {
  rise: [1, RISE_AT + RISE_LAG + RISE_S],
  gentle: [2, GENTLE_S],
};

export const demo = {
  demoKind: { value: 0 },
  demoT: { value: 0 },
  demoCentre: { value: new THREE.Vector3() },
};

// GLSL: demoed(p, seed, shown) is world point p where the demo has it now; shown is 0 until it appears.
export const DEMO = `
  uniform float demoKind, demoT;
  uniform vec3 demoCentre;
  float demoReach(vec3 p) {
    return log(1. + length(p.xz - demoCentre.xz) / ${f(NEAR_M)}) / ${f(Math.log(1 + FAR_M / NEAR_M))};
  }
  vec3 demoed(vec3 p, float seed, out float shown) {
    shown = demoKind > .5 && demoKind < 1.5 ? step(demoReach(p), demoT / ${f(SPREAD_S)}) : 1.;
    if (demoKind < .5) return p;
    vec3 d = p - demoCentre;
    if (demoKind < 1.5) {
      float up = smoothstep(0., 1., clamp((demoT - ${f(RISE_AT)} - demoReach(p) * ${f(RISE_LAG)}) / ${f(RISE_S)}, 0., 1.));
      return vec3(p.x, demoCentre.y + d.y * up, p.z);
    }
    // gentle: a rises 0 to 1 and back to 0
    float e = smoothstep(0., 1., clamp((demoT - seed * ${f(STAGGER)} - demoReach(p) * ${f(GENTLE_S - SWING_S - STAGGER)}) / ${f(SWING_S)}, 0., 1.)),
      a = sin(3.14159265 * e),
      angle = a * (${f(SWING)} + (seed - .5) * 2. * ${f(SCATTER)}),
      c = cos(angle), s = sin(angle);
    vec2 xz = mat2(c, s, -s, c) * d.xz;
    return demoCentre + vec3(xz.x, d.y + a * ${f(GENTLE_LIFT_M)} * seed, xz.y);
  }
`;

let length = 0;
// play a demo (DEMOS) around centre, the scene's foot
export function playDemo(name, centre) {
  if (!(name in DEMOS)) return;
  [demo.demoKind.value, length] = DEMOS[name];
  demo.demoT.value = 0;
  demo.demoCentre.value.copy(centre);
}
export function stopDemo() {
  demo.demoKind.value = 0;
}
export function tickDemo(dt) {
  if (!demo.demoKind.value) return;
  demo.demoT.value += Math.min(dt, 0.1);
  if (demo.demoT.value >= length) stopDemo();
}
