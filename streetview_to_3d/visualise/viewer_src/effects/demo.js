import * as THREE from 'three';

// The demos: the whole world moved at once, as one rule every point follows
// -- the scene's points and the map's (points.js), the buildings' and the
// land's (blocks.js), the water's (water.js) -- each
// from where it stands (and its own seed), so it ends exactly back there.
//
// - rise: the world a flat disk at the scene's foot, spreading out from its
//   middle, then rising out of it to what it is, the middle first.
// - swirl gently: everything swung part way round the middle (SWING) and
//   back where it stands, as a breeze pushes and lets go -- nothing drawn
//   in or up, only a little lifted, a little scattered -- the middle first,
//   further out later, so the world winds a little and unwinds into itself.
const NEAR_M = 20, // how far counts as near: the middle's this much across goes first
  FAR_M = 1000, // ... and all of it by this far (the land's edge, terrain.RADIUS_M)
  SPREAD_S = 1.5, // rise: the disk out to FAR_M
  RISE_AT = 1, // ... the middle rising after this long
  RISE_LAG = 2.5, // ... the edge this much later
  RISE_S = 1.8, // ... each place rising in this long
  GENTLE_S = 5, // swirl gently: all of it
  SWING_S = 3, // ... each place out and back in this long
  SWING = Math.PI / 3, // ... this far round at most
  STAGGER = 0.6, // ... each point starting up to this much later
  GENTLE_LIFT_M = 0.5, // ... lifted this much at most
  SCATTER = 0.2; // ... off its way round this much at most (radians)
export const DEMOS = {
  rise: [1, RISE_AT + RISE_LAG + RISE_S],
  gentle: [2, GENTLE_S],
};

// shared by every shader that moves with them
export const demo = {
  demoKind: { value: 0 },
  demoT: { value: 0 },
  demoCentre: { value: new THREE.Vector3() },
};
const f = (x) => x.toFixed(4);

// GLSL, vertex or fragment. demoed(p, seed, shown): world point p where the
// demo has it now; seed its own (0..1). shown 0 where the world is not yet.
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
    // swirl gently: out and back (a: 0, up to 1 half way, 0 again)
    float e = smoothstep(0., 1., clamp((demoT - seed * ${f(STAGGER)} - demoReach(p) * ${f(GENTLE_S - SWING_S - STAGGER)}) / ${f(SWING_S)}, 0., 1.)),
      a = sin(3.14159265 * e),
      angle = a * (${f(SWING)} + (seed - .5) * 2. * ${f(SCATTER)}),
      c = cos(angle), s = sin(angle);
    vec2 xz = mat2(c, s, -s, c) * d.xz;
    return demoCentre + vec3(xz.x, d.y + a * ${f(GENTLE_LIFT_M)} * seed, xz.y);
  }
`;

let length = 0;
// play a demo (DEMOS) round centre, the scene's foot in its middle
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
