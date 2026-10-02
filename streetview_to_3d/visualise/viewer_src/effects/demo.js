import * as THREE from 'three';

// The demos: the whole world moved at once, as one rule every point follows
// -- the scene's points and the map's (points.js), the buildings' strokes
// (blocks.js), the water's (water.js), the land's corners (land.js) -- each
// from where it stands (and its own seed), so it ends exactly back there.
//
// - rise: the world a flat disk at the scene's foot, spreading out from its
//   middle, then rising out of it to what it is, the middle first.
// - swirl: everything drawn up into a turning funnel round the middle --
//   pulled in, lifted, the further out the higher -- then set down again,
//   each a whole number of turns round, so it lands where it was.
// - swirl gently: everything turned once round the middle where it stands --
//   nothing drawn in or up, only a little lifted, scattered, as leaves on the
//   wind -- the middle first, further out later, so the world winds into a
//   spiral and unwinds into itself.
const NEAR_M = 20, // how far counts as near: the middle's this much across goes first
  FAR_M = 1000, // ... and all of it by this far (the land's edge, terrain.RADIUS_M)
  SPREAD_S = 1.5, // rise: the disk out to FAR_M
  RISE_AT = 1, // ... the middle rising after this long
  RISE_LAG = 2.5, // ... the edge this much later
  RISE_S = 1.8, // ... each place rising in this long
  SWIRL_S = 7,
  STAGGER = 1.2, // swirl: each point starting up to this much later
  TURNS = 3, // ... the middle this many turns more than the edge's one
  PULL = 0.4, // ... drawn in to this much of how far out it is
  LIFT_M = 4, // ... lifted this much
  LIFT = 0.5, // ... and this much of how far out it is
  GENTLE_S = 8,
  TURN_S = 4, // swirl gently: each place turning once in this long
  GENTLE_LIFT_M = 1, // ... lifted this much at most
  SCATTER = 0.4; // ... off its way round this much at most (radians)
export const DEMOS = {
  rise: [1, RISE_AT + RISE_LAG + RISE_S],
  swirl: [2, SWIRL_S],
  gentle: [3, GENTLE_S],
};

// shared by every shader that moves with them
export const demo = {
  demoKind: { value: 0 },
  demoT: { value: 0 },
  demoCentre: { value: new THREE.Vector3() },
};
const f = (x) => x.toFixed(4);

// GLSL, vertex or fragment. demoed(p, seed, whole, shown): world point p
// where the demo has it now; seed its own (0..1); whole 0 for a mesh's
// corners, which must turn together (one turn, the land a sheet), 1 for a
// point. shown 0 where the world is not yet.
export const DEMO = `
  uniform float demoKind, demoT;
  uniform vec3 demoCentre;
  float demoReach(vec3 p) {
    return log(1. + length(p.xz - demoCentre.xz) / ${f(NEAR_M)}) / ${f(Math.log(1 + FAR_M / NEAR_M))};
  }
  float demoShown(vec3 p) {
    return demoKind > .5 && demoKind < 1.5 ? step(demoReach(p), demoT / ${f(SPREAD_S)}) : 1.;
  }
  vec3 demoed(vec3 p, float seed, float whole, out float shown) {
    shown = demoShown(p);
    if (demoKind < .5) return p;
    vec3 d = p - demoCentre;
    float r = length(d.xz);
    if (demoKind < 1.5) {
      float up = smoothstep(0., 1., clamp((demoT - ${f(RISE_AT)} - demoReach(p) * ${f(RISE_LAG)}) / ${f(RISE_S)}, 0., 1.));
      return vec3(p.x, demoCentre.y + d.y * up, p.z);
    }
    if (demoKind > 2.5) {
      float e = smoothstep(0., 1., clamp((demoT - seed * ${f(STAGGER / 2)} - demoReach(p) * ${f(GENTLE_S - TURN_S - STAGGER / 2)}) / ${f(TURN_S)}, 0., 1.)),
        a = sin(3.14159265 * e),
        angle = 6.2831853 * e + a * (seed - .5) * 2. * ${f(SCATTER)},
        c = cos(angle), s = sin(angle);
      vec2 xz = mat2(c, s, -s, c) * d.xz;
      return demoCentre + vec3(xz.x, d.y + a * ${f(GENTLE_LIFT_M)} * seed, xz.y);
    }
    float e = smoothstep(0., 1., clamp((demoT - seed * ${f(STAGGER)}) / ${f(SWIRL_S - STAGGER)}, 0., 1.)),
      a = sin(3.14159265 * e),
      turns = 1. + whole * floor(${f(TURNS)} / (1. + r / ${f(NEAR_M)}) + .5),
      angle = 6.2831853 * turns * e + a * (seed - .5) * 1.5,
      c = cos(angle), s = sin(angle);
    vec2 xz = mat2(c, s, -s, c) * d.xz * mix(1., ${f(PULL)}, a);
    return demoCentre + vec3(xz.x, d.y + a * (${f(LIFT_M)} + r * ${f(LIFT)}) * (.4 + .6 * seed), xz.y);
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
