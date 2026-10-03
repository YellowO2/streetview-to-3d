import * as THREE from 'three';
import { cut, fleet, model, random } from '@viewer/effects/moving';

// Birds (life.json) crossing the scene now and then, built of dabs (moving.js): one small group
// of one kind at a time on a gently bowed, banked course, then an empty sky for WAIT_S.
// Each bird flaps for a while, then glides, on its own timing.
export const KINDS = {
  // span, length (m); speed (m/s); flaps per second; [flapping s, cycle s];
  // group size and height above ground (m), each [min, max]
  gull: {
    span: 1.25,
    length: 0.55,
    speed: 8,
    rate: 2.6,
    flap: [1.2, 5],
    count: [2, 3],
    height: [16, 30],
  },
  pigeon: {
    span: 0.66,
    length: 0.33,
    speed: 10,
    rate: 5,
    flap: [1.6, 4],
    count: [3, 4],
    height: [10, 22],
  },
  swallow: {
    span: 0.32,
    length: 0.17,
    speed: 12,
    rate: 9,
    flap: [0.5, 1.3],
    count: [2, 4],
    height: [4, 10],
    fork: true,
  },
};
const REACH_M = 120, // groups enter and leave this far from the middle
  FADE_M = 30, // growing in over the first and shrinking out over the last of this
  ACROSS_M = 25, // most a course passes to the side of the middle
  BOW_M = 15, // most sideways bow of a course
  WAIT_S = [4, 15], // empty sky between groups
  FIRST_S = [2, 5], // and before the first
  BEAT = 0.55, // wing stroke either way (radians)
  HELD = 0.1, // wings raised while gliding (radians)
  G = 9.8;
// part indices into life.json's bird colours
const BODY = 0,
  WING = 1,
  TIP = 2,
  TAIL = 3; // wing colour, not flapping

// a bird's dabs in its local frame (x ahead, y up, z right), plus side (-1/1 for wing dabs,
// 0 otherwise) and shoulder (wing hinge, m from the middle)
export function birdDabs({ span, length, fork = false }) {
  const { tri, flat, dabs } = model(span / 30);
  const shoulder = length * 0.1;
  const r = length * 0.14,
    nose = [length * 0.5, 0, 0],
    tail = [-length * 0.28, 0, 0];
  // body: a four-sided spindle
  const ring = [
    [length * 0.08, r, 0],
    [length * 0.08, 0, r],
    [length * 0.08, -r * 0.8, 0],
    [length * 0.08, 0, -r],
  ];
  ring.forEach((a, i) => {
    const b = ring[(i + 1) % 4];
    tri(nose, a, b, BODY);
    tri(a, tail, b, BODY);
  });
  const level = (p) => [p[0], 0, p[1]]; // [x, z] -> flat at mid height
  // tail: fanned, or long and forked for a swallow
  flat(
    fork
      ? [
          [-length * 0.24, -r * 0.6],
          [-length * 0.68, -r * 2],
          [-length * 0.44, 0],
          [-length * 0.68, r * 2],
          [-length * 0.24, r * 0.6],
        ]
      : [
          [-length * 0.24, -r * 0.6],
          [-length * 0.5, -r * 1.6],
          [-length * 0.5, r * 1.6],
          [-length * 0.24, r * 0.6],
        ],
    level,
    TAIL,
  );
  // swept-back wings, the outer part (TIP) a darker colour
  const s = span / 2;
  for (const side of [-1, 1]) {
    // swallow: a narrow sickle
    const wing = fork
      ? [
          [length * 0.12, shoulder],
          [length * 0.14, s * 0.45],
          [-length * 0.5, s],
          [-length * 0.2, s * 0.5],
          [-length * 0.1, shoulder],
        ]
      : [
          [length * 0.12, shoulder],
          [length * 0.16, s * 0.45],
          [-length * 0.3, s],
          [-length * 0.34, s * 0.55],
          [-length * 0.16, shoulder],
        ];
    const inner = cut(wing, s * 0.78, -1, 1),
      outer = cut(wing, s * 0.78, 1, 1);
    const at = (p) => [p[0], 0, side * p[1]];
    flat(inner, at, WING);
    flat(outer, at, TIP);
  }
  const made = dabs();
  made.side = made.what.map((w, i) =>
    w === WING || w === TIP ? Math.sign(made.centre[3 * i + 2]) : 0,
  );
  made.shoulder = shoulder;
  return made;
}

// wing angle up (radians) at t (s) for a bird of kind, offset by off (s)
export function wingAngle(kind, t, off) {
  const [flapS, everyS] = kind.flap;
  const into = (((t + off) % everyS) + everyS) % everyS;
  return into < flapS ? HELD + BEAT * Math.sin(2 * Math.PI * kind.rate * into) : HELD;
}

// life.json's birds as a THREE.Group with one fleet per kind (its largest group); one group flies at a time
export function birdPoints(data) {
  const [e, n, h] = data.centre;
  const centre = new THREE.Vector3(e, h, -n);
  const rand = random(Math.round(e * 7 + n * 13));
  const between = ([lo, hi]) => lo + rand() * (hi - lo);
  const group = new THREE.Group();
  const kinds = data.flocks.filter(({ kind }) => KINDS[kind]);
  const here = new THREE.Vector3(),
    ahead = new THREE.Vector3(),
    up = new THREE.Vector3(),
    right = new THREE.Vector3(),
    Y = new THREE.Vector3(0, 1, 0);
  // the group in the air: { of (its fleet), birds, start, dir, side, bow, length, u }
  let passing = null,
    wait = between(FIRST_S),
    t = 0;

  // one fleet per kind; the first one's tick flies the current group
  const flocks = kinds.map(({ kind, colours }, f) => {
    const k = KINDS[kind],
      shape = birdDabs(k);
    const { body, wing, tip } = colours;
    const tick = f === 0 ? fly : () => {};
    return {
      kind: k,
      shape,
      ...fleet(shape, k.count[1], () => [body, wing, tip, wing], rand, tick),
    };
  });

  // start a random kind's group on a course across the scene
  function setOff() {
    const of = flocks[Math.floor(rand() * flocks.length)];
    const k = of.kind,
      a = rand() * 2 * Math.PI;
    const dir = new THREE.Vector3(Math.cos(a), 0, Math.sin(a)),
      side = new THREE.Vector3().crossVectors(dir, Y);
    const start = centre
      .clone()
      .addScaledVector(dir, -REACH_M)
      .addScaledVector(side, (rand() * 2 - 1) * ACROSS_M);
    const height = between(k.height),
      n = Math.round(between([k.count[0], k.count[1] + 0.99]) - 0.49);
    const birds = Array.from({ length: n }, (_, j) => ({
      back: j * k.span * 3 + rand() * k.span, // loose staggered line
      aside: (j % 2 ? 1 : -1) * Math.ceil(j / 2) * k.span * 2.5 + (rand() - 0.5) * k.span,
      height: height + (rand() - 0.5) * 3,
      off: rand() * 10, // flap timing offset
    }));
    passing = {
      of,
      birds,
      start,
      dir,
      side,
      bow: (rand() * 2 - 1) * BOW_M,
      length: 2 * REACH_M,
      u: 0,
    };
  }

  function fly(dt) {
    t += dt;
    if (!passing) {
      wait -= dt;
      if (wait <= 0) setOff();
      return;
    }
    const { of, birds, start, dir, side, bow, length } = passing;
    const k = of.kind,
      shoulder = of.shape.shoulder;
    passing.u += k.speed * dt;
    let gone = true;
    birds.forEach((bird, j) => {
      const u = passing.u - bird.back,
        w = Math.PI / length;
      // along the bowed course, heading along it, banked into the bow
      here
        .copy(start)
        .addScaledVector(dir, u)
        .addScaledVector(side, bow * Math.sin(w * u) + bird.aside);
      here.y = centre.y + bird.height + 0.8 * Math.sin(t * 0.4 + bird.off);
      ahead
        .copy(dir)
        .addScaledVector(side, bow * w * Math.cos(w * u))
        .normalize();
      const bank = Math.atan((k.speed ** 2 * bow * w * w * Math.sin(w * u)) / G);
      up.copy(Y).multiplyScalar(Math.cos(bank)).addScaledVector(side, Math.sin(bank));
      right.crossVectors(ahead, up).normalize();
      up.crossVectors(right, ahead).normalize();
      const away = Math.hypot(here.x - centre.x, here.z - centre.z);
      const shown = u < 0 || u > length ? 0 : Math.min(1, Math.max(0, (REACH_M - away) / FADE_M));
      if (u < length) gone = false;
      const angle = wingAngle(k, t, bird.off),
        c = Math.cos(angle),
        s = Math.sin(angle);
      // wing dabs rotated about the shoulder
      of.put(j, here, ahead, up, right, shown, (i, p, q) => {
        const sd = of.shape.side[i];
        if (!sd) return;
        const dz = p[2] - sd * shoulder,
          dy = p[1];
        p[1] = sd * dz * s + dy * c;
        p[2] = sd * shoulder + dz * c - sd * dy * s;
        const nz = q[2],
          ny = q[1];
        q[1] = sd * nz * s + ny * c;
        q[2] = nz * c - sd * ny * s;
      });
    });
    if (gone) {
      passing = null;
      wait = between(WAIT_S);
    }
  }

  // all hidden until their group sets off
  for (const f of flocks) {
    for (let j = 0; j < f.kind.count[1]; j++) f.put(j, here, Y, Y, Y, 0);
    group.add(f.points);
  }
  return group;
}
