import * as THREE from 'three';
import { fleet, model, random } from '@viewer/life/moving';

// Ducks (life.json): 0.6 m mallards built of dabs (moving.js) paddling between random spots
// within reach of their home, resting between, bobbing slightly.
const COUNT = 4, // per home
  SPEED = 0.35, // m/s
  TURN = 1.2, // heading easing rate (1/s)
  REST_S = [2, 7],
  GAP = 0.02; // dab spacing (m)
// part indices into life.json's duck colours
const BODY = 0,
  HEAD = 1,
  TAIL = 2,
  BILL = 3;
const PARTS = ['body', 'head', 'tail', 'bill'];

// body cross-sections, tail to breast: [x, half-width, top] (m); bottom just under the water
const SECTIONS = [
  [-0.3, 0.015, 0.17],
  [-0.2, 0.09, 0.14],
  [0, 0.13, 0.15],
  [0.14, 0.11, 0.14],
  [0.22, 0.04, 0.1],
];
const FOOT = -0.02;

// a duck's dabs in its local frame (x ahead, y up, z right)
export function duckDabs() {
  const { tri, quad, box, dabs } = model(GAP);
  // a section's outline from one side's bottom over the back to the other's
  const ring = ([x, w, top]) => [
    [x, FOOT, -w],
    [x, top * 0.65, -w],
    [x, top, 0],
    [x, top * 0.65, w],
    [x, FOOT, w],
  ];
  SECTIONS.slice(1).forEach((s1, i) => {
    const a = ring(SECTIONS[i]),
      b = ring(s1);
    const w = i === 0 ? TAIL : BODY;
    for (let k = 0; k < 4; k++) quad(a[k], b[k], b[k + 1], a[k + 1], w);
  });
  // close the breast
  const front = ring(SECTIONS[SECTIONS.length - 1]),
    tip = [0.25, 0.04, 0];
  for (let k = 0; k < 4; k++) tri(front[k], tip, front[k + 1], BODY);
  // neck, head, bill
  box([0.12, 0.2], [0.1, 0.27], [-0.04, 0.04], HEAD);
  box([0.12, 0.25], [0.25, 0.33], [-0.045, 0.045], HEAD);
  box([0.25, 0.31], [0.27, 0.3], [-0.025, 0.025], BILL);
  return dabs();
}

// life.json's ducks as a THREE.Points, paddling each tickMoving (null if none)
export function duckPoints(data) {
  const homes = data.homes.filter((h) => h.reach > 0);
  if (!homes.length) return null;
  const rand = random(Math.round(homes[0].at[0] * 31 + homes[0].at[1] * 17));
  const ducks = homes.flatMap(({ level, at: [e, n], reach }) =>
    Array.from({ length: COUNT }, () => {
      const home = new THREE.Vector3(e, level, -n);
      const duck = { home, reach, at: home.clone(), to: home.clone(), heading: null, rest: 0 };
      aim(duck);
      duck.at.copy(duck.to);
      aim(duck);
      return duck;
    }),
  );
  // pick a new target within reach of home
  function aim(duck) {
    const r = duck.reach * Math.sqrt(rand()),
      a = rand() * 2 * Math.PI;
    duck.to.copy(duck.home).add(new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)));
  }
  const ahead = new THREE.Vector3(),
    up = new THREE.Vector3(0, 1, 0),
    right = new THREE.Vector3(),
    here = new THREE.Vector3();
  let t = 0;
  const { points, put, step } = fleet(
    duckDabs(),
    ducks.length,
    (k) => PARTS.map((p) => data.colours[k % data.colours.length][p]),
    rand,
    (dt) => {
      t += dt;
      ducks.forEach((duck, k) => {
        ahead.subVectors(duck.to, duck.at).setY(0);
        const left = ahead.length();
        if (duck.rest > 0) duck.rest -= dt;
        else if (left < 0.1) {
          duck.rest = REST_S[0] + rand() * (REST_S[1] - REST_S[0]);
          aim(duck);
        } else {
          ahead.normalize();
          if (!duck.heading) duck.heading = ahead.clone();
          else duck.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
          // move along the heading, slower while turning
          duck.at.addScaledVector(duck.heading, SPEED * dt * Math.max(0, duck.heading.dot(ahead)));
        }
        const h = duck.heading || ahead.set(1, 0, 0);
        right.crossVectors(h, up).normalize();
        here.copy(duck.at).setY(duck.home.y + 0.01 * Math.sin(1.7 * t + k));
        put(k, here, h, up, right);
      });
    },
  );
  step(0);
  return points;
}
