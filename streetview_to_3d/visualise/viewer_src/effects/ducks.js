import * as THREE from 'three';
import { fleet, model, random } from '@viewer/effects/moving';

// A few ducks on the water nearest the scene (life.json's ducks,
// postprocess/life.py), built and drawn as the buildings are (moving.js),
// their real size: a mallard, 0.6 m long -- its body riding low on the
// water, its tail tipped up, its neck and head raised, its bill; a drake
// grey with a green head, a hen brown, by turns.
//
// Each paddles slowly round its home, never further than its reach (clear
// of the shore): toward a spot at random, its heading easing round (TURN),
// then still a while on the water (REST_S), then on to another, bobbing a
// little all the while.
const COUNT = 4,
  SPEED = 0.35, // m/s
  TURN = 1.2, // how fast its heading eases toward where it goes, per second
  REST_S = [2, 7],
  GAP = 0.02;
// what each surface is: life.json's duck colours
const BODY = 0,
  HEAD = 1,
  TAIL = 2,
  BILL = 3;
const PARTS = ['body', 'head', 'tail', 'bill'];

// its body's cross-sections, tail to breast: [x, half-width, top] (m), its foot just under the water
const SECTIONS = [
  [-0.3, 0.015, 0.17],
  [-0.2, 0.09, 0.14],
  [0, 0.13, 0.15],
  [0.14, 0.11, 0.14],
  [0.22, 0.04, 0.1],
];
const FOOT = -0.02;

// a duck's dabs, in its own frame (x ahead, y up, z right): model().dabs()'s
export function duckDabs() {
  const { tri, quad, dabs } = model(GAP);
  // a section's outline, round from one side's foot over its back to the other's
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
  // its breast, closed
  const front = ring(SECTIONS[SECTIONS.length - 1]),
    tip = [0.25, 0.04, 0];
  for (let k = 0; k < 4; k++) tri(front[k], tip, front[k + 1], BODY);
  // its neck up from the breast to its head, its head, its bill
  const box = (x0, x1, y0, y1, w, what) => {
    const c = [
      [x0, y0, -w],
      [x1, y0, -w],
      [x1, y1, -w],
      [x0, y1, -w],
      [x0, y0, w],
      [x1, y0, w],
      [x1, y1, w],
      [x0, y1, w],
    ];
    for (const [a, b, cc, d] of [
      [0, 1, 2, 3],
      [5, 4, 7, 6],
      [4, 0, 3, 7],
      [1, 5, 6, 2],
      [3, 2, 6, 7],
    ])
      quad(c[a], c[b], c[cc], c[d], what);
  };
  box(0.12, 0.2, 0.1, 0.27, 0.04, HEAD);
  box(0.12, 0.25, 0.25, 0.33, 0.045, HEAD);
  box(0.25, 0.31, 0.27, 0.3, 0.025, BILL);
  return dabs();
}

// life.json's ducks, drawn: a THREE.Points, paddling each tickMoving (null if none)
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
  // somewhere new within its reach of home
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
          // on as it faces: slower while it still turns
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
