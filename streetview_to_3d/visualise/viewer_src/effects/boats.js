import * as THREE from 'three';
import { fleet, model, random } from '@viewer/effects/moving';

// A boat or two cruising the water near the scene (life.json's boats,
// postprocess/life.py), built and drawn as the buildings are (moving.js),
// its real size: a small motorboat, 5 m long -- its hull narrowing to its
// bow, its sheer rising toward it, its stem raked, a deck, a console
// amidships with a windscreen.
//
// Each goes round its course (one closed, kept off the shore) at a gentle
// SPEED, its heading easing round the bends (TURN), rocking a little on the
// water -- rolling, pitching, rising and falling, each in its own time. One
// to a course, at most MAX_BOATS.
const MAX_BOATS = 3,
  SPEED = 3, // m/s
  LOOK_M = 4, // its heading from the course this far behind and ahead
  TURN = 1.5, // how fast its heading eases onto the course's, per second
  GAP = 0.1,
  ROLL = 0.035, // radians
  PITCH = 0.02,
  HEAVE = 0.05; // m
// its hull's half-beam (m) along it, stern to bow
const PLAN = [
  [-2.4, 0.9],
  [-1.0, 1.0],
  [0.6, 0.92],
  [1.6, 0.6],
  [2.2, 0.25],
  [2.5, 0],
];
const sheer = (x) => 0.55 + 0.32 * Math.max(0, (x + 0.5) / 3) ** 2, // its top edge, rising to its bow
  KEEL = -0.1, // a little under the water, so no gap shows at its line
  RAKE = 0.4, // its stem leaning forward: at the water its bow this much further back
  DECK = 0.5;
// where a point of its plan meets the water: drawn back toward the stern near its bow
const foot = (x) => x - RAKE * Math.max(0, (x - 1.6) / 0.9);
// what each surface is: life.json's boat colours
const HULL = 0,
  DECK_W = 1,
  CABIN = 2,
  GLASS = 3;

// a boat's dabs, in its own frame (x ahead, y up, z right): model().dabs()'s
export function boatDabs() {
  const { quad, flat, dabs } = model(GAP);
  for (const side of [-1, 1])
    PLAN.slice(1).forEach(([x1, w1], i) => {
      const [x0, w0] = PLAN[i];
      quad(
        [foot(x0), KEEL, side * w0],
        [foot(x1), KEEL, side * w1],
        [x1, sheer(x1), side * w1],
        [x0, sheer(x0), side * w0],
        HULL,
      );
    });
  const [sx, sw] = PLAN[0];
  quad([sx, KEEL, -sw], [sx, KEEL, sw], [sx, sheer(sx), sw], [sx, sheer(sx), -sw], HULL); // its transom
  flat(
    [
      ...PLAN.map(([x, w]) => [x, w]),
      ...PLAN.slice(0, -1)
        .reverse()
        .map(([x, w]) => [x, -w]),
    ],
    (p) => [p[0], DECK, p[1]],
    DECK_W,
  );
  // the console: a box amidships, its windscreen leaning back over its front
  const [x0, x1, w, top] = [-0.5, 0.5, 0.5, 1.15];
  for (const side of [-1, 1])
    quad(
      [x0, DECK, side * w],
      [x1, DECK, side * w],
      [x1, top, side * w],
      [x0, top, side * w],
      CABIN,
    );
  quad([x0, DECK, -w], [x0, DECK, w], [x0, top, w], [x0, top, -w], CABIN);
  quad([x1, DECK, -w], [x1, DECK, w], [x1, top, w], [x1, top, -w], CABIN);
  quad([x0, top, -w], [x1, top, -w], [x1, top, w], [x0, top, w], CABIN);
  quad([x1, top, -w], [x1, top, w], [x1 - 0.2, top + 0.32, w], [x1 - 0.2, top + 0.32, -w], GLASS);
  return dabs();
}

// a course ([[e, n], ...], closed) in the viewer's frame at level: { points, at, length }
function course({ level, points }) {
  const p = points.map(([e, n]) => new THREE.Vector3(e, level, -n));
  p.push(p[0].clone());
  const at = [0];
  for (let i = 1; i < p.length; i++) at.push(at[i - 1] + p[i].distanceTo(p[i - 1]));
  return { points: p, at, length: at[at.length - 1] };
}

// where along a course (metres, round and round) is
function along(c, s, out) {
  s = ((s % c.length) + c.length) % c.length;
  let i = 1;
  while (i < c.at.length - 1 && c.at[i] < s) i++;
  const t = (s - c.at[i - 1]) / Math.max(c.at[i] - c.at[i - 1], 1e-6);
  return out.lerpVectors(c.points[i - 1], c.points[i], t);
}

// life.json's boats, drawn: a THREE.Points, cruising each tickMoving (null if none)
export function boatPoints(data) {
  const courses = data.courses
    .filter((c) => c.points.length >= 3)
    .map(course)
    .slice(0, MAX_BOATS);
  if (!courses.length) return null;
  const rand = random(Math.round(courses[0].length));
  const boats = courses.map((c) => ({
    course: c,
    s: rand() * c.length,
    way: rand() < 0.5 ? 1 : -1,
    off: rand() * 10, // its own time
    heading: null,
  }));
  const { hull, deck, cabin, glass } = data.colours;
  const here = new THREE.Vector3(),
    behind = new THREE.Vector3(),
    ahead = new THREE.Vector3(),
    level = new THREE.Vector3(),
    up = new THREE.Vector3(),
    right = new THREE.Vector3(),
    Y = new THREE.Vector3(0, 1, 0);
  let t = 0;
  const { points, put, step } = fleet(
    boatDabs(),
    boats.length,
    () => [hull, deck, cabin, glass],
    rand,
    (dt) => {
      t += dt;
      boats.forEach((boat, k) => {
        const c = boat.course;
        boat.s += boat.way * SPEED * dt;
        along(c, boat.s - boat.way * LOOK_M, behind);
        along(c, boat.s + boat.way * LOOK_M, ahead);
        ahead.sub(behind).setY(0).normalize();
        if (!boat.heading) boat.heading = ahead.clone();
        else boat.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
        // rocking on the water: rolled, pitched, risen, each in its own time
        const o = boat.off;
        level.crossVectors(boat.heading, Y);
        up.copy(Y)
          .addScaledVector(level, ROLL * Math.sin(1.3 * t + o))
          .addScaledVector(boat.heading, PITCH * Math.sin(0.9 * t + 2 * o))
          .normalize();
        right.crossVectors(boat.heading, up).normalize();
        ahead.crossVectors(up, right).normalize();
        along(c, boat.s, here).y += HEAVE * Math.sin(1.1 * t + 3 * o);
        put(k, here, ahead, up, right);
      });
    },
  );
  step(0);
  return points;
}
