import * as THREE from 'three';
import { fleet, model, random } from '@viewer/life/moving';

// Boats (life.json): 5 m motorboats built of dabs (moving.js), one per closed course,
// cruising and gently rocking.
const MAX_BOATS = 3,
  SPEED = 3, // m/s
  LOOK_M = 4, // heading sampled this far behind and ahead
  TURN = 1.5, // heading easing rate (1/s)
  GAP = 0.1, // dab spacing (m)
  ROLL = 0.035, // radians
  PITCH = 0.02, // radians
  HEAVE = 0.05; // m
// hull half-beam (m) along its length, stern to bow
const PLAN = [
  [-2.4, 0.9],
  [-1.0, 1.0],
  [0.6, 0.92],
  [1.6, 0.6],
  [2.2, 0.25],
  [2.5, 0],
];
const sheer = (x) => 0.55 + 0.32 * Math.max(0, (x + 0.5) / 3) ** 2, // gunwale height, rising to the bow
  KEEL = -0.1, // slightly under water so no gap shows at the waterline
  RAKE = 0.4, // stem rake: bow this much further back at the waterline
  DECK = 0.5;
// waterline x for plan x: pulled back near the raked bow
const foot = (x) => x - RAKE * Math.max(0, (x - 1.6) / 0.9);
// part indices into life.json's boat colours
const HULL = 0,
  DECK_W = 1,
  CABIN = 2,
  GLASS = 3;

// a boat's dabs in its local frame (x ahead, y up, z right)
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
  quad([sx, KEEL, -sw], [sx, KEEL, sw], [sx, sheer(sx), sw], [sx, sheer(sx), -sw], HULL); // transom
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
  // console amidships with a raked windscreen
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

// closed course ([[e, n], ...]) in the viewer frame at level: { points, at, length }
function course({ level, points }) {
  const p = points.map(([e, n]) => new THREE.Vector3(e, level, -n));
  p.push(p[0].clone());
  const at = [0];
  for (let i = 1; i < p.length; i++) at.push(at[i - 1] + p[i].distanceTo(p[i - 1]));
  return { points: p, at, length: at[at.length - 1] };
}

// position s metres along a course (wrapping) into out
function along(c, s, out) {
  s = ((s % c.length) + c.length) % c.length;
  let i = 1;
  while (i < c.at.length - 1 && c.at[i] < s) i++;
  const t = (s - c.at[i - 1]) / Math.max(c.at[i] - c.at[i - 1], 1e-6);
  return out.lerpVectors(c.points[i - 1], c.points[i], t);
}

// life.json's boats as a THREE.Points, cruising each tickMoving (null if none)
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
    off: rand() * 10, // rocking phase
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
        // rock: roll, pitch and heave at different rates
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
