import * as THREE from 'three';
import { fleet, model, random } from '@viewer/effects/moving';

// A cat or two about the scene (life.json's cats, postprocess/life.py), on
// its own ground -- a pavement, a verge -- built and drawn as the buildings
// are (moving.js), its real size: 0.45 m long, its body, head and ears,
// tail and four legs, each a part that turns about its joint.
//
// A cat sits a while (SIT_S) at one of its spots, then gets up (RISE_S),
// turns toward another spot near by and walks there, unhurried (SPEED),
// its legs stepping in pairs, and sits down again. Sitting, its body is
// tipped up about its haunches (SIT_ANGLE), its head kept level, its front
// legs straight down to the ground, its back legs folded, its tail laid
// round beside it on the ground: compact, its rump down.
const SPEED = 0.6, // m/s
  STRIDE = 0.3, // m a step
  TURN = 3, // how fast it turns toward where it goes, per second
  SIT_S = [8, 20],
  RISE_S = 0.6,
  SIT_ANGLE = 0.9, // radians
  SIT_SHORT = 0.25, // ... its body that much shorter, gathered up
  SIT_DROP = 0.1, // m: ... and that much lower, its rump on the ground
  SWING = 0.45, // its legs' swing either way, walking (radians)
  GAP = 0.018;
// what each part is; its colour life.json's cat body's, but its tail's
const BODY = 0,
  HEAD = 1,
  TAIL = 2,
  LEGS = [3, 4, 5, 6]; // front left, front right, back left, back right
// where they turn (m: x ahead, y up): its haunches, its tail's root, each leg's hip
const HAUNCH = [-0.2, 0.12],
  TAIL_ROOT = [-0.22, 0.24],
  HIP_Y = 0.17,
  LEG_X = [0.15, 0.15, -0.16, -0.16],
  LEG_Z = [-0.05, 0.05, -0.05, 0.05],
  HEAD_AT = [0.25, 0.3];

// a box's five faces but its foot (corners [x0, x1], [y0, y1], [z0, z1])
function box(quad, [x0, x1], [y0, y1], [z0, z1], what, foot = false) {
  const c = [
    [x0, y0, z0],
    [x1, y0, z0],
    [x1, y1, z0],
    [x0, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x1, y1, z1],
    [x0, y1, z1],
  ];
  const faces = [
    [0, 1, 2, 3],
    [5, 4, 7, 6],
    [4, 0, 3, 7],
    [1, 5, 6, 2],
    [3, 2, 6, 7],
  ];
  if (foot) faces.push([4, 5, 1, 0]);
  for (const [a, b, cc, d] of faces) quad(c[a], c[b], c[cc], c[d], what);
}

// a cat's dabs, in its own frame (x ahead, y up, z right), standing: model().dabs()'s
export function catDabs() {
  const { tri, quad, dabs } = model(GAP);
  box(quad, [-0.22, 0.2], [0.12, 0.28], [-0.08, 0.08], BODY, true); // its body
  box(quad, [0.19, 0.31], [0.24, 0.36], [-0.06, 0.06], HEAD, true); // its head
  for (const s of [-1, 1])
    tri([0.22, 0.36, s * 0.05], [0.27, 0.36, s * 0.05], [0.24, 0.42, s * 0.04], HEAD); // ears
  box(quad, [-0.25, -0.22], [0.2, 0.46], [-0.015, 0.015], TAIL, true); // its tail, up
  LEGS.forEach((what, k) =>
    box(
      quad,
      [LEG_X[k] - 0.025, LEG_X[k] + 0.025],
      [0, HIP_Y],
      [LEG_Z[k] - 0.02, LEG_Z[k] + 0.02],
      what,
      true,
    ),
  );
  return dabs();
}

// p ([x, y, z]) turned by angle (radians, nose up) about [px, py] in the x-y plane
function turn(p, angle, [px, py]) {
  const c = Math.cos(angle),
    s = Math.sin(angle),
    x = p[0] - px,
    y = p[1] - py;
  p[0] = px + x * c - y * s;
  p[1] = py + x * s + y * c;
}

// p ([x, y, z]) of its body as it sits by sit (0-1): gathered toward its
// haunches, tipped up about them, lowered
function seated(p, sit) {
  p[0] = HAUNCH[0] + (p[0] - HAUNCH[0]) * (1 - SIT_SHORT * sit);
  turn(p, sit * SIT_ANGLE, HAUNCH);
  p[1] -= SIT_DROP * sit;
}

// the k-th leg's dab p (and facing q) for a cat sitting by sit (0-1), walking at phase
function leg(k, p, q, sit, phase) {
  const front = k < 2;
  // walking: swung about its hip, diagonal pairs together
  const swing = (1 - sit) * SWING * Math.sin(phase + (k === 0 || k === 3 ? 0 : Math.PI));
  turn(p, swing, [LEG_X[k], HIP_Y]);
  turn(q, swing, [0, 0]);
  if (front) {
    // straight down from its hip, wherever the tipped-up body has it
    const hip = [LEG_X[k], HIP_Y];
    seated(hip, sit);
    p[1] *= hip[1] / HIP_Y;
    p[0] += hip[0] - LEG_X[k];
  } else p[1] *= 1 - 0.7 * sit; // folded under it
}

// life.json's cats, drawn: a THREE.Points, about each tickMoving (null if none)
export function catPoints(data) {
  if (!data.cats.length) return null;
  const rand = random(Math.round(data.cats[0].spots[0][0] * 13 + data.cats[0].spots[0][1] * 7));
  const shape = catDabs();
  const first = Math.floor(rand() * data.colours.length);
  const cats = data.cats.map(({ spots }, k) => {
    const at = spots.map(([e, n, h]) => new THREE.Vector3(e, h, -n));
    const a = rand() * 2 * Math.PI;
    return {
      spots: at,
      from: at[0],
      to: at[0],
      at: at[0].clone(),
      heading: new THREE.Vector3(Math.cos(a), 0, Math.sin(a)),
      sit: 1, // 1 sitting, 0 up
      wait: SIT_S[0] * rand(), // a while sitting yet
      walking: false,
      phase: 0,
      colour: data.colours[(first + k) % data.colours.length],
    };
  });
  const ahead = new THREE.Vector3(),
    up = new THREE.Vector3(0, 1, 0),
    right = new THREE.Vector3();
  const { points, put, step } = fleet(
    shape,
    cats.length,
    (k) => {
      const { body, tail } = cats[k].colour;
      return [body, body, tail, body, body, body, body];
    },
    rand,
    (dt) =>
      cats.forEach((cat, k) => {
        if (!cat.walking) {
          if (cat.sit < 1)
            cat.sit = Math.min(1, cat.sit + dt / RISE_S); // sitting down
          else if ((cat.wait -= dt) <= 0) {
            // up, and off to another spot
            cat.walking = true;
            cat.from = cat.to;
            const others = cat.spots.filter((s) => s !== cat.from);
            cat.to = others.length ? others[Math.floor(rand() * others.length)] : cat.from;
          }
        } else if (cat.sit > 0)
          cat.sit = Math.max(0, cat.sit - dt / RISE_S); // getting up
        else {
          ahead.subVectors(cat.to, cat.at).setY(0);
          const left = ahead.length();
          if (left < 0.05) {
            cat.walking = false;
            cat.wait = SIT_S[0] + rand() * (SIT_S[1] - SIT_S[0]);
          } else {
            ahead.normalize();
            cat.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
            // it turns first, then goes
            const go = SPEED * dt * Math.max(0, cat.heading.dot(ahead)) ** 4;
            cat.at.addScaledVector(ahead, Math.min(go, left));
            cat.phase += (go / STRIDE) * 2 * Math.PI;
            // its height between its spots' as it goes
            const all = cat.from.distanceTo(cat.to) || 1;
            cat.at.y = THREE.MathUtils.lerp(cat.to.y, cat.from.y, Math.min(1, left / all));
          }
        }
        const h = cat.heading;
        right.crossVectors(h, up).normalize();
        const { sit, phase } = cat;
        put(k, cat.at, h, up, right, 1, (i, p, q) => {
          const what = shape.what[i];
          if (what >= LEGS[0]) return leg(what - LEGS[0], p, q, sit, phase);
          // its body seated, its head with it but kept level
          if (what === HEAD) {
            const at = [...HEAD_AT];
            seated(at, sit);
            p[0] += at[0] - HEAD_AT[0];
            p[1] += at[1] - HEAD_AT[1];
            return;
          }
          const angle = sit * SIT_ANGLE + (what === TAIL ? -sit * (SIT_ANGLE + 1.4) : 0);
          if (what === TAIL) turn(p, -sit * (SIT_ANGLE + 1.4), TAIL_ROOT); // laid down behind it
          seated(p, sit);
          turn(q, angle, [0, 0]);
          if (what === TAIL) {
            p[1] = Math.max(p[1], 0.01); // on the ground, curled round beside it
            p[2] += 0.1 * sit;
          }
        });
      }),
  );
  step(0);
  return points;
}
