import * as THREE from 'three';
import { fleet, model, random } from '@viewer/effects/moving';

// Cats (life.json): 0.45 m cats built of dabs (moving.js) that sit at a spot for a while,
// get up, walk to another nearby spot and sit again; jointed body, head, tail and legs.
const SPEED = 0.6, // m/s
  STRIDE = 0.3, // m per step
  TURN = 3, // turning rate (1/s)
  SIT_S = [8, 20],
  RISE_S = 0.6, // time to sit down or stand up
  SIT_ANGLE = 0.9, // body tilt sitting (radians)
  SIT_SHORT = 0.25, // body shortening sitting
  SIT_DROP = 0.1, // m lowered sitting
  SWING = 0.45, // leg swing either way walking (radians)
  GAP = 0.018; // dab spacing (m)
// part indices; all body colour but the tail
const BODY = 0,
  HEAD = 1,
  TAIL = 2,
  LEGS = [3, 4, 5, 6]; // front left, front right, back left, back right
// joints (m: x ahead, y up): haunches, tail root, leg hips
const HAUNCH = [-0.2, 0.12],
  TAIL_ROOT = [-0.22, 0.24],
  HIP_Y = 0.17,
  LEG_X = [0.15, 0.15, -0.16, -0.16],
  LEG_Z = [-0.05, 0.05, -0.05, 0.05],
  HEAD_AT = [0.25, 0.3];

// a standing cat's dabs in its local frame (x ahead, y up, z right)
export function catDabs() {
  const { tri, box, dabs } = model(GAP);
  box([-0.22, 0.2], [0.12, 0.28], [-0.08, 0.08], BODY, true);
  box([0.19, 0.31], [0.24, 0.36], [-0.06, 0.06], HEAD, true);
  for (const s of [-1, 1])
    tri([0.22, 0.36, s * 0.05], [0.27, 0.36, s * 0.05], [0.24, 0.42, s * 0.04], HEAD); // ears
  box([-0.25, -0.22], [0.2, 0.46], [-0.015, 0.015], TAIL, true); // tail, up
  LEGS.forEach((what, k) =>
    box(
      [LEG_X[k] - 0.025, LEG_X[k] + 0.025],
      [0, HIP_Y],
      [LEG_Z[k] - 0.02, LEG_Z[k] + 0.02],
      what,
      true,
    ),
  );
  return dabs();
}

// rotate p ([x, y, z]) by angle (radians, nose up) about [px, py] in the x-y plane
function turn(p, angle, [px, py]) {
  const c = Math.cos(angle),
    s = Math.sin(angle),
    x = p[0] - px,
    y = p[1] - py;
  p[0] = px + x * c - y * s;
  p[1] = py + x * s + y * c;
}

// body point p seated by sit (0-1): shortened toward the haunches, tipped up, lowered
function seated(p, sit) {
  p[0] = HAUNCH[0] + (p[0] - HAUNCH[0]) * (1 - SIT_SHORT * sit);
  turn(p, sit * SIT_ANGLE, HAUNCH);
  p[1] -= SIT_DROP * sit;
}

// leg k's dab p (and facing q) for sit (0-1) and walking phase
function leg(k, p, q, sit, phase) {
  const front = k < 2;
  // walking: swing about the hip, diagonal pairs together
  const swing = (1 - sit) * SWING * Math.sin(phase + (k === 0 || k === 3 ? 0 : Math.PI));
  turn(p, swing, [LEG_X[k], HIP_Y]);
  turn(q, swing, [0, 0]);
  if (front) {
    // front legs reach straight down from the moved hip
    const hip = [LEG_X[k], HIP_Y];
    seated(hip, sit);
    p[1] *= hip[1] / HIP_Y;
    p[0] += hip[0] - LEG_X[k];
  } else p[1] *= 1 - 0.7 * sit; // back legs fold
}

// life.json's cats as a THREE.Points, moved each tickMoving (null if none)
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
      sit: 1, // 1 sitting, 0 standing
      wait: SIT_S[0] * rand(), // seconds left sitting
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
            cat.sit = Math.min(1, cat.sit + dt / RISE_S); // sit down
          else if ((cat.wait -= dt) <= 0) {
            // get up and pick another spot
            cat.walking = true;
            cat.from = cat.to;
            const others = cat.spots.filter((s) => s !== cat.from);
            cat.to = others.length ? others[Math.floor(rand() * others.length)] : cat.from;
          }
        } else if (cat.sit > 0)
          cat.sit = Math.max(0, cat.sit - dt / RISE_S); // stand up
        else {
          ahead.subVectors(cat.to, cat.at).setY(0);
          const left = ahead.length();
          if (left < 0.05) {
            cat.walking = false;
            cat.wait = SIT_S[0] + rand() * (SIT_S[1] - SIT_S[0]);
          } else {
            ahead.normalize();
            cat.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
            // turn first, then walk
            const go = SPEED * dt * Math.max(0, cat.heading.dot(ahead)) ** 4;
            cat.at.addScaledVector(ahead, Math.min(go, left));
            cat.phase += (go / STRIDE) * 2 * Math.PI;
            // interpolate height between the spots
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
          // head follows the seated body but stays level
          if (what === HEAD) {
            const at = [...HEAD_AT];
            seated(at, sit);
            p[0] += at[0] - HEAD_AT[0];
            p[1] += at[1] - HEAD_AT[1];
            return;
          }
          const angle = sit * SIT_ANGLE + (what === TAIL ? -sit * (SIT_ANGLE + 1.4) : 0);
          if (what === TAIL) turn(p, -sit * (SIT_ANGLE + 1.4), TAIL_ROOT); // laid down behind
          seated(p, sit);
          turn(q, angle, [0, 0]);
          if (what === TAIL) {
            p[1] = Math.max(p[1], 0.01); // on the ground, curled beside it
            p[2] += 0.1 * sit;
          }
        });
      }),
  );
  step(0);
  return points;
}
