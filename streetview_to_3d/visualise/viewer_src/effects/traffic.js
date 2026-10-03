import * as THREE from 'three';
import { cut, fleet, model, random } from '@viewer/effects/moving';

// Cars (life.json) driving the map's roads: small hatchbacks built of dabs (moving.js).
// One car per street, on its side of the road; at a road's end it turns onto a free road,
// or with none shrinks away over FADE_S and reappears at a free street's end.
const CAR_EVERY_M = 250, // one car per this much road
  MAX_CARS = 4,
  JOIN_M = 1, // road ends this close meet
  LOOK_M = 2, // heading sampled this far behind and ahead
  TURN = 4, // heading easing rate (1/s)
  FADE_S = 1.5, // shrink/grow time when leaving or arriving
  GAP = 0.1, // dab spacing (m)
  CLEAR = 0.15; // m kept from a wall
// m/s for a road this wide (m): 20-30 km/h
const speedOf = (width) => Math.min(8.5, Math.max(5.5, 4.5 + 0.3 * width));

// side outline (m: x ahead, y up), about 3.7 m long and 1.45 m tall to the dabs' edge;
// faces sit inside the size since dabs reach past them
const FOOT_R = [-1.72, 0.3],
  FOOT_F = [1.72, 0.3],
  NOSE = [1.76, 0.6],
  HOOD = [0.98, 0.78], // bonnet end: front of the waist
  ROOF_F = [0.32, 1.32],
  ROOF_R = [-0.88, 1.36],
  WAIST_R = [-1.45, 0.84], // rear window foot: back of the waist
  TAIL = [-1.72, 0.78];
const BODY_W = 1.5, // about 1.7 m to the dabs' edge: half a lane
  CABIN_W = 1.3,
  FRAME = 0.08, // around each side window
  PILLAR = -0.22, // between the side windows (x)
  WHEELS = [-1.15, 1.15], // axle x
  WHEEL_R = 0.29,
  LAMP = [0.06, 0.34]; // lamp extent, in from each side
// part indices into life.json's car colours
const BODY = 0,
  GLASS = 1,
  TYRE = 2,
  HEAD = 3,
  TAIL_LAMP = 4;

// convex polygon ([x, y], ... anticlockwise) inset by d
function inset(polygon, d) {
  const lines = polygon.map((p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]),
      n = [-(q[1] - p[1]) / len, (q[0] - p[0]) / len]; // inward
    return [p[0] + n[0] * d, p[1] + n[1] * d, (q[0] - p[0]) / len, (q[1] - p[1]) / len];
  });
  return lines.map((a, i) => {
    const b = lines[(i + lines.length - 1) % lines.length];
    // intersection with the previous edge
    const det = b[2] * a[3] - b[3] * a[2];
    const t = ((a[0] - b[0]) * a[3] - (a[1] - b[1]) * a[2]) / det;
    return [b[0] + b[2] * t, b[1] + b[3] * t];
  });
}

// a car's dabs in its local frame (x ahead, y up, z right)
export function carDabs() {
  const { quad, flat, dabs } = model(GAP);
  const at = (z) => (p) => [...p, z];
  // a strip across the car between two outline points
  const across = (p, q, half, w) =>
    quad([...p, -half], [...q, -half], [...q, half], [...p, half], w);
  const body = [FOOT_R, FOOT_F, NOSE, HOOD, WAIST_R, TAIL],
    cabin = [HOOD, ROOF_F, ROOF_R, WAIST_R],
    window = inset(cabin, FRAME),
    windows = [cut(window, PILLAR + FRAME / 2, 1), cut(window, PILLAR - FRAME / 2, -1)];
  for (const side of [-1, 1]) {
    flat(body, at((side * BODY_W) / 2), BODY);
    flat(cabin, at((side * CABIN_W) / 2), BODY, windows);
    for (const pane of windows) flat(pane, at((side * CABIN_W) / 2), GLASS);
    // ledge where the cabin narrows
    const [inner, outer] = [(side * CABIN_W) / 2, (side * BODY_W) / 2];
    quad([...HOOD, inner], [...HOOD, outer], [...WAIST_R, outer], [...WAIST_R, inner], BODY);
    // wheels, slightly proud of the body
    for (const axle of WHEELS) {
      const rim = Array.from({ length: 14 }, (_, k) => [
        axle + WHEEL_R * Math.cos((k / 14) * 2 * Math.PI),
        WHEEL_R + WHEEL_R * Math.sin((k / 14) * 2 * Math.PI),
      ]);
      flat(rim, at(side * (BODY_W / 2 + 0.03)), TYRE);
    }
    // lamps, slightly proud of nose and tail
    const [l0, l1] = [side * (BODY_W / 2 - LAMP[1]), side * (BODY_W / 2 - LAMP[0])];
    quad([1.78, 0.47, l0], [1.78, 0.47, l1], [1.78, 0.57, l1], [1.78, 0.57, l0], HEAD);
    quad([-1.74, 0.6, l0], [-1.74, 0.6, l1], [-1.74, 0.74, l1], [-1.74, 0.74, l0], TAIL_LAMP);
  }
  // skin around the outline, no underside
  across(FOOT_F, NOSE, BODY_W / 2, BODY);
  across(NOSE, HOOD, BODY_W / 2, BODY);
  across(HOOD, ROOF_F, CABIN_W / 2, GLASS);
  across(ROOF_F, ROOF_R, CABIN_W / 2, BODY);
  across(ROOF_R, WAIST_R, CABIN_W / 2, GLASS);
  across(WAIST_R, TAIL, BODY_W / 2, BODY);
  across(TAIL, FOOT_R, BODY_W / 2, BODY);
  return dabs();
}

// life.json's car roads in the viewer frame: { points, at (m along), room (m to a wall), length,
// width, speed, lane, ends: [[{ road, end }] at start, ... at end], street (its street's first road) }
export function network(data) {
  const roads = data.roads
    .filter((r) => r.points.length >= 2)
    .map((r) => {
      const points = r.points.map(([e, n, h]) => new THREE.Vector3(e, h, -n));
      const at = [0];
      for (let i = 1; i < points.length; i++)
        at.push(at[i - 1] + points[i].distanceTo(points[i - 1]));
      return {
        points,
        at,
        room: r.room || points.map(() => r.width / 2),
        length: at[at.length - 1],
        width: r.width,
        speed: speedOf(r.width),
        lane: r.width / 4,
        ends: [[], []],
      };
    });
  const ends = roads.flatMap((r, road) => [0, 1].map((end) => ({ road, end })));
  const where = ({ road, end }) => roads[road].points[end ? roads[road].points.length - 1 : 0];
  const cell = new Map();
  const key = (p, dx = 0, dz = 0) =>
    `${Math.floor(p.x / JOIN_M) + dx},${Math.floor(p.z / JOIN_M) + dz}`;
  for (const e of ends) {
    const k = key(where(e));
    if (!cell.has(k)) cell.set(k, []);
    cell.get(k).push(e);
  }
  for (const e of ends) {
    const p = where(e);
    for (let dx = -1; dx <= 1; dx++)
      for (let dz = -1; dz <= 1; dz++)
        for (const o of cell.get(key(p, dx, dz)) || [])
          if ((o.road !== e.road || o.end !== e.end) && where(o).distanceTo(p) <= JOIN_M)
            roads[e.road].ends[e.end].push(o);
  }
  const street = roads.map((_, i) => i);
  const find = (i) => (street[i] === i ? i : (street[i] = find(street[i])));
  roads.forEach((r, i) =>
    r.ends.forEach((at) => {
      if (at.length === 1) street[find(i)] = find(at[0].road);
    }),
  );
  roads.forEach((r, i) => (r.street = find(i)));
  return roads;
}

// position s metres along road into out (extrapolated past the ends), its direction into ahead;
// returns the room there (m)
function along(road, s, out, ahead) {
  let i = 1;
  while (i < road.at.length - 1 && road.at[i] < s) i++;
  const t = (s - road.at[i - 1]) / Math.max(road.at[i] - road.at[i - 1], 1e-6);
  out.lerpVectors(road.points[i - 1], road.points[i], t);
  if (ahead) {
    const a = new THREE.Vector3(),
      b = new THREE.Vector3();
    along(road, s - LOOK_M, a);
    along(road, s + LOOK_M, b);
    ahead.subVectors(b, a).normalize();
  }
  const u = Math.min(1, Math.max(0, t));
  return road.room[i - 1] * (1 - u) + road.room[i] * u;
}

// life.json's cars as a THREE.Points, driven each tickMoving
export function trafficPoints(data) {
  const roads = network(data);
  const total = roads.reduce((sum, r) => sum + r.length, 0);
  const rand = random(roads.length * 7919 + Math.round(total));
  const streets = [...new Set(roads.map((r) => r.street))];
  const count = Math.min(MAX_CARS, streets.length, Math.max(1, Math.round(total / CAR_EVERY_M)));
  const shape = carDabs();
  const half = BODY_W / 2 + (GAP * 1.6) / 2; // half width to the dabs' edge
  const { body, glass, tyre, head, tail } = data.colours;
  // distinct colours while they last
  const order = body.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  // street -> the car on it
  const taken = new Map();
  const free = (street) => !taken.has(street);
  // start on random streets
  for (let i = streets.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [streets[i], streets[j]] = [streets[j], streets[i]];
  }
  const cars = streets.slice(0, count).map((street, c) => {
    // random road (weighted by length), place and direction
    const mine = roads.map((r, i) => i).filter((i) => roads[i].street === street);
    let pick = rand() * mine.reduce((sum, i) => sum + roads[i].length, 0),
      k = 0;
    while (k < mine.length - 1 && pick > roads[mine[k]].length) pick -= roads[mine[k++]].length;
    const car = {
      road: mine[k],
      s: pick,
      way: rand() < 0.5 ? 1 : -1,
      colour: body[order[c % order.length]],
      heading: null,
      shown: 1, // 0 gone, 1 fully shown
      leaving: false,
      street,
    };
    taken.set(street, car);
    return car;
  });
  // entry points: dead ends
  const entries = roads.flatMap((r, road) =>
    [0, 1].filter((end) => !r.ends[end].length).map((end) => ({ road, end })),
  );
  // re-enter at a free street's dead end (else any free road's start); none free: wait
  const comeIn = (car) => {
    const open = entries.filter((e) => free(roads[e.road].street));
    const at = open.length
      ? open[Math.floor(rand() * open.length)]
      : roads.findIndex((r) => free(r.street)) >= 0
        ? { road: roads.findIndex((r) => free(r.street)), end: 0 }
        : null;
    if (!at) return;
    const { road, end } = at;
    Object.assign(car, {
      road,
      s: end ? roads[road].length : 0,
      way: end ? -1 : 1,
      heading: null,
      shown: 0,
      leaving: false,
      street: roads[road].street,
    });
    taken.set(car.street, car);
  };
  const here = new THREE.Vector3(),
    ahead = new THREE.Vector3(),
    right = new THREE.Vector3(),
    up = new THREE.Vector3(),
    Y = new THREE.Vector3(0, 1, 0);
  const { points, put, step } = fleet(
    shape,
    cars.length,
    (c) => [cars[c].colour, glass, tyre, head, tail],
    rand,
    (dt) =>
      cars.forEach((car, c) => {
        let road = roads[car.road];
        car.s += car.way * road.speed * dt;
        // past the road's end: onto a connecting road, or leave
        if (!car.leaving && (car.s < 0 || car.s > road.length)) {
          const end = car.s > road.length ? 1 : 0;
          // same street, or a free one
          const next = road.ends[end].filter(
            (o) => roads[o.road].street === car.street || free(roads[o.road].street),
          );
          if (next.length) {
            const over = Math.abs(end ? car.s - road.length : car.s);
            const o = next[Math.floor(rand() * next.length)];
            car.road = o.road;
            road = roads[o.road];
            if (road.street !== car.street) {
              taken.delete(car.street);
              car.street = road.street;
              taken.set(car.street, car);
            }
            car.way = o.end === 0 ? 1 : -1;
            car.s = o.end === 0 ? over : road.length - over;
          } else car.leaving = true;
        }
        car.shown = Math.min(1, Math.max(0, car.shown + ((car.leaving ? -1 : 1) * dt) / FADE_S));
        if (car.leaving && car.shown === 0) {
          if (taken.get(car.street) === car) taken.delete(car.street);
          comeIn(car);
          road = roads[car.road];
        }
        const room = along(road, car.s, here, ahead);
        if (car.way < 0) ahead.negate();
        // ease the heading onto the road's
        if (!car.heading || car.heading.dot(ahead) < -0.5) car.heading = ahead.clone();
        else car.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
        const h = car.heading;
        right.crossVectors(h, Y).normalize();
        up.crossVectors(right, h).normalize();
        // lane centre, or as far over as the nearest wall allows
        const over = Math.max(0, Math.min(road.lane, room - half - CLEAR));
        here.addScaledVector(right, (data.side === 'left' ? -1 : 1) * over);
        put(c, here, h, up, right, car.shown);
      }),
  );
  step(0);
  return points;
}
