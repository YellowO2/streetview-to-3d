import * as THREE from 'three';
import { cut, fleet, model, random } from '@viewer/effects/moving';

// Cars driving the map's roads (life.json's cars, postprocess/life.py),
// built and drawn as the buildings are (moving.js).
//
// A car is a small hatchback, its real size: its side's outline (bonnet,
// sloped windscreen, roof, rear window, boot) carried across its width, the
// cabin above its waist narrower than the body below; glass in its windows
// either side of a pillar, its windscreen and rear window; wheels; head and
// tail lights. Its dabs reach past its faces as a building's do, so the
// faces sit that much inside its size (BODY_W, CABIN_W).
//
// A road is a stretch between junctions or ends; two meet where their ends
// are within JOIN_M, and stretches meeting only each other are one street.
// A street has one car on it at a time: a quiet place. A car keeps to its
// side of the road (life.json's side) at its road's own unhurried speed,
// in its lane's middle, unless a wall is nearer than that (the road's room):
// then as far over as its width leaves it. At the end of one road it turns
// onto another there, any but the one it came by, on its own street or one
// no car is on. Where there is none (the reach's edge, a tunnel's mouth, a
// junction of busy streets) it drives on, shrinking away over FADE_S, and
// comes back in at an end of a street no car is on, growing as it comes:
// cars come and go. Its heading eases round a turn (TURN), so it never
// jumps across.
const MAX_CARS = 12,
  JOIN_M = 1,
  LOOK_M = 2, // its heading from the road this far behind and ahead
  TURN = 4, // how fast its heading eases onto the road's, per second
  FADE_S = 1.5, // a car leaving or coming shrinks away or grows over this long
  GAP = 0.1, // its dabs' spacing, as a building's gap
  CLEAR = 0.15; // kept off a wall, past its side
// how fast on a road this wide (m): unhurried, 20-30 km/h
const speedOf = (width) => Math.min(8.5, Math.max(5.5, 4.5 + 0.3 * width));

// its side's outline (metres: x ahead, y up): about 3.7 m long and 1.45 m
// tall to its dabs' edge
const FOOT_R = [-1.72, 0.3],
  FOOT_F = [1.72, 0.3],
  NOSE = [1.76, 0.6],
  HOOD = [0.98, 0.78], // the bonnet's end: its waist, in front
  ROOF_F = [0.32, 1.32],
  ROOF_R = [-0.88, 1.36],
  WAIST_R = [-1.45, 0.84], // the rear window's foot: its waist, behind
  TAIL = [-1.72, 0.78];
const BODY_W = 1.5, // about 1.7 m to its dabs' edge: half a lane
  CABIN_W = 1.3,
  FRAME = 0.08, // round each side window
  PILLAR = -0.22, // between the side windows, along it
  WHEELS = [-1.15, 1.15], // their axles, along it
  WHEEL_R = 0.29,
  LAMP = [0.06, 0.34]; // a lamp between these in from each side
// what each surface is: life.json's car colours
const BODY = 0,
  GLASS = 1,
  TYRE = 2,
  HEAD = 3,
  TAIL_LAMP = 4;

// a convex polygon ([x, y], ... anticlockwise) moved in by d all round
function inset(polygon, d) {
  const lines = polygon.map((p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]),
      n = [-(q[1] - p[1]) / len, (q[0] - p[0]) / len]; // inward
    return [p[0] + n[0] * d, p[1] + n[1] * d, (q[0] - p[0]) / len, (q[1] - p[1]) / len];
  });
  return lines.map((a, i) => {
    const b = lines[(i + lines.length - 1) % lines.length];
    // where the line before meets this one
    const det = b[2] * a[3] - b[3] * a[2];
    const t = ((a[0] - b[0]) * a[3] - (a[1] - b[1]) * a[2]) / det;
    return [b[0] + b[2] * t, b[1] + b[3] * t];
  });
}

// A car's dabs, in its own frame (x ahead, y up, z right): model().dabs()'s.
export function carDabs() {
  const { quad, flat, dabs } = model(GAP);
  const at = (z) => (p) => [...p, z];
  // a strip across from one point of the outline to the next, half wide each side
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
    // the waist's ledge, where the cabin narrows
    const [inner, outer] = [(side * CABIN_W) / 2, (side * BODY_W) / 2];
    quad([...HOOD, inner], [...HOOD, outer], [...WAIST_R, outer], [...WAIST_R, inner], BODY);
    // the wheels, a little proud of the body
    for (const axle of WHEELS) {
      const rim = Array.from({ length: 14 }, (_, k) => [
        axle + WHEEL_R * Math.cos((k / 14) * 2 * Math.PI),
        WHEEL_R + WHEEL_R * Math.sin((k / 14) * 2 * Math.PI),
      ]);
      flat(rim, at(side * (BODY_W / 2 + 0.03)), TYRE);
    }
    // the lamps, a little proud of its nose and tail
    const [l0, l1] = [side * (BODY_W / 2 - LAMP[1]), side * (BODY_W / 2 - LAMP[0])];
    quad([1.78, 0.47, l0], [1.78, 0.47, l1], [1.78, 0.57, l1], [1.78, 0.57, l0], HEAD);
    quad([-1.74, 0.6, l0], [-1.74, 0.6, l1], [-1.74, 0.74, l1], [-1.74, 0.74, l0], TAIL_LAMP);
  }
  // its skin round the outline, not its underside
  across(FOOT_F, NOSE, BODY_W / 2, BODY);
  across(NOSE, HOOD, BODY_W / 2, BODY);
  across(HOOD, ROOF_F, CABIN_W / 2, GLASS);
  across(ROOF_F, ROOF_R, CABIN_W / 2, BODY);
  across(ROOF_R, WAIST_R, CABIN_W / 2, GLASS);
  across(WAIST_R, TAIL, BODY_W / 2, BODY);
  across(TAIL, FOOT_R, BODY_W / 2, BODY);
  return dabs();
}

// life.json's car roads in the viewer's frame (east, height, -north): each
// { points: [Vector3], at: [metres along], room: [m], length, width, speed,
// lane, ends: [[{ road, end }] at its start, ... at its end], street: the
// first road of the ones it runs on into without a junction }
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

// where along road (metres) is -- past an end, on along its last stretch --
// and which way it runs there (unit, its own way); its room there (m)
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

// life.json's cars as its cars, drawn: a THREE.Points, driving each tickMoving
export function trafficPoints(data) {
  const roads = network(data);
  const total = roads.reduce((sum, r) => sum + r.length, 0);
  const rand = random(roads.length * 7919 + Math.round(total));
  const streets = [...new Set(roads.map((r) => r.street))];
  const count = Math.min(MAX_CARS, streets.length);
  const shape = carDabs();
  const half = BODY_W / 2 + (GAP * 1.6) / 2; // to its dabs' edge
  const { body, glass, tyre, head, tail } = data.colours;
  // each car a colour of its own while they last
  const order = body.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  // which street each car is on (null: gone, waiting to come back)
  const taken = new Map();
  const free = (street) => !taken.has(street);
  // each street its car to start with, a few streets at random if they are many
  for (let i = streets.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [streets[i], streets[j]] = [streets[j], streets[i]];
  }
  const cars = streets.slice(0, count).map((street, c) => {
    // a road of it by its length, a place on it, a way along it
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
      shown: 1, // how much of it there is: 0 gone, 1 all of it
      leaving: false,
      street,
    };
    taken.set(street, car);
    return car;
  });
  // where cars come in: the ends no other road meets, else any road's start
  const entries = roads.flatMap((r, road) =>
    [0, 1].filter((end) => !r.ends[end].length).map((end) => ({ road, end })),
  );
  // back in at an end of a street no car is on; none yet, it waits
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
        // off its road's end: onto another there, or away
        if (!car.leaving && (car.s < 0 || car.s > road.length)) {
          const end = car.s > road.length ? 1 : 0;
          // on along its street, or onto one no car is on
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
        // its heading easing onto the road's
        if (!car.heading || car.heading.dot(ahead) < -0.5) car.heading = ahead.clone();
        else car.heading.lerp(ahead, 1 - Math.exp(-TURN * dt)).normalize();
        const h = car.heading;
        right.crossVectors(h, Y).normalize();
        up.crossVectors(right, h).normalize();
        // in its lane's middle, or as far over as the nearest wall leaves it
        const over = Math.max(0, Math.min(road.lane, room - half - CLEAR));
        here.addScaledVector(right, (data.side === 'left' ? -1 : 1) * over);
        put(c, here, h, up, right, car.shown);
      }),
  );
  step(0);
  return points;
}
