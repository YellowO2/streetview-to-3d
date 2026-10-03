import * as THREE from 'three';
import { marks, pointsOf } from '@viewer/effects/blocks';

// Moving things from life.json (cars, birds, boats, ducks, cats): triangle models (x ahead, y up,
// z right) dabbed and drawn like the buildings (blocks.js), with per-dab seeds that move with them.
// A fleet is many copies of one model in one THREE.Points, placed on the CPU every frame.

// Model builder: tri/quad by corners, box by extents, flat polygons via corner([u, v]) -> [x, y, z];
// w is the part index into the fleet's colours. dabs() lays the dabs gap apart.
export function model(gap) {
  const pos = [],
    what = [];
  const tri = (a, b, c, w) => {
    pos.push(...a, ...b, ...c);
    what.push(w, w, w);
  };
  const quad = (a, b, c, d, w) => {
    tri(a, b, c, w);
    tri(a, c, d, w);
  };
  // the five sides of an axis-aligned box, plus its bottom if foot
  const box = ([x0, x1], [y0, y1], [z0, z1], w, foot = false) => {
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
    for (const [a, b, cc, d] of faces) quad(c[a], c[b], c[cc], c[d], w);
  };
  // a polygon ([u, v], ...) with holes, laid on a plane (corner)
  const flat = (polygon, corner, w, holes = []) => {
    const v = (p) => new THREE.Vector2(...p);
    const all = [...polygon, ...holes.flat()];
    for (const [a, b, c] of THREE.ShapeUtils.triangulateShape(
      polygon.map(v),
      holes.map((h) => h.map(v)),
    ))
      tri(corner(all[a]), corner(all[b]), corner(all[c]), w);
  };
  // marks' { centre, facing, tint, dab } plus each dab's part in what
  const dabs = () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    // part index carried through marks as a colour: w / 16
    g.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(
        what.flatMap((w) => [w / 16, 0, 0]),
        3,
      ),
    );
    g.setAttribute('gap', new THREE.Float32BufferAttribute(new Array(what.length).fill(gap), 1));
    g.setIndex([...Array(what.length).keys()]);
    const made = marks(g);
    g.dispose();
    made.what = Array.from({ length: made.dab.length }, (_, i) =>
      Math.round(made.tint[3 * i] * 16),
    );
    return made;
  };
  return { tri, quad, box, flat, dabs };
}

// polygon ([u, v], ...) clipped to coordinate axis >= at (keep > 0) or <= at (keep < 0)
export function cut(polygon, at, keep, axis = 0) {
  const out = [];
  polygon.forEach((p, i) => {
    const q = polygon[(i + 1) % polygon.length];
    const pin = (p[axis] - at) * keep >= 0,
      qin = (q[axis] - at) * keep >= 0;
    if (pin) out.push(p);
    if (pin !== qin) {
      const t = (at - p[axis]) / (q[axis] - p[axis]);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  });
  return out;
}

// seeded random number generator (mulberry32)
export function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const live = new Set();

// advance every live fleet by dt seconds (controller.js)
export function tickMoving(dt) {
  for (const tick of live) tick(Math.min(dt, 0.1));
}

// count copies of shape (model().dabs()) as one THREE.Points; colours(k) gives copy k's [r, g, b] per part.
// tick(dt) runs each frame until disposed. put(k, here, ahead, up, right, shown, bend) places copy k,
// scaled by shown; bend(i, p, n) may first move dab i's local position and facing.
export function fleet(shape, count, colours, rand, tick) {
  const per = shape.dab.length,
    n = per * count;
  const made = {
    centre: new Float32Array(3 * n),
    facing: new Float32Array(3 * n),
    tint: new Float32Array(3 * n),
    dab: new Float32Array(n),
    near: null,
    grain: new Float32Array(n).map(() => rand() * 100), // per-dab seed that moves with it
  };
  for (let k = 0; k < count; k++) {
    const colour = colours(k);
    for (let i = 0; i < per; i++) made.tint.set(colour[shape.what[i]], 3 * (k * per + i));
  }
  const points = pointsOf(made);
  const g = points.geometry;
  const position = g.getAttribute('position'),
    facing = g.getAttribute('facing'),
    dab = g.getAttribute('dab');
  for (const a of [position, facing, dab]) a.setUsage(THREE.DynamicDrawUsage);
  const p = [0, 0, 0],
    q = [0, 0, 0];
  function put(k, here, ahead, up, right, shown = 1, bend = null) {
    for (let i = 0; i < per; i++) {
      for (let d = 0; d < 3; d++) {
        p[d] = shape.centre[3 * i + d];
        q[d] = shape.facing[3 * i + d];
      }
      if (bend) bend(i, p, q);
      const j = k * per + i;
      position.setXYZ(
        j,
        here.x + ahead.x * p[0] + up.x * p[1] + right.x * p[2],
        here.y + ahead.y * p[0] + up.y * p[1] + right.y * p[2],
        here.z + ahead.z * p[0] + up.z * p[1] + right.z * p[2],
      );
      facing.setXYZ(
        j,
        ahead.x * q[0] + up.x * q[1] + right.x * q[2],
        ahead.y * q[0] + up.y * q[1] + right.y * q[2],
        ahead.z * q[0] + up.z * q[1] + right.z * q[2],
      );
      dab.setX(j, shape.dab[i] * shown);
    }
  }
  const step = (dt) => {
    tick(dt);
    position.needsUpdate = true;
    facing.needsUpdate = true;
    dab.needsUpdate = true;
  };
  live.add(step);
  g.addEventListener('dispose', () => live.delete(step));
  points.userData.moving = count;
  return { points, put, step };
}
