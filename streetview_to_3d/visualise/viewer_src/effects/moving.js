import * as THREE from 'three';
import { marks, pointsOf } from '@viewer/effects/blocks';

// What moves in the world (life.json, postprocess/life.py: cars, birds,
// boats) built and drawn as the buildings are (blocks.js): each thing's
// surfaces triangles in its own frame (a model: x ahead, y up, z right),
// laid with dabs by the buildings' own marks -- a jittered grid, strokes
// along its creases -- and drawn by their own shader, lit, varied and
// scattered as a building's dab is, its own way carried with it as it moves
// (pointsOf's grain), so nothing shimmers. Their colours are life.json's,
// never made up here.
//
// A fleet is many of one model in one cloud of points, each placed every
// frame on the CPU (put): where it is, which way it faces, how much of it
// there is (shown: it shrinks away and grows back), and, for one that bends
// (a bird's wings), each dab moved in its own frame first.

// A model's surfaces: tri/quad by corners, flat polygons by a plane's
// corners ([u, v] -> [x, y, z]), each what it is (an index into the
// colours it is drawn in); dabs() lays them, gap apart, as the buildings'.
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
  // its dabs: marks' { centre, facing, tint, dab }, each one's what in what
  const dabs = () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    // what each is, carried through marks as a colour: what / 16
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
  return { tri, quad, flat, dabs };
}

// a polygon ([u, v], ...) cut to u >= at (keep > 0) or u <= at (keep < 0) along axis (0: u, 1: v)
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

// a random number generator, the same each time for the same seed
export function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const live = new Set();

// Every scene's moving things moved on dt seconds (controller.js, as the water's motion).
export function tickMoving(dt) {
  for (const tick of live) tick(Math.min(dt, 0.1));
}

// count of a model's dabs (shape: model().dabs()) as one THREE.Points, the
// k-th's in colours(k) (an [r, g, b] for each what); tick(dt) moves them
// each frame (tickMoving) until the points are disposed. put(k, here,
// ahead, up, right, shown, bend) places the k-th: bend(i, p, n), if given,
// moves its i-th dab's place and facing (in its own frame) first.
export function fleet(shape, count, colours, rand, tick) {
  const per = shape.dab.length,
    n = per * count;
  const made = {
    centre: new Float32Array(3 * n),
    facing: new Float32Array(3 * n),
    tint: new Float32Array(3 * n),
    dab: new Float32Array(n),
    near: null,
    grain: new Float32Array(n).map(() => rand() * 100), // each dab's own way, kept as it moves
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
