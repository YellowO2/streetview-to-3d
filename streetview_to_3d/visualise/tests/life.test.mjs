import assert from 'node:assert/strict';
import { test } from 'node:test';
import { birdDabs, birdPoints, KINDS, wingAngle } from '@viewer/effects/birds';
import { boatDabs, boatPoints } from '@viewer/effects/boats';
import { tickMoving } from '@viewer/effects/moving';
import { duckDabs, duckPoints } from '@viewer/effects/ducks';
import { catDabs, catPoints } from '@viewer/effects/cats';

const extent = (made, d) => {
  const v = Array.from({ length: made.dab.length }, (_, i) => made.centre[3 * i + d]);
  return [Math.min(...v), Math.max(...v)];
};
const spread = (made, d) => extent(made, d)[1] - extent(made, d)[0];
const places = (points) => Array.from(points.geometry.getAttribute('position').array);

const birds = {
  centre: [0, 0, 10],
  flocks: [
    { kind: 'gull', colours: { body: [1, 1, 1], wing: [0.6, 0.6, 0.6], tip: [0, 0, 0] } },
    { kind: 'swallow', colours: { body: [0.1, 0.1, 0.2], wing: [0.1, 0.1, 0.2], tip: [0, 0, 0] } },
  ],
};

test('a bird its real size: a pigeon 0.66 m across its wings, a gull 1.25, a swallow 0.32', () => {
  const [pigeon, gull, swallow] = ['pigeon', 'gull', 'swallow'].map((k) => birdDabs(KINDS[k]));
  assert(Math.abs(spread(pigeon, 2) - 0.66) < 0.06 && Math.abs(spread(gull, 2) - 1.25) < 0.1);
  assert(Math.abs(spread(swallow, 2) - 0.32) < 0.05);
  assert(
    spread(pigeon, 0) < 0.4 && pigeon.side.some((s) => s < 0) && pigeon.side.some((s) => s > 0),
  );
  // a swallow's forked tail reaches further back, for its size
  assert(-extent(swallow, 0)[0] / 0.17 > -extent(pigeon, 0)[0] / 0.33);
});

test('a bird flaps a while, then glides on its wings held up', () => {
  const kind = { flap: [1.6, 4], rate: 5 };
  const flapping = [0.05, 0.1, 0.15].map((t) => wingAngle(kind, t, 0));
  assert(new Set(flapping).size === 3); // beating
  assert.equal(wingAngle(kind, 3, 0), wingAngle(kind, 3.5, 0)); // gliding: still
});

test('birds pass over now and then: a group at a time, at its height, then an empty sky', () => {
  const flocks = birdPoints(birds);
  const [gulls, swallows] = flocks.children;
  // a bird's middle when it shows, per kind; null when none does
  const seen = () =>
    [gulls, swallows].map((f) => {
      const dab = f.geometry.getAttribute('dab').array,
        p = f.geometry.getAttribute('position').array;
      const on = [...dab.keys()].filter((i) => dab[i] > 0);
      return on.length ? on.reduce((y, i) => y + p[3 * i + 1], 0) / on.length : null;
    });
  assert.deepEqual(seen(), [null, null]); // the sky empty to start
  const frames = [];
  for (let i = 0; i < 2000; i++) {
    tickMoving(0.1);
    frames.push(seen());
  }
  // never both kinds at once
  assert(!frames.some(([g, s]) => g !== null && s !== null));
  // each at its own height: gulls high, swallows low
  for (const [g, s] of frames) {
    if (g !== null) assert(g > 10 + 13 && g < 10 + 33);
    if (s !== null) assert(s > 10 + 2 && s < 10 + 12);
  }
  // groups come and go: the sky empty between them, several in 200 s
  const busy = frames.map(([g, s]) => g !== null || s !== null);
  const passes = busy.filter((b, i) => b && !busy[i - 1]).length;
  assert(passes >= 3 && busy.some((b) => !b));
  for (const f of flocks.children) f.geometry.dispose();
});

test('a boat its real size, cruising round its course on the water', () => {
  const boat = boatDabs();
  assert(Math.abs(spread(boat, 0) - 4.9) < 0.3 && Math.abs(spread(boat, 2) - 2) < 0.2);
  assert.equal(boatPoints({ colours: {}, courses: [] }), null);
  const square = [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100],
  ];
  const boats = boatPoints({
    colours: { hull: [1, 1, 1], deck: [0.5, 0.4, 0.3], cabin: [1, 1, 1], glass: [0, 0, 0] },
    courses: [{ level: 2, points: square }],
  });
  const before = places(boats);
  for (let i = 0; i < 20; i++) tickMoving(0.1);
  const after = places(boats);
  const mid = (a, d) => a.filter((_, i) => i % 3 === d).reduce((s, v) => s + v, 0) / (a.length / 3);
  // moved its speed, on its course (east 0-100, north 0-100: z -100..0), at the water's level
  const moved = Math.hypot(mid(after, 0) - mid(before, 0), mid(after, 2) - mid(before, 2));
  assert(moved > 4 && moved < 6.5);
  assert(mid(after, 0) > -4 && mid(after, 0) < 104 && mid(after, 2) < 4 && mid(after, 2) > -104);
  assert(mid(after, 1) > 2 - 0.2 && mid(after, 1) < 3.2);
  boats.geometry.dispose();
});

test('ducks their real size, paddling round their home on the water, never past its reach', () => {
  const duck = duckDabs();
  assert(Math.abs(spread(duck, 0) - 0.62) < 0.08 && spread(duck, 1) < 0.4);
  assert.equal(duckPoints({ colours: [], homes: [] }), null);
  const colours = [
    { body: [0.6, 0.6, 0.6], head: [0, 0.3, 0.2], tail: [0, 0, 0], bill: [0.9, 0.8, 0.2] },
  ];
  const ducks = duckPoints({ colours, homes: [{ level: 2, at: [10, 20], reach: 5 }] });
  assert.equal(ducks.userData.moving, 4);
  const before = places(ducks);
  for (let i = 0; i < 600; i++) {
    tickMoving(0.1);
    const p = places(ducks);
    for (let j = 0; j < p.length; j += 3) {
      assert(Math.hypot(p[j] - 10, p[j + 2] + 20) < 5 + 0.5); // within its reach (and its own length)
      assert(p[j + 1] > 2 - 0.05 && p[j + 1] < 2 + 0.45); // on the water
    }
  }
  assert(places(ducks).some((v, i) => Math.abs(v - before[i]) > 0.5)); // they moved
  ducks.geometry.dispose();
});

test('a cat sits at its spots and walks between them, on the ground', () => {
  const cat = catDabs();
  assert(Math.abs(spread(cat, 0) - 0.56) < 0.08 && Math.abs(spread(cat, 2) - 0.16) < 0.04);
  assert.equal(catPoints({ colours: [], cats: [] }), null);
  const colours = [{ body: [0.8, 0.5, 0.2], tail: [0.7, 0.4, 0.2] }];
  const cats = catPoints({
    colours,
    cats: [
      {
        spots: [
          [0, 0, 3],
          [3, 0, 3],
        ],
      },
    ],
  });
  const middle = () => {
    const p = places(cats);
    return [0, 1, 2].map(
      (d) => p.filter((_, i) => i % 3 === d).reduce((s, v) => s + v, 0) / (p.length / 3),
    );
  };
  const xs = [];
  for (let i = 0; i < 1200; i++) {
    tickMoving(0.05);
    const [x, y] = middle();
    xs.push(x);
    assert(Math.min(...places(cats).filter((_, j) => j % 3 === 1)) > 3 - 0.06); // never under the ground
    assert(y > 3 && y < 3.4);
  }
  // it went from one spot to the other, sitting a while at each
  assert(
    xs.some((x) => Math.abs(x - 3) < 0.4) &&
      xs.filter((x) => Math.abs(x) < 0.4 || Math.abs(x - 3) < 0.4).length > 400,
  );
  cats.geometry.dispose();
});
