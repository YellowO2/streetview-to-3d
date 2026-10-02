import assert from 'node:assert/strict';
import { test } from 'node:test';
import { carDabs, network, tickTraffic, trafficPoints } from '@viewer/effects/traffic';

const colours = {
  body: [[0.8, 0.1, 0.1]],
  glass: [0.1, 0.1, 0.1],
  tyre: [0, 0, 0],
  head: [1, 1, 0.9],
  tail: [0.6, 0.1, 0.1],
};
// a cross: a road east from the middle, one north, one west; and one apart, far off
const roads = [
  {
    width: 6,
    points: [
      [0, 0, 5],
      [200, 0, 5],
    ],
  },
  {
    width: 6,
    points: [
      [0, 0, 5],
      [0, 200, 5],
    ],
  },
  {
    width: 6,
    points: [
      [-200, 0, 5],
      [0.5, 0, 5],
    ],
  },
  {
    width: 6,
    points: [
      [1000, 1000, 5],
      [1300, 1000, 5],
    ],
  },
];

test('roads meet where their ends do, and nowhere else', () => {
  const net = network({ roads });
  assert.equal(net[0].ends[0].length, 2); // the middle: the other two
  assert.deepEqual(net[2].ends[1].map((e) => e.road).sort(), [0, 1]);
  assert.equal(net[0].ends[1].length, 0);
  assert.equal(net[3].ends[0].length + net[3].ends[1].length, 0);
  assert.deepEqual(net[1].points[1].toArray(), [0, 5, -200]); // east, height, -north
});

// each car's middle: its points' mean, on the ground's plane
function middles(points) {
  const p = points.geometry.getAttribute('position').array;
  const per = p.length / 3 / points.userData.traffic;
  return Array.from({ length: points.userData.traffic }, (_, c) => {
    let x = 0,
      y = 0,
      z = 0;
    for (let i = 0; i < per; i++) {
      x += p[3 * (c * per + i)];
      y += p[3 * (c * per + i) + 1];
      z += p[3 * (c * per + i) + 2];
    }
    return [x / per, y / per, z / per];
  });
}

test('cars drive their roads at their height, to one side, and keep going', () => {
  const one = [
    {
      width: 8,
      points: [
        [0, 0, 5],
        [3000, 0, 5],
      ],
    },
  ];
  const cars = trafficPoints({ side: 'right', colours, roads: one });
  assert.equal(cars.userData.traffic, 1); // one road: one car
  const before = middles(cars);
  for (const [, y, z] of before) {
    assert(y > 5 && y < 6.5); // on it
    assert(Math.abs(Math.abs(z) - 2) < 0.2); // a lane's middle off its centre (8 m: 2 m)
  }
  for (let i = 0; i < 10; i++) tickTraffic(0.1); // a second
  const after = middles(cars);
  // each moved its road's speed (a car near an end may have left),
  // keeping right: north of it (z < 0) going west
  after.forEach(([x, , z], c) => {
    if (before[c][0] < 15 || before[c][0] > 2985) return;
    assert(Math.abs(Math.abs(x - before[c][0]) - 6.9) < 0.3); // 25 km/h on an 8 m road;
    const west = x < before[c][0];
    assert.equal(z < 0, west);
  });
  cars.geometry.dispose();
});

test('on the left where the country drives on it', () => {
  const one = [
    {
      width: 8,
      points: [
        [0, 0, 5],
        [3000, 0, 5],
      ],
    },
  ];
  const cars = trafficPoints({ side: 'left', colours, roads: one });
  const before = middles(cars);
  tickTraffic(1);
  middles(cars).forEach(([x, , z], c) => {
    if (before[c][0] < 15 || before[c][0] > 2985) return;
    assert.equal(z > 0, x < before[c][0]);
  });
  cars.geometry.dispose();
});

test('a car at an end no road meets shrinks away and comes back in at one, growing', () => {
  const one = [
    {
      width: 8,
      points: [
        [0, 0, 5],
        [150, 0, 5],
      ],
    },
  ];
  const cars = trafficPoints({ side: 'right', colours, roads: one });
  assert.equal(cars.userData.traffic, 1);
  const dab = cars.geometry.getAttribute('dab').array;
  const sizes = [];
  for (let i = 0; i < 400; i++) {
    tickTraffic(0.1);
    sizes.push(dab[0]);
  }
  const full = Math.max(...sizes);
  const gone = sizes.indexOf(Math.min(...sizes));
  assert(sizes[gone] < 0.05 * full); // away
  assert(sizes.slice(gone).some((d) => d === full)); // and back, whole
  for (const [x] of middles(cars)) assert(x > -30 && x < 180); // never far past an end
  cars.geometry.dispose();
});

test('a car keeps off a wall nearer than its lane: as far over as its width leaves it', () => {
  const tight = [
    {
      width: 8,
      points: [
        [0, 0, 5],
        [3000, 0, 5],
      ],
      room: [1.5, 1.5],
    },
  ];
  const cars = trafficPoints({ side: 'right', colours, roads: tight });
  for (const [, , z] of middles(cars)) assert(Math.abs(z) < 0.6); // 1.5 m to the wall, not 2
  cars.geometry.dispose();
});

test('a car is a car: about 3.7 m long, 1.7 wide, 1.45 tall, glass, wheels and lamps', () => {
  const car = carDabs();
  const ext = [0, 1, 2].map((d) => {
    const v = Array.from({ length: car.dab.length }, (_, i) => car.centre[3 * i + d]);
    return Math.max(...v) - Math.min(...v);
  });
  assert(Math.abs(ext[0] - 3.5) < 0.15 && Math.abs(ext[2] - 1.56) < 0.1 && ext[1] < 1.45);
  for (const w of [0, 1, 2, 3, 4]) assert(car.what.includes(w));
});

test('a street has one car at a time: stretches meeting only each other are one', () => {
  const net = network({ roads });
  // the cross's three meet at one junction: three streets; the one apart its own
  assert.equal(new Set(net.map((r) => r.street)).size, 4);
  const line = [
    {
      width: 6,
      points: [
        [0, 500, 5],
        [100, 500, 5],
      ],
    },
    {
      width: 6,
      points: [
        [100, 500, 5],
        [200, 500, 5],
      ],
    },
  ];
  const one = network({ roads: line });
  assert.equal(one[0].street, one[1].street);
  const cars = trafficPoints({ side: 'right', colours, roads: [...roads, ...line] });
  assert.equal(cars.userData.traffic, 5);
  cars.geometry.dispose();
});
