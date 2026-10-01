import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { GAPS, level, scatter, facadeColour, FLOOR_M, BAY_M } from '@viewer/effects/scatter';

// A surface as parseSurface gives it: triangles, a colour, gap and facade per corner.
function surface(corners, faces, gap, facade) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(corners.flat(), 3));
  g.setAttribute(
    'color',
    new THREE.Float32BufferAttribute(
      corners.flatMap(() => [0.8, 0.7, 0.6]),
      3,
    ),
  );
  g.setAttribute(
    'gap',
    new THREE.Float32BufferAttribute(
      corners.map(() => gap),
      1,
    ),
  );
  if (facade) g.setAttribute('facade', new THREE.Float32BufferAttribute(facade.flat(), 2));
  g.setIndex(faces.flat());
  return g;
}

test('a square of ground is covered evenly, two triangles meeting without a seam', () => {
  const ground = surface(
    [
      [0, 0, 0],
      [10, 0, 0],
      [10, 0, 10],
      [0, 0, 10],
    ],
    [
      [0, 1, 2],
      [0, 2, 3],
    ],
    0.5,
  );
  const points = scatter(ground);
  const step = GAPS[level(0.5)];
  assert(step >= 0.5 && step < 0.5 * 1.25);
  const n = points.getAttribute('position').count;
  assert(Math.abs(n - 100 / step ** 2) < 0.08 * n); // one a grid square, give or take the edges
  assert(points.getAttribute('gap').array.every((g) => g === step));
  // every 1 m square has its points: no seam along the diagonal
  const p = points.getAttribute('position');
  const seen = new Set();
  for (let i = 0; i < n; i++) seen.add(`${Math.floor(p.getX(i))},${Math.floor(p.getZ(i))}`);
  assert.equal(seen.size, 100);
  assert(Array.from({ length: n }, (_, i) => p.getY(i)).every((y) => y === 0));
});

test('a wall is gridded on its floors and bays: windows while there are two points a floor', () => {
  // a wall 12 m long, 9 m high, running east, facing south
  const wall = (gap) =>
    scatter(
      surface(
        [
          [0, 0, 0],
          [12, 0, 0],
          [12, 9, 0],
          [0, 9, 0],
        ],
        [
          [0, 1, 2],
          [0, 2, 3],
        ],
        gap,
        [
          [0, 0],
          [12, 0],
          [12, 9],
          [0, 9],
        ],
      ),
    );
  const near = wall(0.2);
  const n = near.getAttribute('position').count;
  assert(n > 0.8 * (108 / GAPS[level(0.2)] ** 2));
  const c = near.getAttribute('color');
  const reds = Array.from({ length: n }, (_, i) => c.getX(i));
  assert(Math.min(...reds) < 0.55 && Math.max(...reds) > 0.75); // dark glass and bright wall
  // 1.5 m apart: every other point on a window, the rest between -- still floors of windows
  const mid = wall(1.5);
  const mc = mid.getAttribute('color');
  const midReds = Array.from({ length: mc.count }, (_, i) => mc.getX(i));
  assert(Math.min(...midReds) < 0.6 && Math.max(...midReds) > 0.7);
  const far = wall(3);
  const fc = far.getAttribute('color');
  const spread = (a) => Math.max(...a) - Math.min(...a);
  const farReds = Array.from({ length: fc.count }, (_, i) => fc.getX(i));
  const high = Array.from({ length: fc.count }, (_, i) => i).filter(
    (i) => far.getAttribute('position').getY(i) > 4,
  );
  assert(spread(high.map((i) => farReds[i])) < 0.02); // too far apart for windows: the wall's average
  assert(Math.max(...farReds) < 0.8);
});

test('a window, its sill and the wall at its foot', () => {
  const at = (u, v) => facadeColour([0.8, 0.8, 0.8], u, v)[0];
  const glass = at(BAY_M * 0.5, FLOOR_M * 1.5),
    plain = at(BAY_M * 0.1, FLOOR_M * 1.5),
    sill = at(BAY_M * 0.5, FLOOR_M * 1.27);
  assert(glass < plain && sill > plain);
  assert(at(BAY_M * 0.1, 0.1) < plain); // darker at its foot
  assert(at(BAY_M * 0.5, FLOOR_M * 0.5) === at(BAY_M * 0.5, FLOOR_M * 0.5)); // the same every time
});

test('old and new building PLY files scatter with optional facade layout', async () => {
  const { parseSurface } = await import('@viewer/effects/scatter');
  const ply = (layout) =>
    new TextEncoder().encode(
      [
        'ply',
        'format ascii 1.0',
        'element vertex 4',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'property float facade_u',
        'property float facade_v',
        ...(layout ? ['property float facade_bay', 'property float facade_floor'] : []),
        'property float gap',
        'element face 2',
        'property list uchar int vertex_indices',
        'end_header',
        ...[
          [0, 0],
          [12, 0],
          [12, 9],
          [0, 9],
        ].map(([u, v]) => `${u} ${v} 0 204 179 153 ${u} ${v} ${layout ? '2.4 4.5 ' : ''}0.4`),
        '3 0 1 2',
        '3 0 2 3',
        '',
      ].join('\n'),
    ).buffer;
  for (const layout of [false, true]) {
    const mesh = parseSurface(ply(layout), new THREE.Matrix4(), 'building');
    assert.equal(mesh.getAttribute('facade').itemSize, 2);
    assert.equal(Boolean(mesh.getAttribute('facadeLayout')), layout);
    const points = scatter(mesh);
    assert(points.getAttribute('position').count > 100);
    assert(points.getAttribute('color').array.every(Number.isFinite));
    mesh.dispose();
    points.dispose();
  }
  const office = (v) => facadeColour([0.8, 0.8, 0.8], 1.2, v, true, 2.4, 4.5)[0];
  assert(office(6.75) < office(4.95));
});
