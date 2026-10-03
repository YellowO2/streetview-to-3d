import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { GAPS, level, parseSurface, scatter } from '@viewer/world/scatter';

// A surface as parseSurface gives it: triangles, a colour and gap per corner.
function surface(corners, faces, gap) {
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

test('a building ply is read with its facade, its layout only if it has one', () => {
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
    mesh.dispose();
  }
});
