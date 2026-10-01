import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { strokes, blocksStrokes } from '@viewer/effects/blocks';

// a wall length (12) m along, 9 m up, facing south (two triangles, its facade given),
// and a flat roof 12 x 6 m on it; each spaced 0.5 m
function building(length = 12) {
  const g = new THREE.BufferGeometry();
  const P = [
    [0, 0, 0],
    [length, 0, 0],
    [length, 9, 0],
    [0, 9, 0], // the wall
    [0, 9, 0],
    [length, 9, 0],
    [length, 9, -6],
    [0, 9, -6], // the roof
  ];
  g.setAttribute('position', new THREE.Float32BufferAttribute(P.flat(), 3));
  g.setAttribute(
    'color',
    new THREE.Float32BufferAttribute(
      P.flatMap(() => [0.8, 0.7, 0.6]),
      3,
    ),
  );
  g.setAttribute(
    'gap',
    new THREE.Float32BufferAttribute(
      P.map(() => 0.5),
      1,
    ),
  );
  const roof = -1000;
  g.setAttribute(
    'facade',
    new THREE.Float32BufferAttribute(
      [0, 0, length, 0, length, 9, 0, 9, 0, roof, 0, roof, 0, roof, 0, roof],
      2,
    ),
  );
  g.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
  return g;
}

test('a building is built of strokes: over its faces, along its edges', () => {
  const s = strokes(building());
  const kind = (k) => s.shape.filter((_, i) => i % 4 === 2 && s.shape[i] === k).length;
  const spacing = 0.5 * 1.2;
  const faces = kind(0) + kind(2);
  assert.ok(Math.abs(faces - (12 * 9 + 12 * 6) / spacing ** 2) < 0.1 * faces); // one a grid square
  assert.ok(kind(1) > 0);
  for (let i = 0; i < s.shape.length / 4; i++) {
    const [x, y, z] = s.centre.slice(3 * i, 3 * i + 3);
    assert.ok(x >= -1e-3 && x <= 12.001 && y >= -1e-3 && y <= 9.001 && z <= 1e-3 && z >= -6.001);
  }
  // across the wall, east-west on the flat roof; edges not at its foot
  for (let i = 0; i < s.shape.length / 4; i++) {
    if (s.shape[4 * i + 2] === 1)
      assert.ok(s.centre[3 * i + 1] > 0.05 || Math.abs(s.along[3 * i + 1]) > 0.5);
    else assert.ok(Math.abs(s.along[3 * i] - 1) < 1e-6);
  }
});

test('a window is nothing of its own: the strokes of its wall on it, as big as the rest', () => {
  const s = strokes(building());
  const windows = [];
  for (let i = 0; i < s.shape.length / 4; i++) if (s.shape[4 * i + 2] === 2) windows.push(i);
  assert.ok(windows.length > 12); // several strokes a window
  for (const i of windows) {
    assert.equal(s.shape[4 * i + 1], 0.5 * 1.2 * 1.2); // a wall stroke's width
    const [x, y] = s.centre.slice(3 * i, 3 * i + 2);
    const [qx, qy] = [x / 3 - Math.floor(x / 3), y / 3.2 - Math.floor(y / 3.2)];
    assert.ok(qx >= 0.3 - 1e-6 && qx <= 0.7 + 1e-6 && qy >= 0.3 - 1e-6 && qy <= 0.8 + 1e-6);
  }
});

test('the strokes drawn as one instanced mesh', () => {
  const mesh = blocksStrokes(building());
  assert.ok(mesh.isMesh && mesh.geometry.isInstancedBufferGeometry);
  assert.equal(mesh.geometry.instanceCount, mesh.geometry.getAttribute('shape').count);
});
