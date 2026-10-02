import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { marks, blockPoints, buildingPoints } from '@viewer/effects/blocks';

// a wall length (12) m along, 9 m up, facing south (two triangles, its facade and windows' colour given),
// and a flat roof 12 x 6 m on it; each spaced 0.5 m
const GLASS = [0.25, 0.45, 0.65];
function building(length = 12, glass = true) {
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
  if (glass)
    g.setAttribute(
      'glass',
      new THREE.Float32BufferAttribute(
        P.flatMap(() => GLASS),
        3,
      ),
    );
  g.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
  return g;
}

test('a building is built of points: over its faces, along its edges', () => {
  const s = marks(building());
  const spacing = 0.5 * 1.2;
  const faces = s.dab.filter((d) => Math.abs(d - spacing * 1.6) < 1e-6).length,
    edges = s.dab.filter((d) => Math.abs(d - spacing * 0.9) < 1e-6).length;
  assert.equal(faces + edges, s.dab.length);
  assert.ok(Math.abs(faces - (12 * 9 + 12 * 6) / spacing ** 2) < 0.1 * faces); // one a grid square
  assert.ok(edges > 0); // smaller, along its edges
  for (let i = 0; i < s.dab.length; i++) {
    const [x, y, z] = s.centre.slice(3 * i, 3 * i + 3);
    assert.ok(x >= -1e-3 && x <= 12.001 && y >= -1e-3 && y <= 9.001 && z <= 1e-3 && z >= -6.001);
    if (s.dab[i] < spacing) assert.ok(y > 0.05 || Math.abs(x) < 1e-3 || Math.abs(x - 12) < 1e-3); // not at its foot
  }
});

test('a window is nothing of its own: the points of its wall on it, in the colour its facade gives', () => {
  const s = marks(building());
  const onGlass = (i) => GLASS.every((c, d) => Math.abs(s.tint[3 * i + d] - c) < 1e-6);
  const windows = [];
  for (let i = 0; i < s.dab.length; i++) if (onGlass(i)) windows.push(i);
  assert.ok(windows.length > 12); // several points a window
  for (const i of windows) {
    assert.equal(s.dab[i], 0.5 * 1.2 * 1.6); // as big as the wall's
    const [x, y] = s.centre.slice(3 * i, 3 * i + 2);
    const [qx, qy] = [x / 3 - Math.floor(x / 3), y / 3.2 - Math.floor(y / 3.2)];
    assert.ok(qx >= 0.3 - 1e-6 && qx <= 0.7 + 1e-6 && qy >= 0.3 - 1e-6 && qy <= 0.8 + 1e-6);
  }
  // no colour of its own: a facade giving none, no windows
  const plain = marks(building(12, false));
  assert.ok(plain.tint.every((c, i) => Math.abs(c - [0.8, 0.7, 0.6][i % 3]) < 1e-6));
});

test('drawn as the GPU points, a few bytes each', () => {
  const points = blockPoints(building());
  assert.ok(points.isPoints && points.material.isShaderMaterial);
  const g = points.geometry;
  assert.equal(g.getAttribute('tint').array.BYTES_PER_ELEMENT, 1); // colour and facing as bytes
  assert.equal(g.getAttribute('facing').array.BYTES_PER_ELEMENT, 1);
  assert.ok(g.getAttribute('near').array.every((t) => t === 0)); // far off: none near DA3
  assert(points.userData.pointStyle && points.material.uniforms.styleTime); // moved as DA3's points are
});

test('a building DA3 reaches: its points as they are, at the edges smaller', () => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, 1, 0, 1, 1, 0, 0, 2, 0], 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute([0, 0, 1, 0, 0, 1, 0, 0, 1], 3));
  g.setAttribute(
    'color',
    new THREE.Float32BufferAttribute([0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6], 3),
  );
  g.setAttribute('gap', new THREE.Float32BufferAttribute([0.5, 0.5, 0.25], 1));
  g.setAttribute('kind', new THREE.Float32BufferAttribute([0, 0, 1], 1));
  g.setAttribute('near', new THREE.Float32BufferAttribute([0, 0.5, 1], 1));
  const points = buildingPoints(g);
  const at = points.geometry.attributes;
  assert.equal(at.position.count, 3);
  assert.deepEqual(Array.from(at.near.array), [0, 128, 255]); // how near DA3
  const byte = (c) => Math.round(Math.fround(c) * 255);
  assert.deepEqual(
    Array.from(at.tint.array),
    [0.8, 0.7, 0.6, 0.8, 0.7, 0.6, 0.8, 0.7, 0.6].map(byte),
  ); // its own colours
  assert.deepEqual(Array.from(at.dab.array), [0.5 * 1.6, 0.5 * 1.6, 0.25 * 0.9].map(Math.fround)); // an edge's smaller
});
