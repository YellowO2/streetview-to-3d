import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { landSurface, MIN_GAP } from '@viewer/effects/land';

// 100 m square in the viewer's frame (y up), two triangles, wound downwards
function field() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([0, 2, 0, 0, 2, 100, 100, 2, 100, 100, 2, 0], 3),
  );
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}

test('the land is one painted surface, its patches as the points are spaced', () => {
  const land = landSurface(field(), (x) => (x < 50 ? 0.01 : 2));
  assert.ok(land.isMesh && land.material.isShaderMaterial);
  const gap = land.geometry.getAttribute('gap'),
    p = land.geometry.getAttribute('position'),
    n = land.geometry.getAttribute('normal');
  for (let i = 0; i < p.count; i++) {
    assert.equal(gap.getX(i), Math.fround(p.getX(i) < 50 ? MIN_GAP : 2)); // never under MIN_GAP
    assert.ok(n.getY(i) > 0.999); // up, whichever way the triangles wind
  }
  assert.equal(land.geometry.getAttribute('color').count, p.count); // plain if it had none
});
