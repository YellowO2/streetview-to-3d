import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createBirdTrails } from '@viewer/bird-trails';
test('trails retain previous world positions and reset without teleport streaks', () => {
  const bird = new THREE.Group();
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([1, 0, 0], 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([1, 1, 1], 3));
  const cloud = new THREE.Points(geometry);
  bird.add(cloud);
  const trail = createBirdTrails(bird, [cloud]);
  trail.update(1 / 30);
  bird.position.x = 2;
  trail.update(1 / 30);
  const lines = bird.children.find((c) => c.isLineSegments);
  assert.equal(lines.geometry.drawRange.count, 2);
  assert.deepEqual(
    Array.from(lines.geometry.attributes.position.array.slice(0, 6)),
    [3, 0, 0, 1, 0, 0],
  );
  bird.updateMatrixWorld(true);
  assert.equal(lines.matrixWorld.elements[12], 0);
  trail.reset();
  assert.equal(lines.geometry.drawRange.count, 0);
  trail.update(1 / 30);
  assert.equal(lines.geometry.drawRange.count, 0);
  trail.dispose();
  assert.equal(lines.parent, null);
});
