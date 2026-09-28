import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createWater } from '@viewer/effects/water';
test('water remains world anchored, adjusts height, pauses motion and disposes', () => {
  const scene = new THREE.Scene(),
    water = createWater(scene);
  water.configure(new THREE.Box3(new THREE.Vector3(10, -2, 20), new THREE.Vector3(30, 8, 40)), 10);
  water.update(true, 0.2, 0.016);
  assert.deepEqual(water.mesh.position.toArray(), [20, 0, 30]);
  const time = water.mesh.material.uniforms.time.value;
  water.update(true, 0.3, 1, true);
  assert.equal(water.mesh.position.y, 1);
  assert.equal(water.mesh.material.uniforms.time.value, time);
  water.update(false, 0.3, 1);
  assert(!water.mesh.visible);
  assert.equal(water.mesh.material.uniforms.time.value, time);
  water.dispose();
  assert.equal(scene.children.length, 0);
});
