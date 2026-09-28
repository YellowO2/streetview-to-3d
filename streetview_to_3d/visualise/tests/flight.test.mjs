import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { advanceFlight, FLIGHT } from '@viewer/flight-motion';
import { createBird } from '@viewer/bird';
test('flight acceleration and braking are consistent across frame rates', () => {
  const results = [30, 60, 144].map((fps) => {
    const p = new THREE.Vector3(),
      v = new THREE.Vector3(),
      target = new THREE.Vector3(0, 0, -FLIGHT.speed);
    for (let i = 0; i < fps * 2; i++) advanceFlight(p, v, target, 1 / fps);
    assert(v.length() <= FLIGHT.speed);
    for (let i = 0; i < fps * 2; i++) advanceFlight(p, v, new THREE.Vector3(), 1 / fps);
    assert(v.length() < 0.001);
    return p.z;
  });
  assert(Math.max(...results) - Math.min(...results) < 1e-9);
});
test('jewel bird uses coloured points with articulated wings and metre-scale span', () => {
  const model = createBird();
  let count = 0;
  model.bird.scale.setScalar(FLIGHT.birdScale);
  model.bird.traverse((o) => {
    assert(!o.isMesh);
    if (o.isPoints) {
      count += o.geometry.attributes.position.count;
      assert(o.geometry.attributes.color);
    }
  });
  assert(count > 3500 && count < 15000);
  const size = new THREE.Box3().setFromObject(model.bird).getSize(new THREE.Vector3());
  assert(size.x > 0.7 && size.x < 1);
  model.animate(0.1, true);
  assert.notEqual(model.wings[0].rotation.z, 0);
  assert(Math.abs(model.wings[0].rotation.z + model.wings[1].rotation.z) < 1e-9);
  model.dispose();
});
