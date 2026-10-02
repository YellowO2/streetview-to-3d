import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { advanceFlight, FLIGHT, steerBird } from '@viewer/flight-motion';
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
test('bird is faint dabs with articulated wings and metre-scale span', () => {
  const model = createBird();
  let count = 0;
  model.bird.scale.setScalar(FLIGHT.birdScale);
  model.bird.traverse((o) => {
    assert(!o.isMesh && !o.isLine);
    if (o.isPoints && o.name !== 'Bird trail') {
      assert(o.userData.styleAnimated);
      assert(!o.geometry.attributes.birdAlpha);
      count += o.geometry.attributes.position.count;
    }
  });
  assert(count > 60 && count < 400);
  const size = new THREE.Box3().setFromObject(model.bird).getSize(new THREE.Vector3());
  assert(size.x > 0.6 && size.x < 1); // its dabs' middles; each 0.1 across
  model.animate(0.1, true);
  assert.notEqual(model.wings[0].rotation.z, 0);
  assert(Math.abs(model.wings[0].rotation.z + model.wings[1].rotation.z) < 1e-9);
  model.dispose();
});

test('the trail is laid by distance while flying, fixed in the world, and reset', () => {
  const model = createBird();
  model.bird.scale.setScalar(FLIGHT.birdScale);
  let trail;
  model.bird.traverse((o) => {
    if (o.name === 'Bird trail') trail = o;
  });
  const laid = () => trail.geometry.attributes.born.array.filter((t) => t >= 0).length;
  model.animate(0.02, false);
  assert.equal(laid(), 0);
  model.bird.position.set(0, 0, -2);
  model.animate(0.02, true);
  const first = laid();
  assert(first > 2 * 100);
  // still, however it flaps: nothing
  for (let i = 0; i < 10; i++) model.animate(0.02, false);
  assert.equal(laid(), first);
  model.bird.updateMatrixWorld(true);
  assert(trail.matrixWorld.equals(new THREE.Matrix4()));
  model.resetPlume();
  assert.equal(laid(), 0);
  model.dispose();
});

test('the bird faces where it flies, banks into turns, and faces the camera way when still', () => {
  const q = new THREE.Quaternion(),
    heading = new THREE.Quaternion(),
    state = { yaw: 0, pitch: 0, roll: 0 },
    nose = () => new THREE.Vector3(0, 0, -1).applyQuaternion(q);
  // forward and left together: it turns to fly that way, banking left as it turns
  const diagonal = new THREE.Vector3(-1, 0, -1).normalize().multiplyScalar(2);
  steerBird(q, diagonal, heading, state, 1 / 60);
  steerBird(q, diagonal, heading, state, 1 / 60);
  assert(state.roll > 0);
  for (let i = 0; i < 180; i++) steerBird(q, diagonal, heading, state, 1 / 60);
  assert(nose().angleTo(diagonal) < 0.05);
  assert(Math.abs(state.roll) < 0.05);
  // climbing: nose up, never past its limit
  for (let i = 0; i < 180; i++) steerBird(q, new THREE.Vector3(0, 5, -0.1), heading, state, 1 / 60);
  assert(state.pitch > 0.5 && state.pitch <= 0.6 + 1e-9);
  // still: back to the camera's way, level
  for (let i = 0; i < 300; i++) steerBird(q, new THREE.Vector3(), heading, state, 1 / 60);
  assert(nose().angleTo(new THREE.Vector3(0, 0, -1)) < 0.05);
});
