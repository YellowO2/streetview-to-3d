import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { cloudRows, createClouds, inside } from '@viewer/world/clouds';

const cloudField = (coverage) => {
  const rows = cloudRows(coverage);
  for (;;) {
    const { done, value } = rows.next();
    if (done) return value;
  }
};

test('clouds are shaped by noise: a flat base, a field round each surface, the same every time', () => {
  const a = cloudField(0.45);
  assert.deepEqual(a.position, cloudField(0.45).position);
  const n = a.position.length / 3;
  assert(n > 10000 && n < 400000);
  for (let i = 0; i < n; i += 97) {
    const [x, y, z] = a.position.slice(3 * i, 3 * i + 3);
    const [fx, fy, fz, deep] = a.facing.slice(4 * i, 4 * i + 4);
    assert(Math.abs(Math.hypot(fx, fy, fz) - 1) < 1e-5); // a way out
    assert(deep >= -50 && deep <= 105); // near its surface
    assert(y >= 1000 - 80 && Math.abs(x) <= 6000 + 45 && Math.abs(z) <= 6000 + 45); // give or take its jitter
  }
  assert(inside(0, 900, 0, 1) < 0); // nothing under the base
  // more coverage, more cloud
  assert(cloudField(0.8).position.length > a.position.length);
  // the field repeats every 12 km: no seam where it wraps
  for (const [x, z] of [
    [1000, 2000],
    [-3000, 500],
  ])
    for (const y of [1050, 1300])
      assert(Math.abs(inside(x, y, z, 0.6) - inside(x + 12000, y, z - 12000, 0.6)) < 1e-6);
});

test('clouds are worked out a little each frame, drift round the camera back to front, and are disposed', () => {
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(60, 1, 0.2, 1e6);
  const clouds = createClouds(scene);
  const points = scene.children[0];
  assert(points.isPoints && !points.visible);
  clouds.update(true, 0, new THREE.PerspectiveCamera(60, 1, 0.01, 100)); // too small a world
  assert(!points.visible);
  camera.position.set(50000, 10, -30000);
  let frames = 0;
  for (let t = 0; !points.visible && frames < 10000; t += 1 / 60, frames++)
    clouds.update(true, t, camera);
  assert(points.visible && frames > 1); // not all in one frame
  const p = points.geometry.getAttribute('position'),
    order = points.geometry.index.array;
  assert.equal(new Set(order).size, order.length); // every dab once
  const drift = points.material.uniforms.drift.value;
  const wrap = (x) => x - 12000 * Math.floor(x / 12000 + 0.5);
  const away = (i) =>
    Math.hypot(
      wrap(p.getX(i) + drift.x - camera.position.x),
      p.getY(i) - camera.position.y,
      wrap(p.getZ(i) + drift.y - camera.position.z),
    );
  assert(away(order[0]) > away(order[order.length - 1]));
  let disposed = 0;
  for (const resource of [points.geometry, points.material])
    resource.addEventListener('dispose', () => disposed++);
  clouds.dispose();
  assert.equal(disposed, 2);
  assert.equal(scene.children.length, 0);
});
