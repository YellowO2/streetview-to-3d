import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { voxelCells, createVoxels } from '@viewer/effects/voxel';
function cloud() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([-1, 0, 0, -0.9, 0, 0, 1, 0, 0], 3),
  );
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([1, 0, 0, 0, 0, 1, 0, 1, 0], 3));
  return new THREE.Points(geometry, new THREE.PointsMaterial());
}
test('voxel cells average colours, bound count and preserve source arrays', () => {
  const source = cloud(),
    before = source.geometry.attributes.position.array.slice();
  const result = voxelCells(source.geometry, 0.5);
  assert.equal(result.cells.length, 2);
  assert.equal(result.cells[0].r / result.cells[0].count, 0.5);
  assert.equal(result.cells[0].b / result.cells[0].count, 0.5);
  assert.equal(voxelCells(source.geometry, 0.01, 1).cells.length, 1);
  assert.deepEqual(source.geometry.attributes.position.array, before);
});
test('voxel proxies follow transforms and visibility, restore on errors and dispose', () => {
  const scene = new THREE.Scene(),
    group = new THREE.Group(),
    source = cloud();
  group.add(source);
  scene.add(group);
  const voxels = createVoxels(scene);
  voxels.configure(group);
  group.position.x = 12;
  voxels.render(true, 0.5, () => {
    assert(!source.visible);
    const proxy = scene.children.find((o) => o !== group).children[0];
    assert.equal(proxy.matrix.elements[12], 12);
    assert.equal(proxy.count, 2);
  });
  assert(source.visible);
  assert.throws(() =>
    voxels.render(true, 0.5, () => {
      throw Error('draw');
    }),
  );
  assert(source.visible);
  group.visible = false;
  voxels.render(true, 0.5, () =>
    assert(!scene.children.find((o) => o !== group).children[0].visible),
  );
  voxels.render(false, 0.5, () => assert(!scene.children.find((o) => o !== group).visible));
  voxels.dispose();
  assert.equal(scene.children.length, 1);
});
