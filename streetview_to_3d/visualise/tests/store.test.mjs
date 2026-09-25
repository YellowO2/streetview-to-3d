import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { SceneStore, parsePoints, loadAsset } from '@viewer/scene-store';
const flip = new THREE.Matrix4().makeScale(1, -1, -1);
const T = [
  [1.2, 0, 0.3, 10],
  [0, 1.5, 0.2, -3],
  [-0.4, 0, 1.2, 20],
  [0, 0, 0, 1],
];
const buffer = new TextEncoder().encode(
  'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n2 3 4\n',
).buffer;
function fixture() {
  const group = new THREE.Group(),
    data = {
      center: [1, 103],
      nodes: [0, 1].map((i) => ({
        pano: { lat: 1, lon: 103 + i * 0.0001 },
        ply: `node_${i}.ply`,
        position: [i, 0, 0],
        transform: structuredClone(T),
      })),
      edges: [{ a: 0, b: 1 }],
    };
  data.nodes.forEach((n, i) => {
    const points = parsePoints(buffer, n.transform);
    points.userData.nodeIndex = i;
    group.add(points);
  });
  const store = new SceneStore();
  store.install({ group, data, placement: 'world' }, 'Test');
  return store;
}
test('edit/export/reload preserves affine placement, other nodes, poses and metadata', () => {
  const store = fixture(),
    before = store.snapshot(),
    delta = new THREE.Matrix4()
      .makeTranslation(2, 3, -4)
      .multiply(new THREE.Matrix4().makeRotationY(0.25));
  store.preview(delta, before, [0]);
  store.commit(before);
  assert(store.dirty);
  const exported = store.exported(),
    original = new THREE.Matrix4().set(...T.flat()),
    point = new THREE.Vector3(1, 2, 3);
  const visible = point.clone().applyMatrix4(original).applyMatrix4(flip).applyMatrix4(delta);
  const reloaded = point
    .clone()
    .applyMatrix4(new THREE.Matrix4().set(...exported.nodes[0].transform.flat()))
    .applyMatrix4(flip);
  assert(visible.distanceTo(reloaded) < 1e-12);
  assert.deepEqual(exported.nodes[1].transform, T);
  assert.deepEqual(exported.nodes[0].position, [0, 0, 0]);
  assert.deepEqual(exported.edges, store.data.edges);
  store.history();
  assert(!store.dirty);
  store.history(true);
  assert(store.dirty);
  store.resetPiece([0]);
  assert(!store.dirty);
  store.history();
  assert(store.dirty);
  store.markExportRequested();
  assert(!store.dirty);
  store.history();
  assert(store.dirty);
});
test('cancelled and failed loads leave an installed scene intact', async () => {
  const store = fixture(),
    original = store.group;
  const cancelled = await loadAsset(
    { arrayBuffer: async () => buffer },
    null,
    () => {},
    () => true,
  );
  assert.equal(cancelled, null);
  assert.equal(store.group, original);
  await assert.rejects(() =>
    loadAsset(
      { arrayBuffer: async () => new ArrayBuffer(0) },
      null,
      () => {},
      () => false,
    ),
  );
  assert.equal(store.group, original);
});
test('raw GPS preparation keeps points and exported transform equivalent', () => {
  const store = fixture();
  store.data.nodes.forEach((n) => (n.transform = null));
  store.placement = 'raw';
  // Replace the fixture's baked world geometry with raw geometry.
  store.nodes.forEach((o) => {
    o.geometry.dispose();
    o.geometry = parsePoints(buffer).geometry;
  });
  store.prepareGPS(1.46);
  assert.equal(store.placement, 'world');
  assert(store.dirty);
  const p = store.nodes.get(0).geometry.getAttribute('position');
  const expected = new THREE.Vector3(1, 2, 3)
    .applyMatrix4(new THREE.Matrix4().set(...store.exported().nodes[0].transform.flat()))
    .applyMatrix4(flip);
  assert(new THREE.Vector3().fromBufferAttribute(p, 0).distanceTo(expected) < 1e-5);
});
