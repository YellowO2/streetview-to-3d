import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { parseBlocks, scatterBlocks } from '@viewer/effects/blocks';

// One triangle as postprocess/ply_io.write_mesh writes it.
function ply(points, facade) {
  const header =
    'ply\nformat binary_little_endian 1.0\nelement vertex 3\n' +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    'property float facade_u\nproperty float facade_v\n' +
    'element face 1\nproperty list uchar int vertex_indices\nend_header\n';
  const head = new TextEncoder().encode(header);
  const body = new DataView(new ArrayBuffer(3 * 23 + 13));
  points.forEach(([x, y, z], i) => {
    const o = i * 23;
    body.setFloat32(o, x, true);
    body.setFloat32(o + 4, y, true);
    body.setFloat32(o + 8, z, true);
    body.setUint8(o + 12, 200);
    body.setUint8(o + 13, 100);
    body.setUint8(o + 14, 50);
    body.setFloat32(o + 15, facade[i][0], true);
    body.setFloat32(o + 19, facade[i][1], true);
  });
  body.setUint8(69, 3);
  [0, 1, 2].forEach((v, i) => body.setInt32(70 + 4 * i, v, true));
  const out = new Uint8Array(head.length + body.byteLength);
  out.set(head);
  out.set(new Uint8Array(body.buffer), head.length);
  return out.buffer;
}

test('far buildings come as triangles and are drawn as points over them, a layer behind each wall', () => {
  const flip = new THREE.Matrix4().makeScale(1, -1, -1);
  // a wall 10 m long, 6 m high, running east (so its inside is north: -z in the viewer)
  const triangles = parseBlocks(
    ply(
      [
        [0, 0, 0],
        [10, 0, 0],
        [10, -6, 0],
      ],
      [
        [0, 0],
        [10, 0],
        [10, 6],
      ],
    ),
    flip,
  );
  assert.equal(triangles.index.count, 3);
  assert.deepEqual(Array.from(triangles.getAttribute('facade').array), [0, 0, 10, 0, 10, 6]);
  const points = scatterBlocks(triangles, () => 0.5);
  const p = points.geometry.getAttribute('position');
  const front = [],
    behind = [];
  for (let i = 0; i < p.count; i++) (p.getZ(i) === 0 ? front : behind).push(i);
  assert(Math.abs(front.length - 30 / 0.25) < 15); // 30 m2, a point per 0.25 m2
  assert(behind.length > 0 && behind.length < front.length);
  for (const i of behind) assert(p.getZ(i) < 0); // inside, not in front
  assert(p.getY(front[0]) >= 0 && p.getY(front[0]) <= 6); // y up in the viewer
  assert.deepEqual(
    scatterBlocks(triangles, () => 0.5).geometry.getAttribute('position').array,
    p.array,
  );
  triangles.dispose();
  points.geometry.dispose();
  points.material.dispose();
});
