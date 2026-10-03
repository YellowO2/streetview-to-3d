import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { exportPly } from '@viewer/core/export';
import { LAND, WATER, LIFE } from '@viewer/core/scene-format';

function points(position, attributes = {}, userData = {}) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  for (const [name, a] of Object.entries(attributes)) g.setAttribute(name, a);
  const p = new THREE.Points(g, new THREE.PointsMaterial());
  Object.assign(p.userData, userData);
  return p;
}

// a DA3 piece, the land, water and a car, in the viewer frame (east, up, south)
function scene() {
  const group = new THREE.Group();
  const grey = new THREE.Color().setRGB(128 / 255, 128 / 255, 128 / 255, THREE.SRGBColorSpace);
  group.add(
    points(
      [1, 2, 3],
      { color: new THREE.Float32BufferAttribute(grey.toArray(), 3) },
      { nodeIndex: 0 },
    ),
  );
  group.add(
    points(
      [0, 0, 0],
      { tint: new THREE.BufferAttribute(new Uint8Array([255, 0, 0]), 3, true) },
      {
        surroundings: LAND,
      },
    ),
  );
  const water = new THREE.Group();
  water.userData.surroundings = WATER;
  water.position.set(0, 5, 0);
  water.add(points([0, 0, 0], { water: new THREE.Float32BufferAttribute([0, 1], 2) }));
  group.add(water);
  group.add(points([9, 9, 9], {}, { surroundings: LIFE }));
  return group;
}

async function read(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const text = new TextDecoder().decode(bytes.slice(0, 400));
  const end = text.indexOf('end_header\n') + 'end_header\n'.length;
  const view = new DataView(bytes.buffer, end);
  const rows = [];
  for (let at = 0; at < view.byteLength; at += 15)
    rows.push([
      [0, 4, 8].map((o) => view.getFloat32(at + o, true) + 0), // + 0: -0 as 0,
      [12, 13, 14].map((o) => view.getUint8(at + o)),
    ]);
  return { header: text.slice(0, end), rows };
}

test('exports the chosen parts east, north, up, in sRGB, without life', async () => {
  const { header, rows } = await read(exportPly(scene(), ['street', 'ground'], [1.5, 103.7]));
  assert.match(header, /element vertex 3\n/);
  assert.match(header, /comment origin lat 1.5 lon 103.7/);
  assert.deepEqual(rows[0], [
    [1, -3, 2],
    [128, 128, 128],
  ]);
  assert.deepEqual(rows[1], [
    [0, 0, 0],
    [255, 0, 0],
  ]);
  assert.deepEqual(rows[2][0], [0, 0, 5]); // the water's parent moves it
});

test('leaves out what is not chosen', async () => {
  const { rows } = await read(exportPly(scene(), ['street'], null));
  assert.equal(rows.length, 1);
});
