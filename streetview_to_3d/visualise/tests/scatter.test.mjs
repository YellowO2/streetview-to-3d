import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { parseSurface } from '@viewer/world/scatter';

test('a building ply is read with its facade, its layout only if it has one', () => {
  const ply = (layout) =>
    new TextEncoder().encode(
      [
        'ply',
        'format ascii 1.0',
        'element vertex 4',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'property float facade_u',
        'property float facade_v',
        ...(layout ? ['property float facade_bay', 'property float facade_floor'] : []),
        'property float gap',
        'element face 2',
        'property list uchar int vertex_indices',
        'end_header',
        ...[
          [0, 0],
          [12, 0],
          [12, 9],
          [0, 9],
        ].map(([u, v]) => `${u} ${v} 0 204 179 153 ${u} ${v} ${layout ? '2.4 4.5 ' : ''}0.4`),
        '3 0 1 2',
        '3 0 2 3',
        '',
      ].join('\n'),
    ).buffer;
  for (const layout of [false, true]) {
    const mesh = parseSurface(ply(layout), new THREE.Matrix4(), 'building');
    assert.equal(mesh.getAttribute('facade').itemSize, 2);
    assert.equal(Boolean(mesh.getAttribute('facadeLayout')), layout);
    mesh.dispose();
  }
});
