import { test } from 'node:test';
import assert from 'node:assert/strict';
import { panoramaStart } from '@viewer/start-view';

test('starts one metre above the nearest panorama after point-cloud placement', () => {
  const result = panoramaStart({
    center: [1, 103],
    nodes: [
      { ply: 'far.ply', position: [9, 9, 9], pano: { lat: 2, lon: 104 } },
      {
        ply: 'near.ply',
        position: [1, 2, 3],
        pano: { lat: 1, lon: 103, heading: 0 },
        transform: [
          [2, 0, 0, 10],
          [0, 2, 0, 20],
          [0, 0, 2, 30],
          [0, 0, 0, 1],
        ],
      },
    ],
  });
  assert.deepEqual(result.position.toArray(), [12, -23, -36]);
  assert.deepEqual(result.target.toArray(), [12, -23, -37]);
});
test('missing camera positions allow the viewer bounds fallback', () => {
  assert.equal(panoramaStart(null), null);
  assert.equal(panoramaStart({ nodes: [{ ply: 'a.ply' }] }), null);
});
