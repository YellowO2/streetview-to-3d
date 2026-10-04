import { test } from 'node:test';
import assert from 'node:assert/strict';
import { panoramaStart } from '@viewer/core/start-view';

test('a placed scene starts 20 m over the ground at the nearest panorama, looking a little down', () => {
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
  const near = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-9);
  assert(near(result.position.toArray(), [12, -24 + 20 - 2.45, -36]));
  assert(near(result.target.toArray(), [12, -24 + 20 - 2.45 - 0.35, -37]));
});
test('missing camera positions allow the viewer bounds fallback', () => {
  assert.equal(panoramaStart(null), null);
  assert.equal(panoramaStart({ nodes: [{ ply: 'a.ply' }] }), null);
});
