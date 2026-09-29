import assert from 'node:assert/strict';
import { test } from 'node:test';
import { tickWater, waterSurfaces } from '@viewer/effects/water';
test('water surfaces lie flat at their level, east/north turned into the viewer frame', () => {
  const [lake] = waterSurfaces({
    surfaces: [
      {
        level: 2,
        outer: [
          [0, 0],
          [10, 0],
          [10, 20],
          [0, 20],
        ],
        holes: [
          [
            [4, 4],
            [6, 4],
            [6, 6],
          ],
        ],
      },
      { level: NaN, outer: [] },
    ],
  });
  const box = lake.geometry.boundingBox;
  assert.deepEqual(box.min.toArray(), [0, 2, -20]);
  assert.deepEqual(box.max.toArray(), [10, 2, 0]);
  const time = lake.material.uniforms.time.value;
  tickWater(0.05);
  assert(lake.material.uniforms.time.value > time);
  assert.equal(waterSurfaces(null).length, 0);
});
