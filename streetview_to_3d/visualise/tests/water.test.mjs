import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { waterSurfaces, waterGrid } from '@viewer/effects/water';
import { pointMotion } from '@viewer/effects/points';
import { GAPS, level } from '@viewer/effects/scatter';

// a pond 100 m square with an island 20 m square in it, 2 m up
const pond = {
  surfaces: [
    {
      level: 2,
      outer: [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100],
      ],
      holes: [
        [
          [40, 40],
          [60, 40],
          [60, 60],
          [40, 60],
        ],
      ],
    },
  ],
};

test('the water is points on it, none on its island, spaced as asked', () => {
  const grid = waterGrid([...pond.surfaces[0].holes, pond.surfaces[0].outer], () => 1);
  const ins = grid.east.map((e, i) => [e, grid.north[i]]);
  assert(ins.every(([e, n]) => e >= 0 && e <= 100 && n >= 0 && n <= 100));
  assert(!ins.some(([e, n]) => e > 40.5 && e < 59.5 && n > 40.5 && n < 59.5));
  const step = GAPS[level(1)]; // the nearest of GAPS over 1 m
  assert(Math.abs(ins.length - (100 * 100 - 20 * 20) / step ** 2) < 200);
});

test('a level is a mirror at its height, its points left as they are by the style', () => {
  const [water] = waterSurfaces(pond, () => 4);
  assert(water.isReflector);
  water.updateMatrixWorld();
  const at = new THREE.Vector3(30, 70, 0).applyMatrix4(water.matrixWorld);
  assert(at.distanceTo(new THREE.Vector3(30, 2, -70)) < 1e-9); // east, up, south
  const points = water.children.filter((c) => c.isPoints);
  assert(points.length);
  const compile = points[0].material.onBeforeCompile;
  pointMotion(points[0]);
  assert.equal(points[0].material.onBeforeCompile, compile);
  water.dispose();
});
