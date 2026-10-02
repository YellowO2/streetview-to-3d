import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { landPoints, MIN_GAP } from '@viewer/effects/land';
import { covering } from '@viewer/effects/blocks';
import { GAPS, level } from '@viewer/effects/scatter';

// 100 m square in the viewer's frame (y up), two triangles, wound downwards
function field() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute([0, 2, 0, 0, 2, 100, 100, 2, 100, 100, 2, 0], 3),
  );
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}

test('the land is points spaced as the world is, never under MIN_GAP, facing up however wound', () => {
  const land = landPoints(field(), (x) => (x < 50 ? 0.01 : 2));
  const p = land.geometry.getAttribute('position'),
    dab = land.geometry.getAttribute('dab'),
    facing = land.geometry.getAttribute('facing');
  const across = (gap) => Math.fround(GAPS[level(gap)] * covering(0.1)); // land.js SPACE 1, JITTER
  for (let i = 0; i < p.count; i++) {
    // a triangle spaced by its closest corner: the west one's all MIN_GAP
    if (p.getX(i) < 40) assert.equal(dab.getX(i), across(MIN_GAP));
    assert.ok(facing.getY(i) > 0.99); // up: lit as the land faces
  }
});

test('the land as points: spaced by one grid, no edges, fogged', () => {
  const land = landPoints(field(), () => 1);
  assert.ok(land.isPoints && land.material.fog);
  const p = land.geometry.getAttribute('position'),
    facing = land.geometry.getAttribute('facing');
  const s = GAPS[level(1)] * 1; // land.js SPACE
  assert.ok(Math.abs(p.count - (100 / s) ** 2) < 0.1 * p.count); // the grid's, none along its rim
  for (let i = 0; i < p.count; i++) {
    assert.equal(p.getY(i), 2);
    assert.ok(Math.abs(facing.getY(i)) > 0.99); // facing as the land does
  }
});

test('on a slope the grid finer and the dabs bigger, half the stretch each, meeting as on the flat', () => {
  const tilted = field();
  const p = tilted.getAttribute('position');
  for (let i = 0; i < p.count; i++) p.setY(i, p.getZ(i)); // 45 degrees, up toward +z: stretched by sqrt 2
  const flat = landPoints(field(), () => 1).geometry,
    slope = landPoints(tilted, () => 1).geometry;
  const finer = GAPS[level(1 / 2 ** 0.25)] / GAPS[level(1)]; // the grid, a level finer
  assert.ok(finer < 1);
  // as many more as the grid is finer (the same square on the map)
  const more = slope.getAttribute('position').count / flat.getAttribute('position').count;
  assert.ok(Math.abs(more * finer ** 2 - 1) < 0.1);
  // each dab as big as the grid, stretched over the slope, is apart
  const ratio = slope.getAttribute('dab').getX(0) / flat.getAttribute('dab').getX(0);
  assert.ok(Math.abs(ratio - finer * Math.SQRT2) < 1e-3);
});

test('the land as points leaves nothing between its dabs, even where two spacings meet', () => {
  // 100 m square of 10 m cells; closer points on its west half
  const geometry = new THREE.BufferGeometry(),
    position = [],
    index = [];
  for (let i = 0; i <= 10; i++) for (let j = 0; j <= 10; j++) position.push(i * 10, 0, j * 10);
  for (let i = 0; i < 10; i++)
    for (let j = 0; j < 10; j++) {
      const a = i * 11 + j;
      index.push(a, a + 1, a + 12, a, a + 12, a + 11);
    }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
  geometry.setIndex(index);
  const land = landPoints(geometry, (x) => (x < 45 ? 0.5 : 1.2)).geometry;
  const p = land.getAttribute('position'),
    dab = land.getAttribute('dab');
  // each dab at the least it is drawn: the size knob's, its edge its roughest, swelled smaller its most
  const least = 1.1 * Math.sqrt(0.7) * (1 - 0.15 * 0.5);
  const cells = new Map(),
    key = (x, z) => `${Math.floor(x / 2)},${Math.floor(z / 2)}`;
  for (let i = 0; i < p.count; i++) {
    const k = key(p.getX(i), p.getZ(i));
    if (!cells.has(k)) cells.set(k, []);
    cells.get(k).push(i);
  }
  let open = 0;
  for (let x = 5; x < 95; x += 0.37)
    for (let z = 5; z < 95; z += 0.41) {
      let covered = false;
      for (let dx = -2; dx <= 2 && !covered; dx++)
        for (let dz = -2; dz <= 2 && !covered; dz++)
          for (const i of cells.get(key(x + dx * 2, z + dz * 2)) ?? [])
            if (Math.hypot(p.getX(i) - x, p.getZ(i) - z) < (dab.getX(i) / 2) * least) {
              covered = true;
              break;
            }
      if (!covered) open++;
    }
  assert.equal(open, 0);
});

test('nor on a slope, gentle or steep: the dabs grow as the ground stretches past the grid', () => {
  const least = 1.1 * Math.sqrt(0.7) * (1 - 0.15 * 0.5); // a dab at the least it is drawn
  for (const deg of [30, 50, 75]) {
    // 30 m along the slope, rising toward +z, 12 x 12 cells
    const t = Math.tan((deg * Math.PI) / 180),
      run = 30 * Math.cos((deg * Math.PI) / 180),
      position = [],
      index = [];
    for (let i = 0; i <= 12; i++)
      for (let j = 0; j <= 12; j++) position.push(i * 2.5, ((j * run) / 12) * t, (j * run) / 12);
    for (let i = 0; i < 12; i++)
      for (let j = 0; j < 12; j++) {
        const a = i * 13 + j;
        index.push(a, a + 1, a + 14, a, a + 14, a + 13);
      }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(position, 3));
    geometry.setIndex(index);
    const land = landPoints(geometry, () => 1).geometry;
    const p = land.getAttribute('position'),
      dab = land.getAttribute('dab');
    let open = 0;
    for (let x = 3; x < 27; x += 0.31)
      for (let along = 3; along < 27; along += 0.29) {
        const z = (along / 30) * run,
          y = z * t;
        let covered = false;
        for (let i = 0; i < p.count && !covered; i++)
          covered =
            Math.hypot(p.getX(i) - x, p.getY(i) - y, p.getZ(i) - z) < (dab.getX(i) / 2) * least;
        if (!covered) open++;
      }
    assert.equal(open, 0, `${deg} degrees`);
  }
});

test("the land's dabs turn into DA3's points as its corners do: their nearness carried between them", () => {
  const geometry = field();
  // the west edge on DA3's points, the east a metre and more off
  geometry.setAttribute('near', new THREE.Float32BufferAttribute([1, 1, 0, 0], 1));
  const land = landPoints(geometry, () => 1).geometry;
  const p = land.getAttribute('position'),
    near = land.getAttribute('near');
  for (let i = 0; i < p.count; i++)
    assert.ok(Math.abs(near.getX(i) - Math.round((1 - p.getX(i) / 100) * 255) / 255) < 0.01);
});
