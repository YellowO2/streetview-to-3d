import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ViewerState } from '@viewer/state';
import { scenePieces, placementMode, validateScene } from '@viewer/scene-format';
import { resolveEntries } from '@viewer/files';
const data = {
  center: [1, 103],
  nodes: Array.from({ length: 4 }, (_, i) => ({
    pano: { lat: 1, lon: 103 },
    ply: `node_${i}.ply`,
    position: [i, 0, 0],
    transform: null,
  })),
  edges: [
    { a: 0, b: 1, keep_a: [10, 12], keep_b: [10, 12] },
    { a: 1, b: 2, keep_a: [9, 12], keep_b: [9, 12] },
    { a: 2, b: 3, keep_a: [10, 12], keep_b: [10, 12] },
  ],
};
test('mode switches preserve selected piece, visibility and grouping', () => {
  const s = new ViewerState();
  s.regroup(scenePieces(data, 0.76));
  s.select(s.groups[0]);
  s.hidden.add(3);
  s.tool = 'rotate';
  for (const mode of ['edit', 'fly', 'inspect', 'edit']) {
    s.mode = mode;
    assert.deepEqual(s.selected, [0, 1]);
    assert(s.hidden.has(3));
    assert.equal(s.tool, 'rotate');
  }
});
test('regroup preserves selection anchor and explicit hidden nodes', () => {
  const s = new ViewerState();
  s.regroup(scenePieces(data, 0));
  s.select(s.groups[0]);
  s.hidden.add(3);
  s.regroup(scenePieces(data, 0.76));
  assert.deepEqual(s.selected, [0, 1]);
  assert(s.hidden.has(3));
  s.regroup(scenePieces(data, 0.84));
  assert.deepEqual(s.selected, [0]);
  assert(s.hidden.has(3));
});
test('isolation and hiding are distinct; clearing selection restores only isolation', () => {
  const s = new ViewerState();
  s.regroup(scenePieces(data, 0.76));
  s.hidden.add(3);
  s.select(s.groups[0]);
  s.isolated = true;
  assert(!s.visible(2));
  s.select(null);
  assert(s.visible(2));
  assert(!s.visible(3));
  s.select(s.groups[1]);
  assert(s.visible(3));
  s.toggleVisibility(s.groups[1]);
  assert.equal(s.selected, null);
  assert(!s.visible(2));
});
test('scene format rejects missing placements and bad paths without guessing', () => {
  validateScene(data);
  assert.equal(placementMode(data), 'raw');
  const unplaced = structuredClone(data);
  unplaced.edges = [];
  assert.throws(() => placementMode(unplaced));
  const broken = structuredClone(data);
  broken.nodes[0].ply = '../x.ply';
  assert.throws(() => validateScene(broken));
});
test('nested scene paths resolve relative to JSON, not the current page', () => {
  const json = {},
    ply = {};
  const spec = resolveEntries([
    { path: 'run/scene.json', file: json },
    { path: 'run/nodes/a.ply', file: ply },
  ]);
  assert.equal(spec.resolve('nodes/a.ply'), ply);
  assert.throws(() => spec.resolve('missing.ply'));
  assert.equal(spec.name, 'run');
});

test('individual node selection survives piece regrouping and isolates only that node', () => {
  const s = new ViewerState();
  s.regroup(scenePieces(data, 0));
  s.select([1], 'node');
  s.isolated = true;
  s.regroup(scenePieces(data, 0.84));
  assert.equal(s.selectionKind, 'node');
  assert.deepEqual(s.selected, [1]);
  assert(s.visible(1));
  assert(!s.visible(0));
  s.regroup(scenePieces(data, 0));
  assert.deepEqual(s.selected, [1]);
  s.select(s.groups[0]);
  assert.equal(s.selectionKind, 'piece');
  assert.deepEqual(s.selected, [0, 1, 2, 3]);
});
