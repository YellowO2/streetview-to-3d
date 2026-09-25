import { test } from 'node:test';
import assert from 'node:assert/strict';
import { viewerTemplate } from './fixture.mjs';
import { register } from 'node:module';
import { JSDOM } from 'jsdom';
register('./mock-renderer-loader.mjs', import.meta.url);
const waitFor = async (fn) => {
  for (let i = 0; i < 100; i++) {
    if (fn()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  assert.fail('Application did not reach expected state');
};
test('assembled app: load, GPS prepare, regroup, select, adjust, undo, mode switches, failed flight, reload', async () => {
  const dom = new JSDOM(viewerTemplate(), { url: 'https://viewer.test/' });
  const win = dom.window;
  Object.assign(globalThis, {
    window: win,
    document: win.document,
    devicePixelRatio: 1,
    ResizeObserver: class {
      constructor(fn) {
        this.fn = fn;
      }
      observe() {
        this.fn();
      }
    },
    addEventListener: win.addEventListener.bind(win),
    confirm: () => true,
  });
  await import('@viewer/app');
  const $ = (id) => document.getElementById(id);
  assert($('status').hidden);
  assert($('editing').hidden);
  const data = {
    center: [1, 103],
    nodes: [0, 1, 2, 3].map((i) => ({
      pano: { source: 'test', id: String(i), lat: 1, lon: 103 + i * 0.0001 },
      ply: `node_${i}.ply`,
      position: [i, 0, 0],
      transform: null,
    })),
    adjacency: {},
    edges: [
      { a: 0, b: 1, keep_a: [10, 12] },
      { a: 1, b: 2, keep_a: [9, 12] },
      { a: 2, b: 3, keep_a: [10, 12] },
    ],
  };
  const ply =
    'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n1 2 3\n2 3 4\n';
  const files = [
    new File([JSON.stringify(data)], 'scene.json'),
    ...data.nodes.map((n) => new File([ply], n.ply)),
  ];
  Object.defineProperty($('folder'), 'files', { configurable: true, value: files });
  $('folder').dispatchEvent(new win.Event('change'));
  await waitFor(() => $('scene-info').textContent === '8 points' && $('status').hidden);
  document.querySelector('.piece-select').click();
  assert(!$('prepare').hidden);
  $('prepare-gps').click();
  assert($('prepare').hidden);
  assert.match($('edit-state').textContent, /Unsaved/);
  $('confidence').value = '76';
  $('confidence').dispatchEvent(new win.Event('input'));
  assert.equal($('piece-count').textContent, '(2)');
  document.querySelector('.piece-select').click();
  assert(!$('edit-tools').hidden);
  $('east').value = '2';
  $('apply').click();
  assert(!$('undo').disabled);
  $('inspect').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  document.querySelector('.piece-select').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  $('undo').click();
  assert(!$('redo').disabled);
  $('redo').click();
  assert(!$('undo').disabled);
  document.querySelector('.disclosure').click();
  document.querySelector('.node-select').click();
  assert.equal($('selection-title').textContent, 'Node 0');
  assert.equal(document.body.dataset.mode, 'edit');
  $('east').value = '3';
  $('apply').click();
  assert.equal($('east').value, '0');
  $('confidence').value = '0';
  $('confidence').dispatchEvent(new win.Event('input'));
  assert.equal($('selection-title').textContent, 'Node 0');
  document.querySelector('.piece-select').click();
  $('view-settings').open = true;
  $('rotate').click();
  $('inspect').click();
  document.querySelector('.piece-select').click();
  assert.equal($('rotate').getAttribute('aria-pressed'), 'true');
  assert($('view-settings').open);
  // Reproduce denied pointer lock without a browser. The app must recover to Inspect.
  document.querySelector('canvas').requestPointerLock = () => {
    document.dispatchEvent(new win.Event('pointerlockerror'));
    return Promise.resolve();
  };
  document.exitPointerLock = () => {};
  $('fly').click();
  assert.equal(document.body.dataset.mode, 'inspect');
  assert.equal($('selection-title').textContent, 'Piece 1');
  assert(!$('focus').disabled);
  const drop = new win.Event('drop', { cancelable: true });
  Object.defineProperty(drop, 'dataTransfer', {
    value: { items: [], files: [new File([ply], 'single.ply')] },
  });
  win.dispatchEvent(drop);
  await waitFor(() => $('scene-info').textContent === '2 points' && $('status').hidden);
  assert($('editing').hidden);
  assert.equal($('selection-title').textContent, 'Nothing selected');
  assert($('undo').disabled);
  dom.window.close();
});
