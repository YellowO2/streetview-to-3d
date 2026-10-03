import { test } from 'node:test';
import assert from 'node:assert/strict';
import { viewerTemplate } from './fixture.mjs';
import { JSDOM } from 'jsdom';
import { createUI } from '@viewer/ui/shell';
import { ViewerState } from '@viewer/core/state';
test('user flow: select, switch modes, open settings, regroup, hide and restore', () => {
  const dom = new JSDOM(viewerTemplate());
  globalThis.document = dom.window.document;
  const state = new ViewerState();
  state.regroup([
    [0, 1],
    [2, 3],
  ]);
  const store = {
    name: 'Test',
    group: {},
    data: {},
    placement: 'world',
    nodes: new Map([0, 1, 2, 3].map((i) => [i, {}])),
    undo: [],
    redo: [],
    dirty: false,
  };
  let ui;
  const render = () => ui.render(store, state, { points: 100 });
  const noop = () => {};
  const actions = {
    mode: (mode) => {
      state.mode = mode;
      render();
    },
    select: (members, kind) => {
      state.select(members, kind);
      if (members) state.mode = 'edit';
      render();
    },
    focus: noop,
    visibility: (members) => {
      state.toggleVisibility(members);
      render();
    },
    showAll: () => {
      state.hidden.clear();
      render();
    },
    history: noop,
    reset: noop,
    gps: noop,
    adjust: noop,
    files: noop,
    save: noop,
    recenter: noop,
    settings: noop,
  };
  ui = createUI(actions);
  render();
  const $ = (id) => document.getElementById(id);
  document.querySelector('.piece-select').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  $('view-settings').open = true;
  document.querySelector('#east').value = '2.5';
  assert(!document.querySelector('#edit-tools').hidden);
  assert.equal($('east').value, '2.5');
  assert($('view-settings').open);
  $('inspect').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  assert($('editing').hidden);
  $('fly').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  $('exit-fly').click();
  assert.equal(state.mode, 'inspect');
  document.querySelector('.visibility').click();
  assert(state.hidden.has(0));
  $('show-all').click();
  assert.equal(state.hidden.size, 0);
  state.select(state.groups[0]);
  state.hidden.add(3);
  state.regroup([[0], [1], [2], [3]]);
  render();
  assert.equal($('piece-count').textContent, '(4)');
  assert(state.hidden.has(3));
  assert.equal($('selection-title').textContent, 'Piece 1');
  ui.render(store, state, { busy: true });
  assert($('open-folder').disabled);
  render();
  assert(!$('open-folder').disabled);
  dom.window.close();
});

test('public viewer hides editing and exposes optional view settings', () => {
  const dom = new JSDOM(viewerTemplate());
  globalThis.document = dom.window.document;
  const actions = new Proxy({}, { get: () => () => {} });
  const ui = createUI(actions, { editable: false });
  const state = new ViewerState();
  const store = {
    name: 'Demo',
    group: {},
    data: {},
    placement: 'world',
    nodes: new Map(),
    undo: [],
    redo: [],
    dirty: false,
  };
  ui.render(store, state);
  const $ = (id) => document.getElementById(id);
  assert($('scene-manager').hidden);
  assert($('view-panel').hidden);
  assert(!$('fly').disabled);
  $('toggle-settings').click();
  assert(!$('view-panel').hidden);
  assert($('view-settings').open);
  $('toggle-settings').click();
  assert($('view-panel').hidden);
  dom.window.close();
});

test('editor and demo render identical shared toolbar and view settings', () => {
  const shell = (editable) => {
    const dom = new JSDOM(viewerTemplate());
    globalThis.document = dom.window.document;
    const ui = createUI(new Proxy({}, { get: () => () => {} }), { editable });
    ui.render(
      { name: 'Scene', group: null, data: null, nodes: new Map(), undo: [], redo: [] },
      new ViewerState(),
    );
    const result = ['header', '#view-panel'].map(
      (selector) => document.querySelector(selector).outerHTML,
    );
    dom.window.close();
    return result;
  };
  assert.deepEqual(shell(true), shell(false));
});

test('manager collapses and reopens with accessible focus restoration', () => {
  const dom = new JSDOM(viewerTemplate());
  globalThis.document = dom.window.document;
  createUI({});
  const manager = document.getElementById('scene-manager');
  const close = document.getElementById('collapse-manager');
  const open = document.getElementById('expand-manager');
  close.click();
  assert(manager.hidden);
  assert(!open.hidden);
  assert.equal(document.activeElement, open);
  open.click();
  assert(!manager.hidden);
  assert(open.hidden);
  assert.equal(document.activeElement, close);
  dom.window.close();
});

test('Show buildings in the settings tells the viewer to show the map buildings or not', () => {
  const dom = new JSDOM(viewerTemplate());
  globalThis.document = dom.window.document;
  const said = [];
  createUI({ buildings: (on) => said.push(on) });
  const box = document.getElementById('show-buildings');
  assert(box.checked); // shown to begin with
  box.checked = false;
  box.dispatchEvent(new dom.window.Event('change'));
  box.checked = true;
  box.dispatchEvent(new dom.window.Event('change'));
  assert.deepEqual(said, [false, true]);
  dom.window.close();
});
