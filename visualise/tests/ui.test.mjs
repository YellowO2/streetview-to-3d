import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { createUI } from '@viewer/ui';
import { ViewerState } from '@viewer/state';
test('user flow: select, switch modes, open settings, regroup, isolate, hide and restore', () => {
  const dom = new JSDOM(
    readFileSync(new URL('../viewer_src/template.html', import.meta.url), 'utf8'),
  );
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
  let focused = false,
    ui;
  const render = () => ui.render(store, state, { points: 100 });
  const noop = () => {};
  const actions = {
    mode: (mode) => {
      state.mode = mode;
      render();
    },
    select: (members) => {
      state.select(members);
      render();
    },
    focus: () => (focused = true),
    visibility: (members) => {
      state.toggleVisibility(members);
      render();
    },
    showAll: () => {
      state.hidden.clear();
      state.isolated = false;
      render();
    },
    isolate: (v) => {
      state.isolated = v;
      render();
    },
    group: () => {},
    tool: (t) => {
      state.tool = t;
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
  $('edit').click();
  assert(!document.querySelector('#edit-tools').hidden);
  assert.equal($('east').value, '2.5');
  assert($('view-settings').open);
  $('inspect').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  assert($('editing').hidden);
  $('fly').click();
  assert.equal($('selection-title').textContent, 'Piece 1');
  assert($('confidence').disabled);
  assert($('isolate').disabled);
  $('exit-fly').click();
  assert.equal(state.mode, 'inspect');
  assert(!$('isolate').disabled);
  $('isolate').click();
  assert(state.isolated);
  $('clear-selection').click();
  assert(!state.isolated);
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
  $('focus').click();
  assert(focused);
  ui.render(store, state, { busy: true });
  assert($('open-folder').disabled);
  assert($('edit').disabled);
  render();
  assert(!$('open-folder').disabled);
  dom.window.close();
});

test('public viewer hides editing and exposes optional view settings', () => {
  const dom = new JSDOM(
    readFileSync(new URL('../viewer_src/template.html', import.meta.url), 'utf8'),
  );
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
  for (const id of ['edit', 'library', 'history', 'inspector']) assert($(id).hidden);
  assert($('edit').disabled);
  assert(document.querySelector('.save').hidden);
  assert(!$('fly').disabled);
  $('toggle-settings').click();
  assert(!$('inspector').hidden);
  assert($('view-settings').open);
  $('toggle-settings').click();
  assert($('inspector').hidden);
  dom.window.close();
});
