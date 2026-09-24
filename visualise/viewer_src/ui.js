const $ = (id) => document.getElementById(id);
const eyeIcon = (visible) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>${visible ? '' : '<path d="m3 3 18 18"/>'}</svg>`;
// Renders UI from state; rendering has no side effects on scene or selection.
// Persistent DOM keeps open menus, keyboard focus and input drafts stable.
export function createUI(actions, { editable = true } = {}) {
  document.body.dataset.editable = String(editable);
  $('toggle-settings').hidden = editable;
  $('edit').hidden = !editable;
  $('library').hidden = !editable;
  $('history').hidden = !editable;
  document.querySelector('.save').hidden = !editable;
  if (!editable) {
    $('selection-title').hidden = $('selection-description').hidden = true;
    $('placement').hidden = true;
    $('inspector').hidden = true;
    $('view-settings').open = true;
  }
  $('toggle-settings').onclick = () => {
    $('inspector').hidden = !$('inspector').hidden;
    $('toggle-settings').setAttribute('aria-pressed', String(!$('inspector').hidden));
  };
  let rows = [],
    lastGroups = null;
  const bind = (id, fn) => ($(id).onclick = fn);
  for (const mode of ['inspect', 'fly', 'edit']) bind(mode, () => actions.mode(mode));
  bind('exit-fly', () => actions.mode('inspect'));
  bind('recenter', actions.recenter);
  bind('open-folder', () => $('folder').click());
  bind('choose', () => $('folder').click());
  bind('open-files', () => $('file').click());
  for (const id of ['file', 'folder'])
    $(id).onchange = () => {
      const files = [...$(id).files];
      $(id).value = '';
      actions.files(files);
    };
  bind('focus', actions.focus);
  bind('clear-selection', () => actions.select(null));
  bind('show-all', actions.showAll);
  $('isolate').onchange = () => actions.isolate($('isolate').checked);
  $('confidence').oninput = () => actions.group(Number($('confidence').value) / 100);
  bind('move', () => actions.tool('translate'));
  bind('rotate', () => actions.tool('rotate'));
  bind('undo', () => actions.history(false));
  bind('redo', () => actions.history(true));
  bind('reset', actions.reset);
  bind('prepare-gps', () => actions.gps(Number($('da3-scale').value)));
  bind('apply', () => {
    const ids = ['east', 'north', 'up', 'turn'],
      values = ids.map((id) => ($(id).value.trim() === '' ? NaN : Number($(id).value)));
    if (actions.adjust(...values)) ids.forEach((id) => ($(id).value = '0'));
  });
  bind('save', actions.save);
  bind('dismiss-notice', () => ($('notice').hidden = true));
  function settings() {
    const point = 2 ** Number($('point-size').value),
      speed = 2 ** Number($('speed').value),
      chase = Number($('chase').value);
    $('point-value').textContent = `${point.toFixed(1)}×`;
    $('speed-value').textContent = `${speed.toFixed(1)}×`;
    $('chase-value').textContent = `${chase.toFixed(1)}×`;
    actions.settings(point, speed, chase);
  }
  for (const id of ['point-size', 'speed', 'chase']) $(id).oninput = settings;
  return {
    notify(message) {
      $('notice-text').textContent = message;
      $('notice').hidden = false;
    },
    progress(message) {
      $('status').textContent = message;
    },
    drop(visible) {
      $('dropzone').hidden = !visible;
    },
    settings,
    pointStep(direction) {
      $('point-size').value = Number($('point-size').value) + direction * 0.2;
      settings();
    },
    render(store, state, { busy = false, dragging = false, points = 0 } = {}) {
      const blocked = busy || dragging,
        selected = state.selected,
        world = store.placement === 'world';
      document.body.dataset.mode = state.mode;
      $('scene-name').textContent = store.name;
      $('empty').hidden = !!store.group;
      $('flight').hidden = $('reticle').hidden = state.mode !== 'fly';
      ['inspect', 'fly', 'edit'].forEach((mode) => {
        $(mode).setAttribute('aria-pressed', String(mode === state.mode));
        $(mode).disabled =
          blocked ||
          (mode === 'fly' && !store.group) ||
          (mode === 'edit' && (!editable || !store.data));
      });
      $('recenter').disabled = blocked || !store.group;
      for (const id of ['open-folder', 'open-files', 'choose', 'save', 'undo', 'redo'])
        $(id).disabled = blocked;
      // Keep controls visible rather than rearranging panels on mode switches.
      for (const control of document.querySelectorAll('aside button,aside input'))
        control.disabled = blocked || state.mode === 'fly';
      $('show-all').disabled = blocked || !store.data || state.mode === 'fly';
      $('grouping').hidden = !store.data;
      $('piece-count').textContent = store.data ? `(${state.groups.length})` : '';
      $('piece-guide').textContent = store.data
        ? 'Click to select · Double-click to focus. Eye icons hide or show pieces.'
        : store.group
          ? 'Open a scene folder to work with pieces.'
          : 'Open a scene to see its pieces.';
      $('confidence').value = String(state.threshold * 100);
      $('confidence-value').textContent = state.threshold
        ? `${Math.round(state.threshold * 100)}%`
        : 'All links';
      if (lastGroups !== state.groups) {
        lastGroups = state.groups;
        rows = [];
        $('piece-list').replaceChildren();
        state.groups.forEach((members, i) => {
          const row = document.createElement('div');
          row.className = 'piece-row';
          const button = document.createElement('button');
          button.className = 'piece-select';
          button.textContent = `Piece ${i + 1}`;
          const small = document.createElement('small');
          small.textContent = `${members.length} nodes · ${members.filter((n) => store.nodes.has(n)).length} clouds`;
          button.append(small);
          button.onclick = () => actions.select(members);
          button.ondblclick = () => {
            actions.select(members);
            actions.focus();
          };
          const eye = document.createElement('button');
          eye.className = 'visibility';
          eye.onclick = () => actions.visibility(members);
          row.append(button, eye);
          $('piece-list').append(row);
          rows.push({ members, button, eye });
        });
      }
      rows.forEach(({ members, button, eye }, i) => {
        button.setAttribute('aria-pressed', String(members === selected));
        button.disabled = eye.disabled =
          blocked || state.mode === 'fly' || !members.some((n) => store.nodes.has(n));
        const visible = members.some((n) => store.nodes.has(n) && state.visible(n));
        eye.innerHTML = eyeIcon(visible);
        eye.setAttribute('aria-label', `${visible ? 'Hide' : 'Show'} piece ${i + 1}`);
        eye.title = eye.getAttribute('aria-label');
      });
      $('selection-title').textContent = selected
        ? `Piece ${state.groups.indexOf(selected) + 1}`
        : 'No piece selected';
      $('selection-description').textContent = selected
        ? `${selected.length} nodes · ${selected.filter((i) => store.nodes.has(i)).length} clouds. Selected bounds are green.`
        : store.data
          ? 'Click a piece in the scene or choose one from the list.'
          : 'Open a scene folder to select and edit pieces.';
      $('selection-actions').hidden = !selected;
      $('isolate').checked = state.isolated;
      $('editing').hidden = state.mode !== 'edit';
      $('prepare').hidden = world;
      $('edit-tools').hidden = !world || !selected;
      $('edit-prompt').hidden = !!selected || !world;
      $('move').setAttribute('aria-pressed', String(state.tool === 'translate'));
      $('rotate').setAttribute('aria-pressed', String(state.tool === 'rotate'));
      $('tool-help').textContent =
        state.tool === 'translate'
          ? 'Drag arrows to move. Green is up; the ground square moves sideways. Escape cancels a drag.'
          : 'Drag the green ring to turn around the piece’s centre. Escape cancels a drag.';
      $('placement').textContent = world
        ? 'World placement · Metres'
        : store.data
          ? 'Raw DA3 coordinates · Use Edit to prepare GPS placement.'
          : '';
      $('undo').disabled = blocked || !store.undo.length || state.mode === 'fly';
      $('redo').disabled = blocked || !store.redo.length || state.mode === 'fly';
      $('save').disabled = blocked || !world || state.mode === 'fly';
      $('edit-state').textContent = dragging
        ? 'Adjusting piece…'
        : store.dirty
          ? 'Undownloaded changes'
          : store.exportRequested
            ? 'Download requested — replace your JSON'
            : store.data
              ? 'No placement changes'
              : 'No scene open';
      if (!busy)
        $('status').textContent = store.group
          ? `${points.toLocaleString()} points · ${store.data ? 'Scene' : 'Single PLY'}`
          : 'Ready';
      $('help').textContent = dragging
        ? 'Release to apply · Escape to cancel'
        : state.mode === 'fly'
          ? 'Mouse to steer · WASD/QE to fly · Escape returns to Inspect'
          : state.mode === 'edit'
            ? 'Select a piece · Drag handles to edit · Drag empty space to orbit'
            : editable
              ? 'Click to select · Drag to orbit · Double-click to focus · F to recenter'
              : 'Drag to orbit · Right-drag to pan · Scroll to zoom · Double-click to focus';
    },
  };
}
