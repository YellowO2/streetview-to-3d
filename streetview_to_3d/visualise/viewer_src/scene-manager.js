const $ = (id) => document.getElementById(id);
const eyeIcon = (visible) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>${visible ? '' : '<path d="m3 3 18 18"/>'}</svg>`;

// Optional editor component. Owns hierarchy, selection controls and save UI;
// the shared viewer shell never needs to know their layout or state.
export function createSceneManager(actions) {
  const bind = (id, fn) => ($(id).onclick = fn);
  bind('show-all', actions.showAll);
  bind('undo', () => actions.history(false));
  bind('redo', () => actions.history(true));
  bind('reset', actions.reset);
  bind('prepare-gps', () => actions.gps(Number($('da3-scale').value)));
  const fields = ['east', 'north', 'up', 'turn'];
  function applyFields() {
    const values = fields.map((id) => ($(id).value.trim() === '' ? NaN : Number($(id).value)));
    if (values.every((v) => v === 0)) return;
    if (actions.adjust(...values)) fields.forEach((id) => ($(id).value = '0'));
  }
  fields.forEach((id) => ($(id).onchange = applyFields));
  bind('save', actions.save);
  let rows = [],
    lastGroups = null,
    lastData = null,
    selectionKey = '';

  function row(members, label, kind, store) {
    const element = document.createElement('div');
    element.className = `scene-row ${kind}-row`;
    const button = document.createElement('button');
    button.className = `${kind}-select`;
    button.textContent = label;
    button.title =
      kind === 'node'
        ? store.data.nodes?.[members[0]]?.ply || 'No point cloud'
        : 'Select the whole piece';
    button.onclick = () => actions.select(members, kind);
    button.ondblclick = () => {
      actions.select(members, kind);
      actions.focus();
    };
    const eye = document.createElement('button');
    eye.className = 'visibility quiet';
    eye.onclick = () => actions.visibility(members);
    element.append(button, eye);
    rows.push({ members, button, eye, kind, label });
    return element;
  }
  function rebuild(store, state) {
    rows = [];
    $('piece-list').replaceChildren();
    state.groups.forEach((members, i) => {
      const branch = document.createElement('div');
      branch.className = 'piece-branch';
      const heading = row(members, `Piece ${i + 1}`, 'piece', store);
      const children = document.createElement('div');
      children.className = 'node-list';
      children.id = `piece-nodes-${i}`;
      children.hidden = !state.selected?.some((n) => members.includes(n));
      const expand = document.createElement('button');
      expand.className = 'disclosure quiet';
      expand.textContent = '›';
      expand.setAttribute('aria-label', `Expand piece ${i + 1}`);
      expand.setAttribute('aria-controls', children.id);
      expand.setAttribute('aria-expanded', String(!children.hidden));
      expand.onclick = () => {
        children.hidden = !children.hidden;
        expand.setAttribute('aria-expanded', String(!children.hidden));
      };
      heading.prepend(expand);
      const count = document.createElement('span');
      count.className = 'node-count';
      count.textContent = String(members.length);
      heading.querySelector('.piece-select').append(count);
      members.forEach((n) => children.append(row([n], `Node ${n}`, 'node', store)));
      branch.append(heading, children);
      $('piece-list').append(branch);
    });
  }
  return {
    render(store, state, { busy, dragging }) {
      const blocked = busy || dragging || state.mode === 'fly';
      const selected = state.selected,
        world = store.placement === 'world';
      for (const control of document.querySelectorAll(
        '#scene-manager button, #scene-manager input',
      ))
        control.disabled = blocked;
      $('show-all').disabled = blocked || !store.data;
      $('piece-count').textContent = store.data ? `(${state.groups.length})` : '';
      $('piece-guide').hidden = !!store.data;
      $('piece-guide').textContent = store.group
        ? 'Open a scene folder to work with pieces.'
        : 'Open a scene to see its pieces.';
      if (lastGroups !== state.groups || lastData !== store.data) {
        lastGroups = state.groups;
        lastData = store.data;
        rebuild(store, state);
      }
      rows.forEach(({ members, button, eye, kind, label }) => {
        const active =
          kind === state.selectionKind &&
          (kind === 'piece' ? members === selected : selected?.[0] === members[0]);
        button.setAttribute('aria-pressed', String(!!active));
        button.disabled = eye.disabled = blocked || !members.some((n) => store.nodes.has(n));
        const visible = members.some((n) => store.nodes.has(n) && state.visible(n));
        eye.innerHTML = eyeIcon(visible);
        eye.setAttribute('aria-label', `${visible ? 'Hide' : 'Show'} ${label.toLowerCase()}`);
        eye.title = eye.getAttribute('aria-label');
      });
      const key = `${state.selectionKind}:${selected?.join(',') || ''}`;
      if (key !== selectionKey) {
        selectionKey = key;
        fields.forEach((id) => ($(id).value = '0'));
      }
      $('selection-title').textContent = !selected
        ? 'Nothing selected'
        : state.selectionKind === 'node'
          ? `Node ${selected[0]}`
          : `Piece ${state.groups.indexOf(selected) + 1}`;
      $('selection-description').textContent = !selected
        ? 'Select a piece or node.'
        : state.selectionKind === 'node'
          ? store.data.nodes?.[selected[0]]?.ply || 'Panorama node'
          : `${selected.length} nodes · Move together`;
      $('editing').hidden = state.mode !== 'edit';
      $('prepare').hidden = world;
      $('edit-tools').hidden = !world || !selected;
      $('edit-prompt').hidden = !!selected || !world;
      $('placement').textContent = world
        ? 'World coordinates · Metres'
        : store.data
          ? 'Raw coordinates'
          : '';
      $('undo').disabled = blocked || !store.undo.length;
      $('redo').disabled = blocked || !store.redo.length;
      $('save').disabled = blocked || !world;
      $('edit-state').textContent = dragging
        ? 'Adjusting…'
        : store.dirty
          ? 'Unsaved changes'
          : store.exportRequested
            ? 'Scene file exported'
            : store.data
              ? 'No placement changes'
              : 'No scene open';
    },
  };
}
