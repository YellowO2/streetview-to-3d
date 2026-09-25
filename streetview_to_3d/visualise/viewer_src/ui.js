import { createSceneManager } from '@viewer/scene-manager';
const $ = (id) => document.getElementById(id);

// One viewer shell for every host. Editing adds only the Scene Manager.
export function createUI(actions, { editable = true } = {}) {
  $('scene-manager').hidden = !editable;
  const manager = editable ? createSceneManager(actions) : null;
  const bind = (id, fn) => ($(id).onclick = fn);
  for (const mode of ['inspect', 'fly']) bind(mode, () => actions.mode(mode));
  bind('exit-fly', () => actions.mode('inspect'));
  bind('recenter', actions.recenter);
  bind('open-folder', () => $('folder').click());
  $('folder').onchange = () => {
    const files = [...$('folder').files];
    $('folder').value = '';
    actions.files(files);
  };
  bind('toggle-settings', () => {
    $('view-panel').hidden = !$('view-panel').hidden;
    $('toggle-settings').setAttribute('aria-pressed', String(!$('view-panel').hidden));
  });
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
      $('status').hidden = false;
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
      const blocked = busy || dragging;
      document.body.dataset.mode = state.mode;
      $('scene-name').textContent = store.name;
      $('empty').hidden = !!store.group;
      $('flight').hidden = $('reticle').hidden = state.mode !== 'fly';
      for (const mode of ['inspect', 'fly']) {
        $(mode).setAttribute(
          'aria-pressed',
          String(mode === 'inspect' ? state.mode !== 'fly' : state.mode === 'fly'),
        );
        $(mode).disabled = blocked || (mode === 'fly' && !store.group);
      }
      $('recenter').disabled = blocked || !store.group;
      for (const id of ['open-folder', 'toggle-settings']) $(id).disabled = blocked;
      for (const input of document.querySelectorAll('#view-panel input')) input.disabled = blocked;
      manager?.render(store, state, { busy, dragging });
      $('scene-info').hidden = !store.group;
      $('scene-info').textContent = store.group
        ? `${points.toLocaleString()} ${store.splat ? 'splats' : 'points'}`
        : '';
      if (!busy) {
        $('status').textContent = '';
        $('status').hidden = true;
      }
    },
  };
}
