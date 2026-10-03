import { createStyleControls } from '@viewer/ui/style-controls';
import { createSceneManager } from '@viewer/ui/scene-manager';
const $ = (id) => document.getElementById(id);

// One viewer shell for every host. Editing adds only the Scene Manager.
export function createUI(actions, { editable = true } = {}) {
  $('scene-manager').hidden = !editable;
  $('expand-manager').hidden = true;
  $('collapse-manager').onclick = () => {
    $('scene-manager').hidden = true;
    $('expand-manager').hidden = false;
    $('expand-manager').focus();
  };
  $('expand-manager').onclick = () => {
    $('scene-manager').hidden = false;
    $('expand-manager').hidden = true;
    $('collapse-manager').focus();
  };
  const styles = createStyleControls(actions);
  const manager = editable ? createSceneManager(actions) : null;
  const bind = (id, fn) => ($(id).onclick = fn);
  for (const mode of ['inspect', 'fly', 'shoot']) bind(mode, () => actions.mode(mode));
  bind('exit-fly', () => actions.mode('inspect'));
  bind('recenter', actions.recenter);
  bind('open-folder', () => $('folder').click());
  $('folder').onchange = () => {
    const files = [...$('folder').files];
    $('folder').value = '';
    actions.files(files);
  };
  // Settings and Export share the corner: opening one closes the other
  const panels = { 'toggle-settings': 'view-panel', 'toggle-export': 'export-panel' };
  for (const [button, panel] of Object.entries(panels))
    bind(button, () => {
      const open = $(panel).hidden;
      for (const [b, p] of Object.entries(panels)) {
        $(p).hidden = !(open && p === panel);
        $(b).setAttribute('aria-pressed', String(!$(p).hidden));
      }
    });
  bind('export-ply', () =>
    actions.export(['street', 'ground', 'buildings'].filter((p) => $(`export-${p}`).checked)),
  );
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
  $('show-buildings').onchange = () => actions.buildings?.($('show-buildings').checked);
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
    styles: styles.select,
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
      $('flight-keys').textContent = state.gun
        ? 'WASD to move · Q/E down/up · click to shoot, hold for bigger · R to mend · H to hide'
        : 'WASD to fly · Q/E down/up · Shift boost · H to hide';
      $('exit-fly').textContent = `Exit ${state.gun ? 'Edit' : 'Fly'} · Esc`;
      const pressed = {
        inspect: state.mode !== 'fly',
        fly: state.mode === 'fly' && !state.gun,
        shoot: state.mode === 'fly' && state.gun,
      };
      for (const mode of ['inspect', 'fly', 'shoot']) {
        $(mode).setAttribute('aria-pressed', String(pressed[mode]));
        $(mode).disabled = blocked || (mode !== 'inspect' && !store.group);
      }
      $('recenter').disabled = blocked || !store.group;
      for (const id of ['open-folder', 'toggle-settings']) $(id).disabled = blocked;
      $('toggle-export').disabled = blocked || !store.group || !!store.splat;
      if ($('toggle-export').disabled) $('export-panel').hidden = true;
      for (const input of document.querySelectorAll('#view-panel input')) input.disabled = blocked;
      styles.render(store, blocked);
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
