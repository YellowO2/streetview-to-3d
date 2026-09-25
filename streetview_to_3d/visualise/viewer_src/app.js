import * as THREE from 'three';
import { ViewerState } from '@viewer/state';
import { SceneStore, loadAsset, dispose } from '@viewer/scene-store';
import { scenePieces } from '@viewer/scene-format';
import { fileEntries, resolveEntries, droppedFiles } from '@viewer/files';
import { createViewport } from '@viewer/viewport';
import { createNavigation } from '@viewer/navigation';
import { createEditor } from '@viewer/editor';
import { createUI } from '@viewer/ui';

const state = new ViewerState(),
  store = new SceneStore();
const config = JSON.parse(document.getElementById('viewer-config').textContent);
const editable = config.editable !== false;
const ui = createUI(
  {
    mode: setMode,
    recenter,
    focus,
    select,
    showAll: () => {
      state.hidden.clear();
      refresh();
    },
    group: (threshold) => {
      state.threshold = threshold;
      state.regroup(scenePieces(store.data, threshold));
      refresh();
    },
    visibility: (members) => {
      state.toggleVisibility(members);
      refresh();
    },
    tool: (tool) => {
      state.tool = tool;
      refresh();
    },
    history: (redo) => {
      editor.cancel();
      store.history(redo);
      refresh();
    },
    reset: () => {
      if (state.selected) {
        store.resetPiece(state.selected);
        refresh();
      }
    },
    gps: (scale) =>
      attempt(() => {
        store.prepareGPS(scale);
        configure();
        refresh();
        ui.notify(
          'GPS starting placement ready; road height is not aligned. Reset uses this starting placement.',
        );
      }),
    adjust: (...values) => attempt(() => editor.adjust(...values)),
    files: (files) => openEntries(fileEntries(files)),
    save,
    settings: (point, speed, chase) => {
      pointMultiplier = point;
      navigation.settings(speed, chase);
      setPointSize();
    },
  },
  { editable },
);
let view;
try {
  view = createViewport(document.getElementById('viewport'));
} catch (e) {
  clearTimeout(window.viewerBootTimer);
  ui.notify('WebGL could not start. Try a browser with hardware acceleration.');
  throw e;
}
const { scene, camera, renderer, canvas, highlight } = view;
const navigation = createNavigation(scene, camera, canvas, () => setMode('inspect'), ui.notify);
const editor = createEditor(scene, camera, canvas, navigation, store, state, refresh, ui.notify);
let busy = false,
  version = 0,
  radius = 5,
  pointMultiplier = 1,
  points = 0;
function attempt(fn) {
  try {
    fn();
    return true;
  } catch (e) {
    ui.notify(e.message);
    return false;
  }
}
function refresh() {
  store.nodes.forEach((object, i) => (object.visible = state.visible(i)));
  highlight.box.copy(store.box(state.selected || [], true));
  highlight.visible = state.mode !== 'fly' && !!state.selected && !highlight.box.isEmpty();
  editor.sync();
  ui.render(store, state, { busy, dragging: editor.dragging, points });
}
function setMode(mode) {
  if (
    busy ||
    !['inspect', 'fly', 'edit'].includes(mode) ||
    (mode === 'fly' && !store.group) ||
    (mode === 'edit' && (!editable || !store.data))
  )
    return;
  editor.cancel();
  // Update state before releasing pointer lock so the unlock callback cannot
  // overwrite an explicit switch to Edit.
  state.mode = mode;
  if (mode === 'fly') {
    refresh();
    navigation.start();
  } else {
    navigation.stop();
    refresh();
  }
}
function configure() {
  radius = Math.max(store.box().getBoundingSphere(new THREE.Sphere()).radius, 0.001);
  navigation.configure(radius);
  camera.near = Math.max(radius * 0.0001, 0.00001);
  camera.far = radius * 1000;
  camera.updateProjectionMatrix();
  setPointSize();
}
function setPointSize() {
  store.group?.traverse((o) => {
    if (o.isPoints) o.material.size = radius * 0.002 * pointMultiplier;
  });
}
function recenter() {
  if (!store.group) return;
  if (state.mode === 'fly') setMode('inspect');
  editor.cancel();
  navigation.frame(store.box().getBoundingSphere(new THREE.Sphere()));
  refresh();
}
function focus() {
  if (!state.selected) return;
  editor.cancel();
  navigation.frame(store.box(state.selected).getBoundingSphere(new THREE.Sphere()));
  refresh();
}
function select(members, kind = 'piece') {
  if (!editable || editor.dragging || busy) return;
  state.select(members, kind);
  if (members) setMode('edit');
  else refresh();
}
async function openEntries(entries) {
  if (entries.length) attempt(() => load(resolveEntries(entries)));
}
async function load({ source, resolve, name }) {
  if (
    store.dirty &&
    !confirm('Open another scene and discard changes that have not been downloaded?')
  )
    return;
  const token = ++version;
  editor.cancel();
  navigation.stop();
  state.mode = 'inspect';
  busy = true;
  refresh();
  ui.progress('Reading scene…');
  try {
    const asset = await loadAsset(
      source,
      resolve,
      (message) => {
        if (token === version) ui.progress(message);
      },
      () => token !== version,
    );
    if (!asset) return;
    if (token !== version) {
      dispose(asset.group);
      return;
    }
    if (store.group) scene.remove(store.group);
    store.install(asset, name);
    scene.add(store.group);
    state.reset();
    state.regroup(store.data ? scenePieces(store.data) : []);
    points = 0;
    store.group.traverse((o) => {
      if (o.isPoints) points += o.geometry.getAttribute('position').count;
    });
    configure();
    navigation.frame(store.box().getBoundingSphere(new THREE.Sphere()));
  } catch (e) {
    if (token === version) ui.notify(`Could not open scene: ${e.message}`);
  } finally {
    if (token === version) {
      busy = false;
      refresh();
    }
  }
}
function save() {
  if (!editable || store.placement !== 'world' || editor.dragging || busy) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(store.exported(), null, 2) + '\n'], { type: 'application/json' }),
  );
  const a = document.createElement('a');
  a.href = url;
  a.download = 'scene.json';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
  store.markExportRequested();
  refresh();
  ui.notify(
    'Download requested. Replace scene.json beside the original PLYs with the downloaded file. Your original files have not been overwritten.',
  );
}
let down = null;
canvas.addEventListener(
  'pointerdown',
  (e) => (down = e.button === 0 ? { x: e.clientX, y: e.clientY, dragged: false } : null),
);
canvas.addEventListener('pointermove', (e) => {
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) down.dragged = true;
});
canvas.addEventListener('pointercancel', () => (down = null));
canvas.addEventListener('pointerup', (e) => {
  const previous = down;
  down = null;
  if (
    editor.consumePick() ||
    editor.dragging ||
    !previous ||
    previous.dragged ||
    state.mode === 'fly' ||
    busy
  )
    return;
  const hit = view.pick(e, store, radius * 0.002 * pointMultiplier);
  select(hit ? state.groups.find((m) => m.includes(hit.object.userData.nodeIndex)) || null : null);
});
canvas.addEventListener('dblclick', (e) => {
  if (state.mode === 'fly' || editor.overHandle || editor.dragging || busy) return;
  const hit = view.pick(e, store, radius * 0.002 * pointMultiplier);
  if (hit) {
    const members = state.groups.find((m) => m.includes(hit.object.userData.nodeIndex));
    if (members && editable) {
      select(members);
      focus();
    } else {
      navigation.orbit.target.copy(hit.point);
      navigation.orbit.update();
    }
  }
});
addEventListener('keydown', (e) => {
  if (e.target.matches('input,textarea,select') || busy) return;
  if (e.code === 'Escape') {
    if (editor.dragging) editor.cancel();
    else if (state.mode === 'fly') setMode('inspect');
    else if (state.selected) select(null);
    return;
  }
  if (editor.dragging || state.mode === 'fly') return;
  if (editable && (e.ctrlKey || e.metaKey) && ['KeyZ', 'KeyY'].includes(e.code)) {
    e.preventDefault();
    store.history(e.shiftKey || e.code === 'KeyY');
    refresh();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.code === 'KeyF') {
    e.preventDefault();
    recenter();
  }
  if (['Equal', 'Minus', 'NumpadAdd', 'NumpadSubtract'].includes(e.code)) {
    e.preventDefault();
    ui.pointStep(['Equal', 'NumpadAdd'].includes(e.code) ? 1 : -1);
  }
});
let dragDepth = 0;
addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragDepth++;
  ui.drop(true);
});
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    ui.drop(false);
  }
});
addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  ui.drop(false);
  try {
    const entries = await droppedFiles(e.dataTransfer);
    openEntries(entries);
  } catch (error) {
    ui.notify(error.message);
  }
});
addEventListener('beforeunload', (e) => {
  if (store.dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});
let last = performance.now();
const tick = (now) => {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  navigation.tick(dt);
  renderer.render(scene, camera);
};
renderer.setAnimationLoop(tick);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    editor.cancel();
    renderer.setAnimationLoop(null);
  } else {
    last = performance.now();
    renderer.setAnimationLoop(tick);
  }
});
clearTimeout(window.viewerBootTimer);
ui.settings();
refresh();
if (config.sceneUrl) {
  const base = config.sceneUrl.slice(0, config.sceneUrl.lastIndexOf('/') + 1);
  load({
    source: config.sceneUrl,
    resolve: (path) => base + path.split('/').map(encodeURIComponent).join('/'),
    name: 'Scene',
  });
} else if (config.plyUrl) load({ source: config.plyUrl, name: 'Scene point cloud' });
