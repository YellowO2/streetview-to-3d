import { panoramaStart, START_HEIGHT } from '@viewer/core/start-view';
import * as THREE from 'three';
import { ViewerState } from '@viewer/core/state';
import { SceneStore, loadAsset, dispose } from '@viewer/core/scene-store';
import { scenePieces, LAND, BLOCKS } from '@viewer/core/scene-format';
import { fileEntries, resolveEntries, droppedFiles } from '@viewer/core/files';
import { createViewport } from '@viewer/core/viewport';
import { createNavigation } from '@viewer/flight/navigation';
import { createEditor } from '@viewer/core/editor';
import { createUI } from '@viewer/ui/shell';
import { exportPly } from '@viewer/core/export';

const SPLAT_RADIUS = 20; // view radius for a splat, which has no bounds
const PLACED_POINT_M = 0.1; // DA3 point size in a placed (metric) scene
// distance haze: the far land and buildings fade into the sky colour
const HAZE = 0xc9dbe6,
  HAZE_M = [250, 900]; // clear to, gone by (inside the 1 km land edge, terrain.RADIUS_M)

const state = new ViewerState(),
  store = new SceneStore();
const config = JSON.parse(document.getElementById('viewer-config').textContent);
const editable = config.editable !== false;
const ui = createUI(
  {
    mode: setMode,
    style: (name, options) => view.styles.set(name, options),
    demo: (name) => view.styles.demo(name),
    recenter,
    focus,
    select,
    showAll: () => {
      state.hidden.clear();
      refresh();
    },
    visibility: (members) => {
      state.toggleVisibility(members);
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
    buildings: (on) => {
      showBuildings = on;
      setBuildings();
    },
    export: (parts) => {
      if (!store.group || store.splat || busy) return;
      if (!parts.length) return ui.notify('Choose something to export.');
      const name = store.name.replace(/[^\w.-]+/g, '_') || 'scene';
      download(exportPly(store.group, parts, store.data?.center), `${name}.ply`);
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
  points = 0,
  showBuildings = true;
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
  // Shoot is Fly with the gun
  const gun = mode === 'shoot';
  if (gun) mode = 'fly';
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
  state.gun = gun;
  if (mode === 'fly') {
    refresh();
    navigation.start(gun);
  } else {
    navigation.stop();
    refresh();
  }
}
function configure() {
  const box = store.box();
  radius = box.isEmpty()
    ? SPLAT_RADIUS
    : Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 0.001);
  navigation.configure(radius);
  // near plane as far out as possible: at 1 cm the kilometre-away terrain flickered
  camera.near = Math.min(0.2, Math.max(radius * 0.002, 0.00001));
  camera.far = radius * 1000;
  camera.updateProjectionMatrix();
  // fully hazed just inside the land's edge, however far it reaches
  let land = null;
  store.group?.traverse((o) => {
    if (o.userData.surroundings === LAND) land = o;
  });
  if (land && !land.geometry.boundingSphere) land.geometry.computeBoundingSphere();
  const far = Math.min(HAZE_M[1], 0.9 * (land?.geometry.boundingSphere.radius ?? Infinity));
  scene.fog =
    store.placement === 'world' ? new THREE.Fog(HAZE, (HAZE_M[0] * far) / HAZE_M[1], far) : null;
  view.styles.configure(store, radius);
  setPointSize();
  setBuildings();
}
// show or hide the map's buildings (buildings.ply near DA3, blocks.ply far); DA3's own stay
function setBuildings() {
  store.group?.traverse((o) => {
    if (o.userData.surroundings === 'buildings' || o.userData.surroundings === BLOCKS)
      o.visible = showBuildings;
  });
}
function setPointSize() {
  // DA3 points share one size (fixed in metres when placed); map points use their spacing,
  // never smaller than DA3's so they match where they meet
  const own = store.placement === 'world' ? PLACED_POINT_M : radius * 0.002;
  store.group?.traverse((o) => {
    if (o.userData.pointStyle)
      o.material.uniforms.pointM.value = own * pointMultiplier; // dabs near DA3
    else if (o.isPoints)
      o.material.size = Math.max(o.userData.pointSize ?? 0, own) * pointMultiplier;
  });
}
function frameAll() {
  const box = store.box();
  if (box.isEmpty()) navigation.place(new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -1));
  else navigation.frame(box.getBoundingSphere(new THREE.Sphere()));
}
function recenter() {
  if (!store.group) return;
  if (state.mode === 'fly') setMode('inspect');
  editor.cancel();
  frameAll();
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
async function load({ source, resolve, name, splat = false }) {
  if (store.dirty && !confirm('Open another scene and discard unsaved changes?')) return;
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
      { splat },
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
    state.regroup(
      store.data
        ? scenePieces(store.data)
            .map((members) => members.filter((n) => store.nodes.has(n)))
            .filter((members) => members.length)
        : [],
    );
    points = store.splat?.numSplats || 0;
    store.group.traverse((o) => {
      if (o.isPoints) points += o.geometry.getAttribute('position').count;
    });
    configure();
    const start = panoramaStart(store.data);
    if (start) navigation.place(start.position, start.target);
    else if (!store.splat) {
      const center = store.box().getCenter(new THREE.Vector3());
      center.y += START_HEIGHT;
      navigation.place(center, center.clone().add(new THREE.Vector3(0, 0, -1)));
    } else frameAll();
  } catch (e) {
    if (token === version) ui.notify(`Could not open scene: ${e.message}`);
  } finally {
    if (token === version) {
      busy = false;
      refresh();
    }
  }
}
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
async function save() {
  if (!editable || store.placement !== 'world' || editor.dragging || busy) return;
  const blob = new Blob([JSON.stringify(store.exported(), null, 2) + '\n'], {
    type: 'application/json',
  });
  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: 'scene.json',
        types: [{ description: 'Saved scene', accept: { 'application/json': ['.json'] } }],
      });
      const writer = await handle.createWritable();
      await writer.write(blob);
      await writer.close();
      store.markExportRequested();
      refresh();
      ui.notify('Scene saved. Keep it with the original scene files.');
    } catch (error) {
      if (error.name !== 'AbortError') ui.notify(`Could not save scene: ${error.message}`);
    }
    return;
  }
  download(blob, 'scene.json');
  store.markExportRequested();
  refresh();
  ui.notify(
    'Scene file exported. Keep it in the original scene folder to reopen your saved placement.',
  );
}
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
  view.styles.render(dt, state.mode === 'edit' || editor.dragging);
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
ui.styles(config.style || 'paint');
refresh();
if (config.sceneUrl) {
  const base = config.sceneUrl.slice(0, config.sceneUrl.lastIndexOf('/') + 1);
  load({
    source: config.sceneUrl,
    resolve: (path) => base + path.split('/').map(encodeURIComponent).join('/'),
    name: 'Scene',
  });
} else if (config.splatUrl) load({ source: config.splatUrl, name: 'Splat', splat: true });
else if (config.plyUrl) load({ source: config.plyUrl, name: 'Scene point cloud' });
