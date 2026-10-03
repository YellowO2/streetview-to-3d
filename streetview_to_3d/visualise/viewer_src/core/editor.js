import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// Translate gizmo for the selected pieces. A drag is one transaction: preview, then commit or cancel.
// Only this controller disables orbit during a manipulation.
export function createEditor(scene, camera, canvas, navigation, store, state, changed, failed) {
  const pivot = new THREE.Object3D();
  scene.add(pivot);
  const control = new TransformControls(camera, canvas);
  control.setSize(0.8);
  control.setSpace('world');
  const helper = control.getHelper();
  helper.userData.styleOverlay = true;
  scene.add(helper);
  control.enabled = false;
  let drag = null;
  const sync = () => {
    if (drag) return;
    const active =
      state.mode === 'edit' &&
      store.placement === 'world' &&
      state.selected?.some((i) => store.nodes.get(i)?.visible);
    control.enabled = !!active;
    if (!active) {
      control.detach();
      return;
    }
    store.box(state.selected, true).getCenter(pivot.position);
    pivot.quaternion.identity();
    pivot.updateMatrixWorld(true);
    control.setMode('translate');
    control.showX = control.showY = control.showZ = true;
    if (control.object !== pivot) control.attach(pivot);
  };
  const cancel = () => {
    if (!drag) return;
    const before = drag.before;
    drag = null;
    store.restore(before);
    control.dragging = false;
    control.axis = null;
    navigation.orbit.enabled = state.mode !== 'fly';
    changed();
  };
  canvas.addEventListener(
    'pointerdown',
    (e) => {
      if (!control.enabled || e.button !== 0) return;
      const rect = canvas.getBoundingClientRect();
      control.pointerHover({
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: 1 - ((e.clientY - rect.top) / rect.height) * 2,
        button: 0,
      });
      if (control.axis) {
        navigation.orbit.enableDamping = false;
        navigation.orbit.update();
        navigation.orbit.enableDamping = true;
        navigation.orbit.enabled = false;
      }
    },
    true,
  );
  control.addEventListener('mouseDown', () => {
    pivot.updateMatrixWorld(true);
    drag = {
      before: store.snapshot(),
      inverse: pivot.matrixWorld.clone().invert(),
      members: [...state.selected],
    };
    changed();
  });
  control.addEventListener('objectChange', () => {
    if (!drag) return;
    try {
      pivot.updateMatrixWorld(true);
      store.preview(pivot.matrixWorld.clone().multiply(drag.inverse), drag.before, drag.members);
      changed();
    } catch (e) {
      cancel();
      failed(e.message);
    }
  });
  control.addEventListener('mouseUp', () => {
    if (drag) {
      store.commit(drag.before);
      drag = null;
      changed();
    }
  });
  control.addEventListener(
    'dragging-changed',
    (e) => (navigation.orbit.enabled = !e.value && state.mode !== 'fly'),
  );
  canvas.addEventListener('pointerup', () => {
    if (!drag) navigation.orbit.enabled = state.mode !== 'fly';
  });
  canvas.addEventListener('pointercancel', cancel);
  addEventListener('blur', cancel);
  return {
    sync,
    cancel,
    get dragging() {
      return !!drag;
    },
    adjust(east, north, height, degrees) {
      if (!state.selected || store.placement !== 'world') return;
      if (![east, north, height, degrees].every(Number.isFinite))
        throw Error('Enter a number in every adjustment field.');
      const centre = store.box(state.selected).getCenter(new THREE.Vector3());
      const delta = new THREE.Matrix4()
        .makeTranslation(east, height, -north)
        .multiply(new THREE.Matrix4().makeTranslation(...centre.toArray()))
        .multiply(new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(degrees)))
        .multiply(new THREE.Matrix4().makeTranslation(...centre.clone().negate().toArray()));
      const before = store.snapshot();
      store.preview(delta, before, state.selected);
      store.commit(before);
      changed();
    },
  };
}
