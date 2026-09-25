import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { createBird } from '@viewer/bird';

// Owns camera input, never selection or piece transforms. Escape/unlock returns
// to Inspect through the supplied callback, without clearing scene state.
export function createNavigation(scene, camera, canvas, onRelease, onError) {
  const orbit = new OrbitControls(camera, canvas);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.zoomToCursor = true;
  const heading = new THREE.PerspectiveCamera(),
    look = new PointerLockControls(heading, canvas);
  look.minPolarAngle = 0.12;
  look.maxPolarAngle = Math.PI - 0.12;
  look.pointerSpeed = 0.65;
  const { bird, wings } = createBird();
  scene.add(bird);
  const keys = new Set(),
    forward = new THREE.Vector3(),
    right = new THREE.Vector3(),
    move = new THREE.Vector3(),
    offset = new THREE.Vector3();
  let flying = false,
    radius = 5,
    wingTime = 0,
    speed = 1,
    chase = 1;
  const captured = () => document.pointerLockElement === canvas;
  const chaseOffset = () =>
    offset
      .set(0, radius * 0.025 * 1.1, radius * 0.025 * 4.5 * chase)
      .applyQuaternion(heading.quaternion);
  const stop = () => {
    keys.clear();
    if (!flying) return;
    flying = false;
    look.unlock();
    bird.visible = false;
    camera.getWorldDirection(forward);
    orbit.target.copy(camera.position).addScaledVector(forward, radius * 0.1125);
    orbit.enabled = true;
    orbit.update();
  };
  look.addEventListener('unlock', () => {
    keys.clear();
    if (flying) {
      stop();
      onRelease();
    }
  });
  look.addEventListener('lock', () => {
    if (!flying) look.unlock();
    else canvas.focus();
  });
  document.addEventListener('pointerlockerror', () => {
    stop();
    onRelease();
    onError('Mouse capture was blocked. Fly works in a regular browser that allows pointer lock.');
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('keydown', (e) => {
    if (!flying || !captured() || e.target.matches('input,textarea,select')) return;
    if (
      ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'Space', 'ShiftLeft', 'ShiftRight'].includes(
        e.code,
      )
    ) {
      e.preventDefault();
      keys.add(e.code);
    }
  });
  const pause = () => {
    if (flying) {
      stop();
      onRelease();
    }
    keys.clear();
  };
  addEventListener('blur', pause);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause();
  });
  return {
    orbit,
    stop,
    get flying() {
      return flying;
    },
    configure(r) {
      radius = Math.max(r, 0.001);
      bird.scale.setScalar(radius * 0.025);
      orbit.minDistance = radius * 0.001;
      orbit.maxDistance = radius * 100;
    },
    settings(s, c) {
      speed = s;
      chase = c;
    },
    start() {
      if (flying) {
        look.lock();
        return;
      }
      orbit.enableDamping = false;
      orbit.update();
      orbit.enableDamping = true;
      orbit.enabled = false;
      const angles = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
      angles.x = THREE.MathUtils.clamp(angles.x, -Math.PI / 2 + 0.12, Math.PI / 2 - 0.12);
      angles.z = 0;
      heading.quaternion.setFromEuler(angles);
      bird.position.copy(camera.position).sub(chaseOffset());
      bird.quaternion.copy(heading.quaternion);
      bird.visible = true;
      flying = true;
      look.lock();
    },
    // Stand at `position` looking at `target` -- inside a splat, which is
    // seen from where its panorama was taken rather than from outside.
    place(position, target) {
      stop();
      orbit.enableDamping = false;
      orbit.update();
      orbit.target.copy(target);
      camera.position.copy(position);
      orbit.update();
      orbit.enableDamping = true;
    },
    frame(sphere) {
      stop();
      const v = THREE.MathUtils.degToRad(camera.fov / 2),
        h = Math.atan(Math.tan(v) * camera.aspect);
      const distance = (Math.max(sphere.radius, radius * 0.005) / Math.sin(Math.min(v, h))) * 1.15;
      orbit.enableDamping = false;
      orbit.update();
      orbit.target.copy(sphere.center);
      camera.position
        .copy(sphere.center)
        .add(new THREE.Vector3(0, 0.22, 1).normalize().multiplyScalar(distance));
      orbit.update();
      orbit.enableDamping = true;
    },
    tick(dt) {
      if (!flying) {
        if (orbit.enabled) orbit.update();
        return;
      }
      move.set(0, 0, 0);
      heading.getWorldDirection(forward);
      right.set(1, 0, 0).applyQuaternion(heading.quaternion);
      if (captured()) {
        if (keys.has('KeyW')) move.add(forward);
        if (keys.has('KeyS')) move.sub(forward);
        if (keys.has('KeyD')) move.add(right);
        if (keys.has('KeyA')) move.sub(right);
        if (keys.has('KeyE') || keys.has('Space')) move.y++;
        if (keys.has('KeyQ')) move.y--;
      }
      const moving = move.lengthSq() > 0,
        boost = keys.has('ShiftLeft') || keys.has('ShiftRight');
      bird.position.addScaledVector(move.normalize(), radius * 0.35 * speed * (boost ? 3 : 1) * dt);
      bird.quaternion.slerp(heading.quaternion, 1 - Math.exp(-12 * dt));
      wingTime += dt * (moving ? 13 : 5);
      wings[0].rotation.z = Math.sin(wingTime) * 0.32;
      wings[1].rotation.z = -Math.sin(wingTime) * 0.32;
      camera.position.copy(bird.position).add(chaseOffset());
      camera.quaternion.copy(heading.quaternion);
    },
  };
}
