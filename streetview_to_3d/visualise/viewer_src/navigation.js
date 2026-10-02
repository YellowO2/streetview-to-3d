import { FLIGHT, advanceFlight, steerBird } from '@viewer/flight-motion';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { createBird } from '@viewer/bird';
import { createGun } from '@viewer/gun';
import { clearShots } from '@viewer/effects/shot';

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
  const { bird, animate, resetPlume } = createBird();
  scene.add(bird);
  const gun = createGun(scene); // Shoot: flying as the bird does, but from the eye, shooting
  const steer = { yaw: 0, pitch: 0, roll: 0 }; // the bird's way (steerBird)
  const keys = new Set(),
    forward = new THREE.Vector3(),
    right = new THREE.Vector3(),
    move = new THREE.Vector3(),
    offset = new THREE.Vector3(),
    velocity = new THREE.Vector3(),
    smoothHeading = new THREE.Quaternion(),
    cameraTarget = new THREE.Vector3();
  let flying = false,
    shooting = false,
    radius = 5,
    speed = 1,
    chase = 1;
  const captured = () => document.pointerLockElement === canvas;
  const chaseOffset = () =>
    shooting
      ? offset.set(0, 0, 0)
      : offset.set(0, FLIGHT.height, FLIGHT.distance * chase).applyQuaternion(smoothHeading);
  const stop = () => {
    keys.clear();
    velocity.set(0, 0, 0);
    if (!flying) return;
    flying = false;
    look.unlock();
    bird.visible = false;
    resetPlume();
    camera.getWorldDirection(forward);
    orbit.target.copy(camera.position).addScaledVector(forward, FLIGHT.distance * chase);
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
  // a shot on letting go, the bigger the longer held (gun.js)
  document.addEventListener('mousedown', (e) => {
    if (!flying || !shooting || !captured() || e.button !== 0) return;
    gun.press();
  });
  document.addEventListener('mouseup', (e) => {
    if (!flying || !shooting || e.button !== 0) return;
    gun.release(camera.position, forward.set(0, 0, -1).applyQuaternion(camera.quaternion));
  });
  addEventListener('keyup', (e) => keys.delete(e.code));
  addEventListener('keydown', (e) => {
    if (!flying || !captured() || e.target.matches('input,textarea,select')) return;
    if (e.code === 'KeyH') {
      // the keys' bar out of the way (the mouse is captured: no button to click), H again to bring it back
      document.getElementById('flight')?.classList.toggle('dismissed');
      return;
    }
    if (shooting && e.code === 'KeyR') {
      // the world whole again
      clearShots();
      gun.reset();
      return;
    }
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
    get shooting() {
      return flying && shooting;
    },
    configure(r) {
      radius = Math.max(r, 0.001);
      bird.scale.setScalar(FLIGHT.birdScale);
      orbit.minDistance = radius * 0.001;
      orbit.maxDistance = radius * 100;
    },
    settings(s, c) {
      speed = s;
      chase = c;
    },
    // fly as the bird, or (gun) shoot from the eye
    start(withGun = false) {
      if (flying) {
        if (shooting !== withGun) {
          // straight from one to the other: the bird left where the eye is
          shooting = withGun;
          bird.position.copy(camera.position).sub(chaseOffset());
          resetPlume();
          bird.visible = !shooting;
        }
        look.lock();
        return;
      }
      shooting = withGun;
      orbit.enableDamping = false;
      orbit.update();
      orbit.enableDamping = true;
      orbit.enabled = false;
      const angles = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ');
      angles.x = THREE.MathUtils.clamp(angles.x, -Math.PI / 2 + 0.12, Math.PI / 2 - 0.12);
      angles.z = 0;
      heading.quaternion.setFromEuler(angles);
      smoothHeading.copy(heading.quaternion);
      velocity.set(0, 0, 0);
      bird.position.copy(camera.position).sub(chaseOffset());
      bird.quaternion.copy(heading.quaternion);
      Object.assign(steer, { yaw: angles.y, pitch: 0, roll: 0 });
      resetPlume();
      bird.visible = !shooting;
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
      gun.update(Math.min(Math.max(dt, 0), 0.05));
      if (!flying) {
        gun.hold(camera, false);
        if (orbit.enabled) orbit.update();
        return;
      }
      dt = Math.min(Math.max(dt, 0), 0.05);
      smoothHeading.slerp(heading.quaternion, 1 - Math.exp(-10 * dt));
      move.set(0, 0, 0);
      forward.set(0, 0, -1).applyQuaternion(smoothHeading);
      right.set(1, 0, 0).applyQuaternion(smoothHeading);
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
      move.normalize().multiplyScalar(FLIGHT.speed * speed * (boost ? FLIGHT.boost : 1));
      advanceFlight(bird.position, velocity, move, dt);
      if (shooting)
        camera.position.copy(bird.position); // the eye, no bird
      else {
        steerBird(bird.quaternion, velocity, smoothHeading, steer, dt);
        animate(dt, moving);
        cameraTarget.copy(bird.position).add(chaseOffset());
        camera.position.lerp(cameraTarget, 1 - Math.exp(-12 * dt));
      }
      camera.quaternion.copy(smoothHeading);
      gun.hold(camera, shooting); // in the eye's hand, as it now is
    },
  };
}
