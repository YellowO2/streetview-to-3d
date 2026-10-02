import * as THREE from 'three';
import { birdPaint, birdDabs } from '@viewer/bird-paint';
import { carve } from '@viewer/effects/shot';

// The gun (Shoot): each shot a ball of dabs -- the bird's paint, faint and
// white (bird-paint.js) -- flying straight on from the eye, shooting away
// every point it passes through (effects/shot.js), through all of them, till
// it is gone (LIFE_S) or off the grid the shots are kept on.
export const RADIUS = 1, // m: a shot this big, its hole as big
  SPEED = 30, // m/s
  LIFE_S = 3,
  MUZZLE = 1.6, // m: leaving this far ahead of the eye
  COOLDOWN_S = 0.15,
  SHELLS = [1, 0.66, 0.33], // its dabs on these shells (of RADIUS)
  SPACING = 0.12, // m apart on each
  POOL = 8;

export function createGun(scene) {
  const { material } = birdPaint();
  const positions = [];
  for (const shell of SHELLS) {
    const r = RADIUS * shell,
      count = Math.max(8, Math.round((4 * Math.PI * r * r) / SPACING ** 2));
    for (let i = 0; i < count; i++) {
      const y = 1 - (2 * (i + 0.5)) / count,
        ring = Math.sqrt(1 - y * y),
        theta = i * 2.39996323;
      positions.push(Math.cos(theta) * ring * r, y * r, Math.sin(theta) * ring * r);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const first = birdDabs(geometry, material, positions.length / 3);
  const shots = Array.from({ length: POOL }, (_, i) => {
    const ball = i ? new THREE.Points(geometry, material) : first;
    Object.assign(ball.userData, first.userData);
    ball.frustumCulled = false;
    ball.name = 'Shot';
    ball.visible = false;
    scene.add(ball);
    return { ball, way: new THREE.Vector3(), age: 0, from: new THREE.Vector3() };
  });
  let next = 0,
    wait = 0;
  return {
    shots,
    // a shot from the eye (origin), along way (a unit vector)
    fire(origin, way) {
      if (wait > 0) return false;
      wait = COOLDOWN_S;
      const shot = shots[next];
      next = (next + 1) % POOL;
      shot.way.copy(way);
      shot.ball.position.copy(origin).addScaledVector(way, MUZZLE);
      shot.from.copy(shot.ball.position);
      shot.age = 0;
      shot.ball.visible = true;
      return true;
    },
    update(dt) {
      wait -= dt;
      for (const shot of shots) {
        if (!shot.ball.visible) continue;
        shot.age += dt;
        shot.from.copy(shot.ball.position);
        shot.ball.position.addScaledVector(shot.way, SPEED * dt);
        if (!carve(shot.from, shot.ball.position, RADIUS) || shot.age >= LIFE_S)
          shot.ball.visible = false;
      }
    },
    reset() {
      for (const shot of shots) shot.ball.visible = false;
      wait = 0;
    },
    dispose() {
      shots.forEach(({ ball }) => ball.removeFromParent());
      geometry.dispose();
      material.dispose();
    },
  };
}
