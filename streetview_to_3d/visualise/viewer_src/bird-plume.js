import * as THREE from 'three';
import { birdPaint, birdDabs, DAB } from '@viewer/bird-paint';

// The bird's trail: while it flies, each wingtip lays overlapping world-fixed dabs that fade
// (bird-paint.js), from a fixed pool reused oldest first.
const STEP = 0.15, // dab spacing along the path, of a dab
  MOVING = 0.4, // m/s: slowest speed that lays a trail
  POOL = 4096;

export function createBirdPlume(bird, brushes) {
  const { material, uniforms } = birdPaint({ trail: true });
  const position = new Float32Array(POOL * 3),
    born = new Float32Array(POOL).fill(-1);
  const g = new THREE.BufferGeometry();
  for (const [name, array, size] of [
    ['position', position, 3],
    ['born', born, 1],
  ])
    g.setAttribute(name, new THREE.BufferAttribute(array, size).setUsage(THREE.DynamicDrawUsage));
  const trail = birdDabs(g, material, POOL);
  trail.name = 'Bird trail';
  // positions are in world space: ignore the bird's transform
  trail.matrixAutoUpdate = false;
  trail.matrixWorldAutoUpdate = false;
  bird.add(trail);

  const last = brushes.map(() => new THREE.Vector3()),
    here = new THREE.Vector3(),
    was = new THREE.Vector3();
  let time = 0,
    next = 0,
    fresh = true;
  const step = DAB * STEP;
  return {
    trail,
    update(dt) {
      time += dt;
      uniforms.time.value = time;
      bird.updateWorldMatrix(true, true);
      const flying = !fresh && dt > 0 && was.distanceTo(bird.position) / dt >= MOVING;
      was.copy(bird.position);
      let laid = false;
      brushes.forEach((brush, b) => {
        here.copy(brush.userData.tip).applyMatrix4(brush.matrixWorld);
        if (!flying) {
          last[b].copy(here);
          return;
        }
        const moved = here.distanceTo(last[b]),
          n = Math.floor(moved / step);
        for (let k = 1; k <= n; k++) {
          const s = (k * step) / moved;
          last[b].toArray(position, next * 3);
          position[next * 3] += (here.x - last[b].x) * s;
          position[next * 3 + 1] += (here.y - last[b].y) * s;
          position[next * 3 + 2] += (here.z - last[b].z) * s;
          born[next] = time - dt * (1 - s);
          next = (next + 1) % POOL;
        }
        if (n) {
          last[b].lerp(here, (n * step) / moved);
          laid = true;
        }
      });
      fresh = false;
      if (laid) {
        g.attributes.position.needsUpdate = true;
        g.attributes.born.needsUpdate = true;
      }
    },
    reset() {
      born.fill(-1);
      g.attributes.born.needsUpdate = true;
      time = 0;
      next = 0;
      fresh = true;
    },
    dispose() {
      g.dispose();
      material.dispose();
    },
  };
}
