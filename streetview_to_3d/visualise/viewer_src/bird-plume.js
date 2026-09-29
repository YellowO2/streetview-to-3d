import * as THREE from 'three';

// A fixed particle pool: each point forms near a feather, releases into world
// space, and fades along the wake. No accumulating particles or random jitter.
export function createBirdPlume(bird, clouds) {
  const particles = [];
  const transforms = clouds.map(({ points, rest }, cloudIndex) => {
    const alpha = new THREE.Float32BufferAttribute(new Float32Array(rest.length / 3), 1);
    alpha.setUsage(THREE.DynamicDrawUsage);
    points.geometry.setAttribute('birdAlpha', alpha);
    const inverse = new THREE.Matrix4();
    for (let i = 0; i < alpha.count; i++) {
      const seed = ((i * 73 + cloudIndex * 29) % 997) / 997;
      particles.push({
        points,
        rest,
        alpha,
        inverse,
        i,
        seed,
        age: seed * (2.6 + seed),
        life: 2.6 + seed,
        position: new THREE.Vector3(),
        velocity: new THREE.Vector3(),
        active: false,
      });
    }
    return { points, inverse };
  });
  const local = new THREE.Vector3();
  let time = 0;
  return {
    update(dt) {
      time += dt;
      bird.updateWorldMatrix(true, true);
      for (const { points, inverse } of transforms) inverse.copy(points.matrixWorld).invert();
      for (const p of particles) {
        const { points, rest, alpha, inverse, i, seed, life, position, velocity } = p;
        p.age += dt;
        if (p.age >= life) {
          p.age %= life;
          p.active = false;
        }
        const progress = p.age / life;
        // Broad, continuous downstream movement rather than oscillating points.
        local.fromArray(rest, i * 3);
        const attached = 0.38;
        if (progress < attached || !p.active) {
          local.z += progress * 0.32;
          local.y += progress * 0.06;
          position.copy(local).applyMatrix4(points.matrixWorld);
          velocity
            .set(0.035 * Math.sin(seed * 6.28), 0.045, 0.3)
            .transformDirection(bird.matrixWorld)
            .multiplyScalar(0.22);
          p.active = true;
        } else {
          position.addScaledVector(velocity, dt);
          position.y += Math.sin(time * 0.65 + seed * 6.28) * dt * 0.006;
        }
        local.copy(position).applyMatrix4(inverse);
        points.geometry.attributes.position.setXYZ(i, local.x, local.y, local.z);
        const fadeIn = Math.min(1, progress / 0.12);
        const fadeOut = 1 - THREE.MathUtils.smoothstep(progress, 0.3, 1);
        alpha.setX(i, fadeIn * fadeOut * 0.8);
      }
      for (const { points } of transforms) {
        points.geometry.attributes.position.needsUpdate = true;
        points.geometry.attributes.birdAlpha.needsUpdate = true;
      }
    },
    reset() {
      particles.forEach((p) => {
        p.age = p.seed * p.life;
        p.active = false;
      });
      time = 0;
    },
  };
}
