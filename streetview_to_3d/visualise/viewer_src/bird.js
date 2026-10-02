import { createBirdPlume } from '@viewer/bird-plume';
import { birdPaint, birdDabs, DAB } from '@viewer/bird-paint';
import { FLIGHT } from '@viewer/flight-motion';
import * as THREE from 'three';

// A faint white bird of dabs (bird-paint.js), as many as fit its shape SPACE
// of a dab apart, as the DA3 points are; articulated shoulders, delayed
// feather/tail motion. Its wingtips lay its trail (bird-plume.js).
const SPACE = 0.4; // dabs this much of one apart, overlapping as the DA3 points do
export function createBird() {
  const bird = new THREE.Group();
  bird.name = 'White particle bird';
  const clouds = [];
  const { material } = birdPaint();
  // the dabs' spacing in the bird's own units, its scale undone
  const step = (DAB * SPACE) / FLIGHT.birdScale;
  function points(parent, vertices) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    const p = birdDabs(g, material, vertices.length / 3);
    clouds.push(p);
    parent.add(p);
    return p;
  }
  const hash = (i, j) => Math.abs(Math.sin(i * 12.9898 + j * 78.233) * 43758.5453) % 1;
  function body(center, scale) {
    // as many as fit its surface (Thomsen's ellipsoid area)
    const [a, b, c] = scale,
      p = 1.6075,
      area = 4 * Math.PI * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p),
      count = Math.max(4, Math.round(area / step ** 2));
    const positions = [];
    for (let i = 0; i < count; i++) {
      const y = 1 - (2 * (i + 0.5)) / count,
        r = Math.sqrt(1 - y * y),
        theta = i * 2.39996323;
      positions.push(
        center[0] + Math.cos(theta) * r * a,
        center[1] + y * b,
        center[2] + Math.sin(theta) * r * c,
      );
    }
    points(bird, positions);
  }
  body([0, 0, 0], [0.16, 0.17, 0.43]);
  body([0, 0.13, -0.4], [0.135, 0.145, 0.18]);
  body([0, -0.08, -0.25], [0.13, 0.09, 0.19]);
  // feathers, each a pivot at its base, as a fan (overlapping): their dabs
  // on one grid over the fan, step apart (each a little off its cell's
  // centre), each moving with the feather it falls on (nearest its middle)
  function fan(parent, specs) {
    const feathers = specs.map(({ base, tip, width }) => {
      const pivot = new THREE.Group();
      pivot.position.set(...base);
      parent.add(pivot);
      const d = tip.map((x, i) => x - base[i]);
      pivot.userData.tip = new THREE.Vector3(...d);
      return { pivot, base, d, width, length: Math.hypot(d[0], d[2]), vertices: [] };
    });
    const xs = specs.flatMap(({ base, tip }) => [base[0], tip[0]]),
      zs = specs.flatMap(({ base, tip }) => [base[2], tip[2]]);
    const [x0, z0] = [Math.min(...xs), Math.min(...zs)];
    for (let i = 0; x0 + i * step <= Math.max(...xs); i++)
      for (let j = 0; z0 + j * step <= Math.max(...zs); j++) {
        const x = x0 + (i + hash(i, j)) * step - step / 2,
          z = z0 + (j + hash(j, i)) * step - step / 2;
        let best = null,
          nearest = 1;
        for (const f of feathers) {
          const px = x - f.base[0],
            pz = z - f.base[2],
            t = (px * f.d[0] + pz * f.d[2]) / f.length ** 2;
          if (t < 0 || t > 1) continue;
          const across = (px * -f.d[2] + pz * f.d[0]) / f.length,
            w = f.width * Math.pow(Math.sin(Math.PI * t), 0.65),
            cross = Math.abs(across) / w;
          if (cross < nearest) [best, nearest] = [{ f, t, px, pz }, cross];
        }
        if (!best) continue;
        const { f, t, px, pz } = best;
        f.vertices.push(px, f.d[1] * t + 0.045 * Math.sin(Math.PI * t) + 0.025 * nearest, pz);
      }
    for (const f of feathers) if (f.vertices.length) points(f.pivot, f.vertices);
    return feathers.map(({ pivot }) => pivot);
  }
  const wings = [],
    feathers = [];
  for (const side of [-1, 1]) {
    const shoulder = new THREE.Group();
    shoulder.position.set(side * 0.1, 0.02, -0.12);
    bird.add(shoulder);
    wings.push(shoulder);
    const ts = Array.from({ length: 12 }, (_, i) => i / 11);
    fan(
      shoulder,
      ts.map((t) => ({
        base: [side * (0.03 + t * 0.52), 0, t * 0.2],
        tip: [side * (0.42 + t * 0.85), -0.02, 0.52 - t * 0.3],
        width: 0.065,
      })),
    ).forEach((pivot, i) => feathers.push({ pivot, side, t: ts[i] }));
  }
  const tails = fan(
    bird,
    [-2, -1, 0, 1, 2].map((i) => ({
      base: [i * 0.035, -0.025, 0.3],
      tip: [i * 0.15, -0.13, 1.35 + (2 - Math.abs(i)) * 0.22],
      width: 0.06,
    })),
  );
  // the trail from each wing's outermost feather's tip
  const brushes = feathers.filter(({ t }) => t === 1).map(({ pivot }) => pivot);
  const plume = createBirdPlume(bird, brushes);
  let phase = 0,
    rate = 4;
  bird.visible = false;
  return {
    bird,
    wings,
    resetPlume() {
      plume.reset();
    },
    animate(dt, moving) {
      dt = Math.min(Math.max(dt, 0), 0.05);
      rate += ((moving ? 4.8 : 3.2) - rate) * (1 - Math.exp(-3 * dt));
      phase += dt * rate;

      const beat = Math.sin(phase);
      wings.forEach((wing, i) => {
        const side = i === 0 ? -1 : 1;
        wing.rotation.z = side * (0.12 + beat * 0.38);
        wing.rotation.y = side * (0.08 + Math.cos(phase) * 0.12);
        wing.rotation.x = Math.cos(phase - 0.3) * 0.08;
      });
      for (const { pivot, side, t } of feathers) {
        pivot.rotation.z = side * Math.sin(phase - 0.4 - t * 0.45) * 0.16 * t;
        pivot.rotation.y = side * (0.5 + 0.5 * Math.cos(phase)) * t * 0.15;
      }
      tails.forEach((tail, i) => {
        tail.rotation.x = Math.sin(phase - 0.9 - i * 0.12) * 0.065;
        tail.rotation.y = Math.sin(phase * 0.55 - i * 0.25) * 0.04;
      });
      plume.update(dt);
    },
    dispose() {
      clouds.forEach((points) => points.geometry.dispose());
      material.dispose();
      plume.dispose();
      bird.removeFromParent();
    },
  };
}
