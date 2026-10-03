import * as THREE from 'three';
import { carve } from '@viewer/flight/shot';
import { DISC, fibonacciSphere } from '@viewer/util';

// Shoot mode's gun: a point model at the eye's lower right firing glowing balls that carve
// holes (shot.js). Holding the button longer fires a bigger ball, up to BIG.
export const RADIUS = 1, // m: a click's shot and hole radius
  BIG = 4, // m: fully charged radius
  CHARGE_S = 4, // hold time to full charge
  SPEED = 30, // m/s
  AIM_M = 30; // shots from the muzzle cross the eye's line this far ahead
const LIFE_S = 3,
  MUZZLE = 1.6, // m: start ahead of the eye when no gun is held
  COOLDOWN_S = 0.15,
  SHELLS = [1, 0.66, 0.33], // ball point shells, of RADIUS
  SPACING = 0.12, // m between points on a shell
  POOL = 8;
const BALL = { size: 0.35, colour: 0x9ff3ff, opacity: 0.55 }, // additive glow
  HELD = new THREE.Vector3(0.16, -0.17, -0.42), // gun offset from the eye: right, down, ahead (m)
  STEP = 0.007, // model point spacing (m)
  DOT = 0.011; // model point size (m)
const DARK = [0.18, 0.2, 0.23],
  LIGHT = [0.55, 0.58, 0.62],
  GLOW = [0.55, 0.95, 1];
// parts (x right, y up, -z ahead; m): boxes [centre, size, colour], tubes [from z, to z, radius, colour]
const BOXES = [
    [[0, 0, 0], [0.06, 0.08, 0.24], DARK], // body
    [[0, -0.095, 0.07], [0.045, 0.12, 0.06], DARK], // grip
    [[0, 0.05, -0.02], [0.02, 0.02, 0.09], LIGHT], // sight
    [[0.031, 0, -0.02], [0.002, 0.012, 0.18], GLOW], // light strip
  ],
  TUBES = [[-0.12, -0.4, 0.022, LIGHT]], // barrel
  RING = { z: -0.4, radius: 0.03, colour: GLOW }; // the muzzle

// the model's points over box faces, tube sides and the muzzle ring
function model() {
  const pos = [],
    col = [];
  const put = (p, c) => {
    pos.push(...p);
    col.push(...c);
  };
  for (const [[cx, cy, cz], size, c] of BOXES)
    for (let axis = 0; axis < 3; axis++) {
      const [u, v] = [0, 1, 2].filter((a) => a !== axis);
      for (const side of [-0.5, 0.5])
        for (let i = 0; i <= Math.ceil(size[u] / STEP); i++)
          for (let j = 0; j <= Math.ceil(size[v] / STEP); j++) {
            const p = [0, 0, 0];
            p[axis] = side * size[axis];
            p[u] = (i / Math.ceil(size[u] / STEP) - 0.5) * size[u];
            p[v] = (j / Math.ceil(size[v] / STEP) - 0.5) * size[v];
            put([cx + p[0], cy + p[1], cz + p[2]], c);
          }
    }
  for (const [from, to, r, c] of TUBES) {
    const around = Math.ceil((2 * Math.PI * r) / STEP),
      along = Math.ceil(Math.abs(to - from) / STEP);
    for (let i = 0; i < around; i++)
      for (let j = 0; j <= along; j++) {
        const a = (2 * Math.PI * i) / around;
        put([Math.cos(a) * r, Math.sin(a) * r, from + ((to - from) * j) / along], c);
      }
  }
  const around = Math.ceil((2 * Math.PI * RING.radius) / (STEP / 2));
  for (let i = 0; i < around; i++) {
    const a = (2 * Math.PI * i) / around;
    put([Math.cos(a) * RING.radius, Math.sin(a) * RING.radius, RING.z], RING.colour);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return g;
}

// round points; soft fades them to the edge
function round(material, soft) {
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      `${DISC}
      ${soft ? 'diffuseColor.a *= 1. - smoothstep(0., 1., r);' : ''}
      #include <opaque_fragment>`,
    );
  };
  material.customProgramCacheKey = () => `gun-round:${soft}`;
  return material;
}

export function createGun(scene) {
  const ballMaterial = round(
    new THREE.PointsMaterial({
      size: BALL.size,
      color: BALL.colour,
      transparent: true,
      opacity: BALL.opacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
    true,
  );
  const positions = [];
  for (const shell of SHELLS) {
    const r = RADIUS * shell,
      count = Math.max(8, Math.round((4 * Math.PI * r * r) / SPACING ** 2));
    for (const p of fibonacciSphere(count)) positions.push(...p.map((x) => x * r));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const shots = Array.from({ length: POOL }, () => {
    const ball = new THREE.Points(geometry, ballMaterial);
    ball.userData.styleAnimated = true; // not a world point: no demos or shots
    ball.frustumCulled = false;
    ball.name = 'Shot';
    ball.visible = false;
    scene.add(ball);
    return { ball, way: new THREE.Vector3(), age: 0, from: new THREE.Vector3(), radius: RADIUS };
  });
  const held = new THREE.Points(
    model(),
    round(new THREE.PointsMaterial({ size: DOT, vertexColors: true }), false),
  );
  held.userData.styleAnimated = true;
  held.name = 'Gun';
  held.frustumCulled = false;
  held.visible = false;
  scene.add(held);
  const muzzle = new THREE.Vector3(),
    aim = new THREE.Vector3();
  let next = 0,
    wait = 0,
    holding = -1; // seconds the button has been held; -1 when up
  const gun = {
    shots,
    held,
    // show the gun at the camera, or hide it
    hold(camera, on) {
      held.visible = on;
      if (!on) return;
      held.quaternion.copy(camera.quaternion);
      held.position.copy(HELD).applyQuaternion(camera.quaternion).add(camera.position);
      held.updateMatrixWorld();
    },
    press() {
      holding = 0;
    },
    // fire a shot sized by how long the button was held
    release(origin, way) {
      if (holding < 0) return false;
      const radius = RADIUS + (BIG - RADIUS) * Math.min(1, holding / CHARGE_S);
      holding = -1;
      return gun.fire(origin, way, radius);
    },
    // fire from the eye (origin) along unit way: from the muzzle toward the aim point when held
    fire(origin, way, radius = RADIUS) {
      if (wait > 0) return false;
      wait = COOLDOWN_S;
      const shot = shots[next];
      next = (next + 1) % POOL;
      shot.radius = radius;
      shot.ball.scale.setScalar(radius / RADIUS);
      if (held.visible) {
        muzzle.set(0, 0, RING.z).applyMatrix4(held.matrixWorld);
        aim.copy(origin).addScaledVector(way, AIM_M);
        shot.ball.position.copy(muzzle);
        shot.way.subVectors(aim, muzzle).normalize();
      } else {
        shot.ball.position.copy(origin).addScaledVector(way, MUZZLE);
        shot.way.copy(way);
      }
      shot.from.copy(shot.ball.position);
      shot.age = 0;
      shot.ball.visible = true;
      return true;
    },
    update(dt) {
      wait -= dt;
      if (holding >= 0) holding += dt;
      for (const shot of shots) {
        if (!shot.ball.visible) continue;
        shot.age += dt;
        shot.from.copy(shot.ball.position);
        shot.ball.position.addScaledVector(shot.way, SPEED * dt);
        if (!carve(shot.from, shot.ball.position, shot.radius) || shot.age >= LIFE_S)
          shot.ball.visible = false;
      }
    },
    reset() {
      for (const shot of shots) shot.ball.visible = false;
      wait = 0;
      holding = -1;
    },
    dispose() {
      shots.forEach(({ ball }) => ball.removeFromParent());
      held.removeFromParent();
      geometry.dispose();
      ballMaterial.dispose();
      held.geometry.dispose();
      held.material.dispose();
    },
  };
  return gun;
}
