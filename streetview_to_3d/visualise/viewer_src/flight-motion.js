import * as THREE from 'three';

export const FLIGHT = Object.freeze({
  birdScale: 0.29,
  speed: 2,
  boost: 3,
  distance: 2.4,
  height: 0.6,
});

// Exact integration of velocity easing exponentially to target: frame-rate independent.
export function advanceFlight(position, velocity, target, dt) {
  const response = 5;
  const decay = Math.exp(-response * dt);
  const weight = (1 - decay) / response;
  position.x += target.x * dt + (velocity.x - target.x) * weight;
  position.y += target.y * dt + (velocity.y - target.y) * weight;
  position.z += target.z * dt + (velocity.z - target.z) * weight;
  velocity.lerp(target, 1 - decay);
}

// Bird orientation: faces its velocity (the camera heading when still), pitches with climb and
// banks into turns, all eased. state ({ yaw, pitch, roll }) carries over between frames.
const PITCH = 0.6, // radians
  BANK = 0.6, // radians
  TURN = 4, // yaw easing rate (1/s)
  EASE = 5; // pitch and bank easing rate (1/s)
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
export function steerBird(quaternion, velocity, heading, state, dt) {
  const ahead = euler.setFromQuaternion(heading, 'YXZ').y,
    level = Math.hypot(velocity.x, velocity.z),
    going = THREE.MathUtils.smoothstep(level, 0.1, 0.6);
  // blend from the camera heading toward the travel direction, the short way round
  const toward = level > 1e-6 ? Math.atan2(-velocity.x, -velocity.z) : ahead,
    wanted = ahead + going * shortest(toward - ahead),
    turn = shortest(wanted - state.yaw) * (1 - Math.exp(-TURN * dt));
  state.yaw += turn;
  const rate = dt > 0 ? turn / dt : 0,
    climb = THREE.MathUtils.clamp(Math.atan2(velocity.y, Math.max(level, 0.5)), -PITCH, PITCH),
    bank = THREE.MathUtils.clamp(rate * level * 0.3, -BANK, BANK),
    ease = 1 - Math.exp(-EASE * dt);
  state.pitch += (climb - state.pitch) * ease;
  state.roll += (bank - state.roll) * ease;
  return quaternion.setFromEuler(euler.set(state.pitch, state.yaw, state.roll, 'YXZ'));
}
const shortest = (a) => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));
