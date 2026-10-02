import * as THREE from 'three';

export const FLIGHT = Object.freeze({
  birdScale: 0.29,
  speed: 2,
  boost: 3,
  distance: 2.4,
  height: 0.6,
});

// Exact integration of exponentially approaching a target velocity. Same travel
// at different frame rates, including acceleration and braking.
export function advanceFlight(position, velocity, target, dt) {
  const response = 5;
  const decay = Math.exp(-response * dt);
  const weight = (1 - decay) / response;
  position.x += target.x * dt + (velocity.x - target.x) * weight;
  position.y += target.y * dt + (velocity.y - target.y) * weight;
  position.z += target.z * dt + (velocity.z - target.z) * weight;
  velocity.lerp(target, 1 - decay);
}

// The bird's way, as a bird flies: facing where it goes -- the camera's way
// (heading) as it slows to still -- its nose up or down as it climbs or dives
// (at most PITCH), banked into its turns (the faster and sharper, the more,
// at most BANK); each eased, never snapped. `state` ({ yaw, pitch, roll })
// carries it frame to frame.
const PITCH = 0.6,
  BANK = 0.6,
  TURN = 4, // how fast it turns to face its way (1/s)
  EASE = 5; // how fast its pitch and bank follow (1/s)
const euler = new THREE.Euler(0, 0, 0, 'YXZ');
export function steerBird(quaternion, velocity, heading, state, dt) {
  const ahead = euler.setFromQuaternion(heading, 'YXZ').y,
    level = Math.hypot(velocity.x, velocity.z),
    going = THREE.MathUtils.smoothstep(level, 0.1, 0.6);
  // from the camera's way toward where it goes, by the shortest way round
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
