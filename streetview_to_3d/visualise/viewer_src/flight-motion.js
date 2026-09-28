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
