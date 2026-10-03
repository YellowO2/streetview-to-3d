// Points under MIN_PX px are drawn at MIN_PX, each kept with probability (px/MIN_PX)^2.
const MIN_PX = 5;

// GLSL, vertex: size multiplier for a point px pixels across with seed in 0..1; 0 if dropped
export const THIN = `
  float thin(float px, float seed) {
    if (px >= ${MIN_PX.toFixed(1)} || px <= 0.) return 1.;
    float k = px / ${MIN_PX.toFixed(1)};
    return fract(seed * 29.71 + .13) < k * k ? 1. / k : 0.;
  }
`;
