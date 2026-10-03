// Far off, fewer points: a point drawn under MIN_PX pixels across is drawn
// MIN_PX across or not at all -- so many of them kept ((its size over
// MIN_PX) squared; each its own, fixed) that they cover the world as much as
// ever, but there are far fewer to draw where the world is far and its points
// tiny on the screen, piled onto the same pixels. Every shader that draws
// the world's points sizes them by THIN: DA3's and the map's (points.js), the
// buildings' and the land's (blocks.js), the water's (water.js).
export const MIN_PX = 5;

// GLSL, for a vertex shader: how many times as big a point px pixels across
// (seed: its own, 0..1) is drawn -- 0 for one not drawn
export const THIN = `
  float thin(float px, float seed) {
    if (px >= ${MIN_PX.toFixed(1)} || px <= 0.) return 1.;
    float k = px / ${MIN_PX.toFixed(1)};
    return fract(seed * 29.71 + .13) < k * k ? 1. / k : 0.;
  }
`;
