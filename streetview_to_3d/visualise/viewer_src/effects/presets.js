// Shared defaults keep UI controls and renderer settings in sync.
const BASE = {
  strength: 1,
  density: 100,
  pointSize: 1,
  pixels: 3,
  floating: false,
  amount: 0.45,
  scan: false,
  atmosphere: true,
};
export const STYLE_DEFAULTS = {
  original: { ...BASE },
  paint: { ...BASE, strength: 0.61, density: 100, pointSize: 1.4, amount: 0.3, floating: true },
  dither: { ...BASE },
  matrix: { ...BASE, density: 100, atmosphere: false },
};

export function normalizeStyle(name) {
  return name === 'anime' || name === 'painterly' ? 'paint' : name;
}
