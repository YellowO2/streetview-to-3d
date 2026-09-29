// Shared defaults keep UI controls and renderer settings in sync.
const BASE = {
  strength: 1,
  density: 100,
  pointSize: 1,
  blocks: 1,
  pixels: 3,
  floating: false,
  amount: 0.45,
  scan: false,
  atmosphere: true,
};
export const STYLE_DEFAULTS = {
  original: { ...BASE },
  paint: { ...BASE, strength: 0.61, density: 95, pointSize: 0.8, amount: 0.25, floating: true },
  voxel: { ...BASE, amount: 0 },
  dither: { ...BASE },
};

export function normalizeStyle(name) {
  return name === 'anime' || name === 'painterly' ? 'paint' : name;
}
