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
  characters: '', // the points as these characters (glyphs.js); none: discs
  characterSize: 3, // ... this many times as big, so many fewer
};
export const STYLE_DEFAULTS = {
  original: { ...BASE },
  paint: { ...BASE, strength: 0.61, density: 100, pointSize: 1.4, amount: 0.3, floating: true },
  dither: { ...BASE },
};
// the points as characters: painted as Soft paint is, each point a character ('01': the Matrix)
STYLE_DEFAULTS.characters = { ...STYLE_DEFAULTS.paint, characters: '01' };

export function normalizeStyle(name) {
  return name === 'anime' || name === 'painterly' ? 'paint' : name;
}
