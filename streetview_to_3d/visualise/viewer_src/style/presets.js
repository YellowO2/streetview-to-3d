// Style presets shared by the UI controls and the renderer.
const BASE = {
  strength: 1,
  density: 100,
  pointSize: 1,
  pixels: 3,
  floating: false,
  amount: 0.45,
  scan: false,
  atmosphere: true,
  characters: '', // points drawn as these characters (glyphs.js); empty: discs
  characterSize: 3, // character size multiplier (fewer drawn as they grow)
};
export const STYLE_DEFAULTS = {
  original: { ...BASE },
  paint: { ...BASE, strength: 0.61, density: 100, pointSize: 1.4, amount: 0.3, floating: true },
  dither: { ...BASE },
};
// Characters: Soft paint with points drawn as characters
STYLE_DEFAULTS.characters = { ...STYLE_DEFAULTS.paint, characters: '01' };

export function normalizeStyle(name) {
  return name === 'anime' || name === 'painterly' ? 'paint' : name;
}
