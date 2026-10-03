// Hooks every shader drawing the world's points shares: demos, shots, characters and far-off thinning.
import { demo, DEMO } from '@viewer/effects/demo';
import { shot, SHOT } from '@viewer/effects/shot';
import { glyphs, GLYPH, GLYPH_GROW } from '@viewer/effects/glyphs';
import { THIN } from '@viewer/effects/thin';

export const worldUniforms = { ...demo, ...shot, ...glyphs };

// GLSL, vertex. worldSize: a point's pixel size grown for characters and thinned far off; 0 if not drawn.
export const WORLD_VERTEX =
  DEMO +
  SHOT +
  GLYPH_GROW +
  THIN +
  `
  float worldSize(float px, float growSeed, float thinSeed) {
    px *= glyphGrow(growSeed);
    return px * thin(px, thinSeed);
  }
`;

// GLSL, fragment
export const WORLD_FRAGMENT = GLYPH;

// GLSL, fragment: sets pointShade (a character's rim darker) and discards pixels off the
// point's character (when seed is given and characters are on) or where offDisc holds.
export const cutPoint = (seed, offDisc) => `
  float pointShade = 1.;
  ${seed ? `if (glyphOn > .5) { pointShade = glyphAt(gl_PointCoord, ${seed}); if (pointShade == 0.) discard; } else` : ''}
  if (${offDisc}) discard;`;
