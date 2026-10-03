import * as THREE from 'three';

// The world's points as characters (the Characters style): each point the
// shape of one of a set of characters (setGlyphs: '01' the Matrix's), the
// one its own seed picks, fixed -- its colour, size and motion its own, as
// ever; only its shape cut out by the character instead of a disc. A
// character only reads some pixels across, so they are drawn glyphSize
// times as big and, to fill the world as much as ever, only so many of them
// (KEEP of glyphSize squared; each its own, fixed) -- for every point, its
// size as its spacing has it or not (glyphGrow). Every
// shader that draws the world's points takes glyphs' uniforms and GLYPH:
// DA3's and the map's (points.js), the buildings' and the land's
// (blocks.js), the water's (water.js).
const CELL = 64, // a character this many pixels square in the atlas
  MOST = 32, // the atlas room for this many characters, always: a texture resized keeps its old pixels on the GPU
  RIM = 0.07, // round each character a rim this much of its cell wide (as subtitles have one), so neighbours read apart
  SHADE = 0.3, // ... its point's own colour this dark
  KEEP = 6.3; // as big, the characters drawn this many times over the world (70% of them at 3x): a character's strokes fill a little of its square

const canvas = globalThis.document?.createElement('canvas'),
  context = canvas?.getContext?.('2d'),
  atlas = context ? new THREE.CanvasTexture(canvas) : null;
if (atlas) atlas.flipY = false; // the atlas as drawn: its top row at v 0

// shared by every shader the characters are cut in
export const glyphs = {
  glyphOn: { value: 0 },
  glyphAtlas: { value: atlas },
  glyphCount: { value: 1 },
  glyphSize: { value: 3 },
};

// the atlas: a row of the characters a rim wider, under it a row of them as they are
if (canvas) {
  canvas.width = CELL * MOST;
  canvas.height = 2 * CELL;
}
// a string's characters as one reads them: an emoji with its skin tone or style one, not its codes
const split = (text) =>
  globalThis.Intl?.Segmenter
    ? [...new Intl.Segmenter().segment(text)].map((g) => g.segment)
    : [...text];

// the points as these characters (a string; none: as discs), size times as big
export function setGlyphs(characters, size = 3) {
  glyphs.glyphSize.value = size;
  const chars = split(characters ?? '')
    .filter((c) => c.trim())
    .slice(0, MOST);
  glyphs.glyphOn.value = chars.length && atlas ? 1 : 0;
  if (!glyphs.glyphOn.value) return;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#fff';
  context.font = `900 ${Math.round(CELL * 0.95)}px ui-monospace, Menlo, monospace`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  chars.forEach((c, i) => {
    const x = (i + 0.5) * CELL,
      y = CELL * 0.54;
    context.fillText(c, x, CELL + y);
    for (let k = 0; k < 12; k++) {
      const a = (k / 12) * 2 * Math.PI;
      context.fillText(c, x + Math.cos(a) * RIM * CELL, y + Math.sin(a) * RIM * CELL);
    }
  });
  atlas.needsUpdate = true;
  glyphs.glyphCount.value = chars.length;
}

// GLSL, for a vertex shader: how many times as big a point (seed: its own,
// 0..1) is drawn -- 0 for one not drawn
export const GLYPH_GROW = `
  uniform float glyphOn, glyphSize;
  float glyphGrow(float seed) {
    if (glyphOn < .5) return 1.;
    return fract(seed * 13.17 + .71) < ${KEEP.toFixed(2)} / (glyphSize * glyphSize) ? glyphSize : 0.;
  }
`;

// GLSL, for a fragment shader: how this pixel of a point (its gl_PointCoord)
// lies on its character (seed: the point's own, 0..1) -- 1 on it, SHADE on
// its rim (its colour that dark), 0 off it
export const GLYPH = `
  uniform float glyphOn, glyphCount;
  uniform sampler2D glyphAtlas;
  float glyphAt(vec2 at, float seed) {
    float i = min(floor(fract(seed * 7.13 + .37) * glyphCount), glyphCount - 1.);
    vec2 rim = vec2((i + at.x) / ${MOST.toFixed(1)}, at.y * .5); // gl_PointCoord: y down, as the atlas
    if (texture2D(glyphAtlas, rim + vec2(0., .5)).a > .5) return 1.;
    return texture2D(glyphAtlas, rim).a > .5 ? ${SHADE.toFixed(2)} : 0.;
  }
`;
