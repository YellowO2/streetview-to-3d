import * as THREE from 'three';

// Characters style: each world point cut to a character its seed picks, instead of a disc.
// Characters are drawn glyphSize times bigger and kept with probability KEEP / glyphSize^2.
const CELL = 64, // atlas cell size (px)
  MOST = 32, // atlas capacity, fixed: a resized texture keeps its old pixels on the GPU
  RIM = 0.07, // outline width, of the cell, so neighbours read apart
  SHADE = 0.3, // outline brightness, of the point colour
  KEEP = 6.3; // coverage factor (70% kept at 3x): strokes fill only part of a cell

const canvas = globalThis.document?.createElement('canvas'),
  context = canvas?.getContext?.('2d'),
  atlas = context ? new THREE.CanvasTexture(canvas) : null;
if (atlas) atlas.flipY = false; // top row at v 0

export const glyphs = {
  glyphOn: { value: 0 },
  glyphAtlas: { value: atlas },
  glyphCount: { value: 1 },
  glyphSize: { value: 3 },
};

// atlas: top row the outlined characters, bottom row the plain ones
if (canvas) {
  canvas.width = CELL * MOST;
  canvas.height = 2 * CELL;
}
// split into user-perceived characters (emoji with modifiers stay whole)
const split = (text) =>
  globalThis.Intl?.Segmenter
    ? [...new Intl.Segmenter().segment(text)].map((g) => g.segment)
    : [...text];

// draw points as these characters (empty: discs), size times bigger
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

// GLSL, vertex: size multiplier for a point with seed in 0..1; 0 if dropped
export const GLYPH_GROW = `
  uniform float glyphOn, glyphSize;
  float glyphGrow(float seed) {
    if (glyphOn < .5) return 1.;
    return fract(seed * 13.17 + .71) < ${KEEP.toFixed(2)} / (glyphSize * glyphSize) ? glyphSize : 0.;
  }
`;

// GLSL, fragment: 1 on the point's character, SHADE on its outline, 0 off it
export const GLYPH = `
  uniform float glyphOn, glyphCount;
  uniform sampler2D glyphAtlas;
  float glyphAt(vec2 at, float seed) {
    float i = min(floor(fract(seed * 7.13 + .37) * glyphCount), glyphCount - 1.);
    vec2 rim = vec2((i + at.x) / ${MOST.toFixed(1)}, at.y * .5); // gl_PointCoord y runs down, like the atlas
    if (texture2D(glyphAtlas, rim + vec2(0., .5)).a > .5) return 1.;
    return texture2D(glyphAtlas, rim).a > .5 ? ${SHADE.toFixed(2)} : 0.;
  }
`;
