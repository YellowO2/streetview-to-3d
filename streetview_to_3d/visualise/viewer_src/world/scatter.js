// Triangle surface plys, and the spacings their dabs' world-fixed grids snap to (blocks.marks):
// a triangle's gap rounded up to GAPS, so neighbouring triangles join without seams.
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// allowed point spacings (m), each 1.25x the last
export const GAPS = Array.from({ length: 28 }, (_, k) => 0.05 * 1.25 ** k);
export const FLAT = 0.7; // normal.y above this: gridded east/north, else along-slope/up

// index of the smallest of GAPS at least gap
export function level(gap) {
  let k = 0;
  while (k < GAPS.length - 1 && GAPS[k] < gap * 0.999) k++;
  return k;
}

const loader = new PLYLoader();
loader.setCustomPropertyNameMapping({
  facade: ['facade_u', 'facade_v'],
  facadeLayout: ['facade_bay', 'facade_floor'],
  glass: ['glass_r', 'glass_g', 'glass_b'],
  gap: ['gap'],
});

// A surface ply (also blocks.ply's facade and glass) as triangles in the viewer frame (flip: y up, z south).
export function parseSurface(buffer, flip, name) {
  const geometry = loader.parse(buffer);
  if (!geometry.getAttribute('position')?.count || !geometry.index) {
    geometry.dispose();
    throw Error(`${name} has no triangles.`);
  }
  const layout = geometry.getAttribute('facadeLayout');
  if (layout && !layout.array.some((value) => Number.isFinite(value) && value > 0)) {
    geometry.deleteAttribute('facadeLayout');
  }
  geometry.applyMatrix4(flip);
  return geometry;
}
