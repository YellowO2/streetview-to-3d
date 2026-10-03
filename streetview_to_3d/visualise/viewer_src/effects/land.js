import * as THREE from 'three';
import { covering, marks, pointsOf } from '@viewer/effects/blocks';

// The land (land.ply triangles) drawn as dabs like the buildings (blocks.js), spaced by its gap
// (or gapOf), each just big enough to leave no gaps (blocks.covering), blending into DA3 by `near`.
export const MIN_GAP = 0.3; // m: smallest spacing when the ply has no gap
const SPACE = 1, // dab spacing, of the gap
  JITTER = 0.1; // of the spacing, a third of a building's

// Adds upward normals, a gap per corner (the ply's, or gapOf(x, z)) and a default colour.
function prepare(geometry, gapOf) {
  geometry.computeVertexNormals();
  // triangles may be wound either way: flip normals up
  const nor = geometry.getAttribute('normal');
  for (let i = 0; i < nor.count; i++)
    if (nor.getY(i) < 0) nor.setXYZ(i, -nor.getX(i), -nor.getY(i), -nor.getZ(i));
  const p = geometry.getAttribute('position'),
    own = geometry.getAttribute('gap');
  const gap = new Float32Array(p.count).map((_, i) =>
    own ? own.getX(i) : Math.max(MIN_GAP, gapOf(p.getX(i), p.getZ(i))),
  );
  geometry.setAttribute('gap', new THREE.Float32BufferAttribute(gap, 1));
  if (!geometry.getAttribute('color'))
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(new Float32Array(3 * p.count).fill(0.5), 3),
    );
  return geometry;
}

// land.ply triangles (viewer frame) as drawn dabs; gapOf(x, z): point spacing there
export function landPoints(geometry, gapOf) {
  const made = marks(prepare(geometry, gapOf), {
    edges: false,
    levels: true,
    smooth: true,
    space: SPACE,
    jitter: JITTER,
    round: covering(JITTER),
  });
  geometry.dispose();
  return pointsOf(made, { fog: true });
}
