import * as THREE from 'three';
import { marks, pointsOf } from '@viewer/world/blocks';

// The land (land.ply triangles) drawn as dabs like the buildings (blocks.js), spaced by its gap
// (or gapOf), blending into DA3 by `near`.

// Adds a gap per corner (the ply's, or gapOf(x, z)), a default colour and, if smooth, upward
// normals.
function prepare(geometry, gapOf, smooth) {
  if (smooth) {
    geometry.computeVertexNormals();
    // triangles may be wound either way: flip normals up
    const nor = geometry.getAttribute('normal');
    for (let i = 0; i < nor.count; i++)
      if (nor.getY(i) < 0) nor.setXYZ(i, -nor.getX(i), -nor.getY(i), -nor.getZ(i));
  }
  const p = geometry.getAttribute('position'),
    own = geometry.getAttribute('gap');
  const gap = new Float32Array(p.count).map((_, i) =>
    own ? own.getX(i) : gapOf(p.getX(i), p.getZ(i)),
  );
  geometry.setAttribute('gap', new THREE.Float32BufferAttribute(gap, 1));
  if (!geometry.getAttribute('color'))
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(new Float32Array(3 * p.count).fill(0.5), 3),
    );
  return geometry;
}

// Triangles (viewer frame) as the land's dabs; gapOf(x, z): point spacing there. smooth: facing
// from corner normals, else each triangle's own; colourAt: see blocks.marks. Disposes geometry.
export function landMarks(geometry, gapOf, { smooth = true, colourAt } = {}) {
  const made = marks(prepare(geometry, gapOf, smooth), {
    edges: false,
    levels: true,
    smooth,
    colourAt,
  });
  geometry.dispose();
  return made;
}

// land.ply triangles (viewer frame) as drawn dabs
export function landPoints(geometry, gapOf) {
  return pointsOf(landMarks(geometry, gapOf), { fog: true });
}
