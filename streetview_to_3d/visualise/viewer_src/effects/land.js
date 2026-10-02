import * as THREE from 'three';
import { covering, marks, pointsOf } from '@viewer/effects/blocks';

// The land (land.ply, postprocess/terrain.py: triangles, its colours the
// satellite's and the panos') as points, as the buildings are (blocks.js):
// on a grid fixed in the world, as far apart as the world's points are
// there (gapOf, never under MIN_GAP: near by the scene's own ground stands
// over it), finer on a slope and the points as much bigger; each as big as
// covers what JITTER moves them apart (blocks.covering), so nothing under
// the land ever shows between them -- their spacing (SPACE) the one thing
// to choose: closer, smaller points, more of them. Lit as the land faces
// there, hazed as everything is, and turning into DA3's points by
// land.ply's near (postprocess/seams.toward).
export const MIN_GAP = 0.3;
const SPACE = 1, // its points this much of the world's spacing apart: closer than a building's
  JITTER = 0.1; // each off its grid's place at most this much of its spacing: a third of a building's

// The land's triangles (the viewer's frame), each corner its facing (up),
// colour and spacing (gapOf(x, z): the world's points' there, never under MIN_GAP).
function prepare(geometry, gapOf) {
  geometry.computeVertexNormals();
  // a triangle wound either way: every normal up
  const nor = geometry.getAttribute('normal');
  for (let i = 0; i < nor.count; i++)
    if (nor.getY(i) < 0) nor.setXYZ(i, -nor.getX(i), -nor.getY(i), -nor.getZ(i));
  const p = geometry.getAttribute('position');
  const gap = new Float32Array(p.count).map((_, i) =>
    Math.max(MIN_GAP, gapOf(p.getX(i), p.getZ(i))),
  );
  geometry.setAttribute('gap', new THREE.Float32BufferAttribute(gap, 1));
  if (!geometry.getAttribute('color'))
    geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(new Float32Array(3 * p.count).fill(0.5), 3),
    );
  return geometry;
}

// The land's triangles (the viewer's frame) as its points; gapOf(x, z): the
// world's points' spacing there.
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
