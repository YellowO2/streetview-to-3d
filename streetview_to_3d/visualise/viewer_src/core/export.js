// The loaded scene as one binary PLY of coloured points: metres, Z up (east, north, up), origin
// at the scene centre. Points as loaded, not as styled; moving life is left out.
import * as THREE from 'three';
import { LAND, ROADS, WATER, BLOCKS } from '@viewer/core/scene-format';
import { waterColour } from '@viewer/world/water';

// each export choice: the scene.json keys it takes (null: DA3's own points)
export const PARTS = {
  street: [null],
  ground: ['terrain', LAND, ROADS, WATER],
  buildings: ['buildings', BLOCKS],
};

// the scene.json key a points object belongs to: its own or its parent's (water's points)
function keyOf(o) {
  for (; o; o = o.parent) if (o.userData.surroundings) return o.userData.surroundings;
  return null;
}

// group's points under the chosen parts (e.g. ['street', 'ground']) as a PLY Blob;
// center: [lat, lon] of the origin, written as a header comment
export function exportPly(group, parts, center) {
  const keys = new Set(parts.flatMap((p) => PARTS[p]));
  const chosen = [];
  group.updateMatrixWorld(true);
  group.traverse((o) => {
    // DA3 pieces hidden in the Scene Manager stay out
    if (o.isPoints && keys.has(keyOf(o)) && (o.userData.nodeIndex == null || o.visible))
      chosen.push(o);
  });
  const count = chosen.reduce((n, o) => n + o.geometry.getAttribute('position').count, 0);
  const header =
    'ply\nformat binary_little_endian 1.0\n' +
    (center
      ? `comment origin lat ${center[0]} lon ${center[1]}; x east, y north, z up (m)\n`
      : '') +
    `element vertex ${count}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
  const head = new TextEncoder().encode(header);
  const buffer = new ArrayBuffer(head.length + 15 * count),
    view = new DataView(buffer);
  new Uint8Array(buffer).set(head);
  const p = new THREE.Vector3(),
    c = new THREE.Color();
  let at = head.length;
  for (const o of chosen) {
    const g = o.geometry,
      position = g.getAttribute('position'),
      colour = g.getAttribute('color') ?? g.getAttribute('tint'),
      water = g.getAttribute('water');
    for (let i = 0; i < position.count; i++) {
      p.fromBufferAttribute(position, i).applyMatrix4(o.matrixWorld);
      // viewer frame (east, up, south) to east, north, up
      view.setFloat32(at, p.x, true);
      view.setFloat32(at + 4, -p.z, true);
      view.setFloat32(at + 8, p.y, true);
      if (colour) c.fromBufferAttribute(colour, i);
      else if (water) waterColour(water.getX(i), c);
      else c.copy(o.material.color ?? c.setRGB(1, 1, 1));
      c.convertLinearToSRGB();
      view.setUint8(at + 12, Math.round(THREE.MathUtils.clamp(c.r, 0, 1) * 255));
      view.setUint8(at + 13, Math.round(THREE.MathUtils.clamp(c.g, 0, 1) * 255));
      view.setUint8(at + 14, Math.round(THREE.MathUtils.clamp(c.b, 0, 1) * 255));
      at += 15;
    }
  }
  return new Blob([buffer], { type: 'application/octet-stream' });
}
