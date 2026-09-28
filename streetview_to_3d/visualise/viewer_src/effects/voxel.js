import { createGeometryStyle } from '@viewer/effects/geometry-style';
import * as THREE from 'three';

// Average colours within occupied cells. Grow cells if necessary to bound GPU memory.
export function voxelCells(geometry, requestedSize, limit = 60000) {
  const position = geometry.getAttribute('position'),
    colour = geometry.getAttribute('color');
  if (!geometry.boundingBox) geometry.computeBoundingBox();
  const origin = geometry.boundingBox.min;
  let size = Math.max(requestedSize, 1e-6);
  for (;;) {
    const cells = new Map();
    for (let i = 0; i < position.count; i++) {
      const x = Math.floor((position.getX(i) - origin.x) / size),
        y = Math.floor((position.getY(i) - origin.y) / size),
        z = Math.floor((position.getZ(i) - origin.z) / size);
      const key = `${x},${y},${z}`;
      let cell = cells.get(key);
      if (!cell) {
        cell = { x, y, z, r: 0, g: 0, b: 0, count: 0 };
        cells.set(key, cell);
      }
      cell.r += colour ? colour.getX(i) : 0.4;
      cell.g += colour ? colour.getY(i) : 0.6;
      cell.b += colour ? colour.getZ(i) : 0.7;
      cell.count++;
      if (cells.size > limit) break;
    }
    if (cells.size <= limit) return { size, origin, cells: [...cells.values()] };
    size *= 1.5;
  }
}

export function createVoxels(scene) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshLambertMaterial();
  return createGeometryStyle(
    scene,
    (source, size, count) => {
      const data = voxelCells(source.geometry, size, Math.max(1, Math.floor(60000 / count)));
      const mesh = new THREE.InstancedMesh(geometry, material, data.cells.length);
      const matrix = new THREE.Matrix4(),
        colour = new THREE.Color();
      data.cells.forEach((cell, i) => {
        matrix.makeScale(data.size * 0.97, data.size * 0.97, data.size * 0.97);
        matrix.setPosition(
          (cell.x + 0.5) * data.size + data.origin.x,
          (cell.y + 0.5) * data.size + data.origin.y,
          (cell.z + 0.5) * data.size + data.origin.z,
        );
        mesh.setMatrixAt(i, matrix);
        mesh.setColorAt(
          i,
          colour.setRGB(cell.r / cell.count, cell.g / cell.count, cell.b / cell.count),
        );
      });
      mesh.userData.disposeStyle = () => mesh.dispose();
      return mesh;
    },
    () => {
      geometry.dispose();
      material.dispose();
    },
  );
}
