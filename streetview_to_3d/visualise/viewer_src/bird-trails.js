import * as THREE from 'three';

// Sparse, short world-space histories: actual motion trails, not duplicate birds.
export function createBirdTrails(bird, clouds) {
  const samples = [];
  for (const cloud of clouds)
    for (let i = 0; i < cloud.geometry.attributes.position.count; i += 12)
      samples.push({ cloud, i, history: [] });
  const steps = 6,
    vertices = new Float32Array(samples.length * (steps - 1) * 6),
    colors = new Float32Array(vertices.length),
    geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(vertices, 3).setUsage(THREE.DynamicDrawUsage),
  );
  geometry.setAttribute(
    'color',
    new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage),
  );
  geometry.setDrawRange(0, 0);
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.05,
    depthWrite: false,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.frustumCulled = false;
  lines.matrixAutoUpdate = false;
  bird.add(lines);
  let elapsed = 0;
  return {
    reset() {
      samples.forEach((s) => (s.history.length = 0));
      geometry.setDrawRange(0, 0);
      elapsed = 0;
    },
    update(dt) {
      elapsed += dt;
      bird.updateWorldMatrix(true, true);
      lines.matrix.copy(bird.matrixWorld).invert();
      lines.matrixWorldNeedsUpdate = true;
      if (elapsed < 1 / 30) return;
      elapsed = 0;
      let cursor = 0;
      for (const sample of samples) {
        const { cloud, i, history } = sample;
        // Recycle the oldest sample once the short history is full.
        const point = history.length === steps ? history.pop() : new THREE.Vector3();
        history.unshift(
          point
            .fromBufferAttribute(cloud.geometry.attributes.position, i)
            .applyMatrix4(cloud.matrixWorld),
        );
        const color = cloud.geometry.attributes.color;
        for (let j = 0; j < history.length - 1; j++)
          for (let end = 0; end < 2; end++) {
            const p = history[j + end];
            vertices[cursor] = p.x;
            vertices[cursor + 1] = p.y;
            vertices[cursor + 2] = p.z;
            const fade = (1 - j / steps) * (cloud.geometry.attributes.birdAlpha?.getX(i) ?? 1);
            colors[cursor] = color.getX(i) * fade;
            colors[cursor + 1] = color.getY(i) * fade;
            colors[cursor + 2] = color.getZ(i) * fade;
            cursor += 3;
          }
      }
      geometry.setDrawRange(0, cursor / 3);
      geometry.attributes.position.needsUpdate = true;
      geometry.attributes.color.needsUpdate = true;
    },
    dispose() {
      lines.removeFromParent();
      geometry.dispose();
      material.dispose();
    },
  };
}
