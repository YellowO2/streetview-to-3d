import * as THREE from 'three';

// Sparse world-space wing histories leave a continuous wake behind flight.
export function createBirdTrails(bird, clouds, restPositions = []) {
  const samples = [];
  clouds.forEach((cloud, index) => {
    for (let i = 0; i < cloud.geometry.attributes.position.count; i += 60)
      samples.push({ cloud, i, rest: restPositions[index], history: [] });
  });
  const steps = 48,
    vertices = new Float32Array(samples.length * (steps - 1) * 6),
    colors = new Float32Array(vertices.length),
    alphas = new Float32Array(vertices.length / 3),
    geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(vertices, 3).setUsage(THREE.DynamicDrawUsage),
  );
  geometry.setAttribute(
    'color',
    new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage),
  );
  geometry.setAttribute(
    'trailAlpha',
    new THREE.BufferAttribute(alphas, 1).setUsage(THREE.DynamicDrawUsage),
  );
  geometry.setDrawRange(0, 0);
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.3,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader =
      'attribute float trailAlpha; varying float vTrailAlpha;\n' +
      shader.vertexShader.replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n vTrailAlpha = trailAlpha;',
      );
    shader.fragmentShader =
      'varying float vTrailAlpha;\n' +
      shader.fragmentShader.replace(
        '#include <opaque_fragment>',
        'diffuseColor.a *= vTrailAlpha;\n #include <opaque_fragment>',
      );
  };
  material.customProgramCacheKey = () => 'bird-wake-alpha-v1';
  const lines = new THREE.LineSegments(geometry, material);
  lines.frustumCulled = false;
  lines.matrixAutoUpdate = false;
  // A faint radial halo shares the trail buffers; no full-screen bloom pass.
  const haloMaterial = new THREE.PointsMaterial({
    vertexColors: true,
    size: 0.025,
    transparent: true,
    opacity: 0.035,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  haloMaterial.onBeforeCompile = (shader) => {
    material.onBeforeCompile(shader);
    shader.fragmentShader = shader.fragmentShader.replace(
      'diffuseColor.a *= vTrailAlpha;',
      'float radius = length(gl_PointCoord - .5) * 2.; diffuseColor.a *= vTrailAlpha * exp(-radius * radius * 5.) * (1. - smoothstep(.7, 1., radius));',
    );
  };
  haloMaterial.customProgramCacheKey = () => 'bird-wake-halo-v1';
  const halo = new THREE.Points(geometry, haloMaterial);
  halo.frustumCulled = false;
  lines.add(halo);
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
        const { cloud, i, rest, history } = sample;
        // Recycle the oldest sample once the short history is full.
        const point = history.length === steps ? history.pop() : new THREE.Vector3();
        // Track the emitter, not recycled plume particles: no respawn streaks.
        if (rest) point.fromArray(rest, i * 3);
        else point.fromBufferAttribute(cloud.geometry.attributes.position, i);
        history.unshift(point.applyMatrix4(cloud.matrixWorld));
        const color = cloud.geometry.attributes.color;
        for (let j = 0; j < history.length - 1; j++)
          for (let end = 0; end < 2; end++) {
            const p = history[j + end];
            vertices[cursor] = p.x;
            vertices[cursor + 1] = p.y;
            vertices[cursor + 2] = p.z;
            const fade = (1 - (j + end) / (steps - 1)) ** 1.5;
            alphas[cursor / 3] = fade;
            colors[cursor] = 0.88 + color.getX(i) * 0.12;
            colors[cursor + 1] = 0.9 + color.getY(i) * 0.1;
            colors[cursor + 2] = 0.94 + color.getZ(i) * 0.06;
            cursor += 3;
          }
      }
      geometry.setDrawRange(0, cursor / 3);
      geometry.attributes.position.needsUpdate = true;
      geometry.attributes.color.needsUpdate = true;
      geometry.attributes.trailAlpha.needsUpdate = true;
    },
    dispose() {
      lines.removeFromParent();
      geometry.dispose();
      material.dispose();
      haloMaterial.dispose();
    },
  };
}
