import * as THREE from 'three';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

// Sparse world-space wing histories leave a continuous wake behind flight.
export function createBirdTrails(bird, clouds, restPositions = []) {
  const samples = [];
  clouds.forEach((cloud, index) => {
    for (let i = 0; i < cloud.geometry.attributes.position.count; i += 70)
      samples.push({ cloud, i, rest: restPositions[index], history: [] });
  });
  const steps = 48,
    vertices = new Float32Array(samples.length * (steps - 1) * 6),
    colors = new Float32Array(vertices.length),
    alphas = new Float32Array(vertices.length / 3);
  const ribbon = new LineSegmentsGeometry().setPositions(vertices).setColors(colors);
  const fadeBuffer = new THREE.InstancedInterleavedBuffer(alphas, 2).setUsage(
    THREE.DynamicDrawUsage,
  );
  ribbon.setAttribute('fadeStart', new THREE.InterleavedBufferAttribute(fadeBuffer, 1, 0));
  ribbon.setAttribute('fadeEnd', new THREE.InterleavedBufferAttribute(fadeBuffer, 1, 1));
  ribbon.instanceCount = 0;
  const material = new LineMaterial({
    vertexColors: true,
    linewidth: 1.8,
    transparent: true,
    opacity: 0.18,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader =
      'attribute float fadeStart, fadeEnd; varying float wakeFade;\n' +
      shader.vertexShader
        .replace(
          'void main() {',
          'void main() {\n wakeFade = position.y < .5 ? fadeStart : fadeEnd;',
        )
        .replace('offset *= linewidth;', 'offset *= linewidth * mix(.12, 1., pow(wakeFade, .55));');
    shader.fragmentShader =
      'varying float wakeFade;\n' +
      shader.fragmentShader.replace(
        'vec4( diffuseColor.rgb, alpha )',
        'vec4( diffuseColor.rgb, alpha * wakeFade )',
      );
  };
  material.customProgramCacheKey = () => 'bird-wake-taper-v1';
  const lines = new LineSegments2(ribbon, material);
  lines.frustumCulled = false;
  lines.matrixAutoUpdate = false;
  // The glow uses the same continuous ribbon, softened across its width.
  const haloMaterial = material.clone();
  haloMaterial.linewidth = 6;
  haloMaterial.opacity = 0.07;
  haloMaterial.onBeforeCompile = (shader) => {
    material.onBeforeCompile(shader);
    shader.fragmentShader = shader.fragmentShader.replace(
      'alpha * wakeFade',
      'alpha * wakeFade * exp(-vUv.x * vUv.x * 4.)',
    );
  };
  haloMaterial.customProgramCacheKey = () => 'bird-wake-ribbon-glow-v1';
  const halo = new LineSegments2(ribbon, haloMaterial);
  halo.frustumCulled = false;
  lines.add(halo);
  bird.add(lines);
  let elapsed = 0;
  return {
    reset() {
      samples.forEach((s) => (s.history.length = 0));
      ribbon.instanceCount = 0;
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
        for (let j = 0; j < history.length - 1; j++)
          for (let end = 0; end < 2; end++) {
            const p = history[j + end];
            vertices[cursor] = p.x;
            vertices[cursor + 1] = p.y;
            vertices[cursor + 2] = p.z;
            const fade = (1 - (j + end) / (steps - 1)) ** 1.5;
            alphas[cursor / 3] = fade;
            colors[cursor] = 1;
            colors[cursor + 1] = 1;
            colors[cursor + 2] = 1;
            cursor += 3;
          }
      }
      ribbon.instanceCount = cursor / 6;
      ribbon.attributes.instanceStart.data.needsUpdate = true;
      ribbon.attributes.instanceColorStart.data.needsUpdate = true;
      fadeBuffer.needsUpdate = true;
    },
    dispose() {
      lines.removeFromParent();
      ribbon.dispose();
      material.dispose();
      haloMaterial.dispose();
    },
  };
}
