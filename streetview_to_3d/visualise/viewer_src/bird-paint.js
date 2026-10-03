import * as THREE from 'three';
import { haze, HAZED } from '@viewer/effects/haze';
import { DISC, f, v3 } from '@viewer/effects/util';

// Paint for the bird (bird.js) and its trail (bird-plume.js): faint white soft dabs, hazed far off.
// Trail dabs carry the time they were laid (born) and fade and drift as they age.
export const DAB = 0.1; // dab size (m), as DA3 points (app.js PLACED_POINT_M)
const LIFE_S = 1.6, // trail dab lifetime
  WHITE = [0.97, 0.97, 0.96],
  ALPHA = 0.2, // bird dab opacity
  TRAIL = 0.035, // newest trail dab opacity: many overlap
  DRIFT = [0.09, 0.02, -0.035]; // m/s: trail drift, along the clouds' wind (clouds.js WIND), rising

export function birdPaint({ trail = false } = {}) {
  const material = new THREE.PointsMaterial({ size: DAB, transparent: true, depthWrite: false });
  const uniforms = { haze, time: { value: 0 } };
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader =
      `${trail ? 'attribute float born; uniform float time;' : ''}
      varying vec3 paintWorld;
      varying float paintAlpha;\n` +
      shader.vertexShader
        .replace(
          '#include <begin_vertex>',
          trail
            ? `#include <begin_vertex>
          float k = (time - born) / ${f(LIFE_S)};
          paintAlpha = born < 0. || k >= 1. ? 0.
            : ${f(TRAIL)} * smoothstep(0., .08, k) * (1. - k) * (1. - k);
          float age = time - born;
          transformed += ${v3(DRIFT)} * age;`
            : `#include <begin_vertex>\n paintAlpha = ${f(ALPHA)};`,
        )
        .replace(
          '#include <fog_vertex>',
          `#include <fog_vertex>
          paintWorld = (modelMatrix * vec4(transformed, 1.)).xyz;
          if (paintAlpha <= 0.) { gl_Position = vec4(2., 2., 2., 1.); gl_PointSize = 0.; }`,
        );
    shader.fragmentShader =
      `uniform float haze;
      varying vec3 paintWorld;
      varying float paintAlpha;
      ${HAZED}\n` +
      shader.fragmentShader.replace(
        '#include <opaque_fragment>',
        `outgoingLight = hazed(${v3(WHITE)}, length(paintWorld - cameraPosition), haze);
          ${DISC}
          diffuseColor.a = paintAlpha * (1. - smoothstep(.15, 1., r));
          #include <opaque_fragment>`,
      );
  };
  material.customProgramCacheKey = () => `bird-paint-v4:${trail}`;
  return { material, uniforms };
}

// a THREE.Points of bird paint with stable per-dab seeds for the style's float
export function birdDabs(geometry, material, count) {
  geometry.setAttribute(
    'styleSeed',
    new THREE.Float32BufferAttribute(
      Array.from({ length: count }, (_, i) => ((i * 73) % 997) / 997),
      1,
    ),
  );
  const points = new THREE.Points(geometry, material);
  points.userData.styleAnimated = true;
  points.frustumCulled = false;
  return points;
}
