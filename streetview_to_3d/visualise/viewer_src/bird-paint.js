import * as THREE from 'three';
import { haze, HAZED } from '@viewer/effects/haze';

// The bird's paint (bird.js; its trail, bird-plume.js): points as the world's
// are -- the DA3 points' size, styled as they are (effects/points.js) --
// faint and white as a breath of wind: each dab soft to its edge, as a
// cloud's, and ALPHA see-through, whichever way it turns; hazed far off as
// the land is.
//
// A trail's dab also its time laid (born): its trail one smooth streak of
// them, overlapping, all fading alike as they age, drifting a little on the
// wind.
export const DAB = 0.1; // a dab this big: as the DA3 points (app.js PLACED_POINT_M)
export const LIFE_S = 1.6; // a trail's dab gone after this long
const WHITE = [0.97, 0.97, 0.96],
  ALPHA = 0.2, // a dab of the bird
  TRAIL = 0.035, // a trail's dab, at its newest: many overlap in its streak
  DRIFT = [0.09, 0.02, -0.035]; // m/s: a trail carried on the clouds' wind (clouds.js WIND), rising

const f = (x) => x.toFixed(4);
const v3 = (v) => `vec3(${v.map(f).join(', ')})`;

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
          // soft to its edge
          float r = dot(gl_PointCoord * 2. - 1., gl_PointCoord * 2. - 1.);
          if (r > 1.) discard;
          diffuseColor.a = paintAlpha * (1. - smoothstep(.15, 1., r));
          #include <opaque_fragment>`,
      );
  };
  material.customProgramCacheKey = () => `bird-paint-v4:${trail}`;
  return { material, uniforms };
}

// a Points of the bird's paint, styled as the world's points are
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
