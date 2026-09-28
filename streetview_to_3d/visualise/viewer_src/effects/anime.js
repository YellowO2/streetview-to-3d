// Image-only pass: reusable after a point-cloud or Gaussian-splat colour render.
// Input is linear colour; OutputPass performs the final display conversion.
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { Vector2 } from 'three';

export function createAnimePass() {
  return new ShaderPass({
    uniforms: {
      tDiffuse: { value: null },
      tDepth: { value: null },
      useDepth: { value: 1 },
      texel: { value: new Vector2(1, 1) },
      strength: { value: 0.75 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D tDiffuse, tDepth;
      uniform vec2 texel;
      uniform float strength, useDepth;
      varying vec2 vUv;
      float styleLuma(vec3 c) { return dot(c, vec3(.2126, .7152, .0722)); }
      void main() {
        vec3 source = texture2D(tDiffuse, vUv).rgb;
        // Leave the sky untouched: value bands and warm grading make blue skies muddy.
        if (useDepth > .5 && texture2D(tDepth, vUv).r >= .999999) {
          gl_FragColor = vec4(source, 1.0);
          return;
        }
        vec3 colour = source;
        float weights = 1.0;
        // A small edge-aware wash reduces capture noise without blurring silhouettes.
        for (int y = -1; y <= 1; y++) {
          for (int x = -1; x <= 1; x++) {
            vec3 neighbour = texture2D(tDiffuse, vUv + vec2(float(x), float(y)) * texel * 2.0).rgb;
            float weight = exp(-length(neighbour - source) * 16.0);
            colour += neighbour * weight;
            weights += weight;
          }
        }
        colour /= weights;
        float light = styleLuma(colour);
        // Soft value bands, keeping the captured hue instead of using a scene-specific palette.
        float band = floor(light * 7.0 + .5) / 7.0;
        colour *= mix(light, band, .3) / max(light, .025);
        colour = mix(vec3(styleLuma(colour)), colour, 1.10);
        float shade = 1.0 - smoothstep(.05, .6, light);
        colour += vec3(.015, .032, .05) * shade;
        colour = mix(colour, vec3(.94, .90, .76), .075 * smoothstep(.2, .8, light));
        colour = colour * .94 + vec3(.025, .029, .022);
        // Restrained colour edges; strong point-by-point ink would amplify holes.
        float left = styleLuma(texture2D(tDiffuse, vUv - vec2(texel.x * 2.0, 0.)).rgb);
        float right = styleLuma(texture2D(tDiffuse, vUv + vec2(texel.x * 2.0, 0.)).rgb);
        float edge = smoothstep(.10, .4, abs(left - right));
        colour *= 1.0 - edge * .075;
        gl_FragColor = vec4(mix(source, clamp(colour, 0.0, 1.0), strength), 1.0);
      }
    `,
  });
}
