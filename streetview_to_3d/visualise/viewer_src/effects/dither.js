// Screen-space ordered dithering in display colour, then back to linear for OutputPass.
// No point geometry changes; one image pass and a fixed eight-colour palette.
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { Vector2 } from 'three';
// Cap the pattern at 480 cells along the longest viewport edge. Work in CSS
// pixels first so Retina screens do not silently make the grain denser.
export function ditherCellSize(width, height, pixelRatio, requestedSize) {
  return Math.max(3, requestedSize, Math.max(width, height) / 480) * pixelRatio;
}
export function createDitherPass() {
  return new ShaderPass({
    uniforms: {
      tDiffuse: { value: null },
      tDepth: { value: null },
      useDepth: { value: 1 },
      fogAmount: { value: 1 },
      near: { value: 0.01 },
      far: { value: 1000 },
      fogDistance: { value: 50 },
      resolution: { value: new Vector2(1, 1) },
      pixelSize: { value: 2 },
      strength: { value: 1 },
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
      uniform vec2 resolution;
      uniform float pixelSize, strength, near, far, fogDistance, useDepth, fogAmount;
      varying vec2 vUv;
      float bayer2(vec2 p) {
        p = mod(floor(p), 2.0);
        return 2.0 * p.x + 3.0 * p.y - 4.0 * p.x * p.y;
      }
      vec3 toDisplay(vec3 c) {
        return mix(c * 12.92, 1.055 * pow(max(c, 0.0), vec3(1.0 / 2.4)) - .055, step(vec3(.0031308), c));
      }
      vec3 toLinear(vec3 c) {
        return mix(c / 12.92, pow((c + .055) / 1.055, vec3(2.4)), step(vec3(.04045), c));
      }
      void main() {
        vec2 cell = floor(vUv * resolution / pixelSize);
        vec2 uv = (cell + .5) * pixelSize / resolution;
        vec3 source = texture2D(tDiffuse, vUv).rgb;
        vec3 colour = toDisplay(texture2D(tDiffuse, uv).rgb);
        float threshold = (4.0 * bayer2(cell) + bayer2(floor(cell / 2.0)) + .5) / 16.0 - .5;
        float depth = texture2D(tDepth, uv).r;
        if (useDepth > .5 && depth >= .999999) {
          gl_FragColor = vec4(source, 1.0);
          return;
        }
        {
          float distance = near * far / max(far - depth * (far - near), .00001);
          float fog = fogAmount * useDepth * smoothstep(fogDistance * 2.0, fogDistance * 10.0, distance);
          float luma = dot(colour, vec3(.2126, .7152, .0722));
          colour = mix(vec3(luma), colour, .38) * .95 + .025;
          colour = mix(colour, vec3(.55, .57, .57), fog * .40);
          float vignette = smoothstep(.3, 1.4, length(vUv * 2.0 - 1.0));
          colour *= 1.0 - vignette * .14;
        }
        colour += threshold * .16;
        vec3 palette[8];
        palette[0] = vec3(.12, .15, .17);
        palette[1] = vec3(.23, .27, .28);
        palette[2] = vec3(.35, .39, .39);
        palette[3] = vec3(.46, .49, .48);
        palette[4] = vec3(.55, .57, .57);
        palette[5] = vec3(.66, .68, .65);
        palette[6] = vec3(.78, .78, .72);
        palette[7] = vec3(.89, .89, .83);
        vec3 selected = palette[0];
        float best = 10.0;
        for (int i = 0; i < 8; i++) {
          vec3 delta = colour - palette[i];
          float distance = dot(delta, delta);
          if (distance < best) { best = distance; selected = palette[i]; }
        }
        gl_FragColor = vec4(mix(source, toLinear(selected), strength), 1.0);
      }
    `,
  });
}
