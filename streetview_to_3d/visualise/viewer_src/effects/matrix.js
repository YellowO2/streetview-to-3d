import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { Vector2 } from 'three';

// Screen-cell sampling follows the GPU ASCII approach documented by
// https://github.com/isladjan/ascii and https://github.com/emilwidlund/ASCII.
// Original binary glyph shader: no per-point displacement, atlas, or extra geometry.
export const MATRIX_CELL_SIZE = 11;

export function createMatrixPass() {
  return new ShaderPass({
    uniforms: {
      tDiffuse: { value: null },
      resolution: { value: new Vector2(1, 1) },
      cellSize: { value: MATRIX_CELL_SIZE },
      time: { value: 0 },
      strength: { value: 1 },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.);
      }
    `,
    fragmentShader: `
      uniform sampler2D tDiffuse;
      uniform vec2 resolution;
      uniform float cellSize, time, strength;
      varying vec2 vUv;
      float box(vec2 p, vec2 halfSize, vec2 aa) {
        vec2 edge = 1. - smoothstep(halfSize - aa, halfSize + aa, abs(p));
        return edge.x * edge.y;
      }
      void main() {
        vec2 size = vec2(cellSize * .67, cellSize);
        vec2 grid = vUv * resolution / size;
        vec2 cell = floor(grid);
        vec2 uv = (cell + .5) * size / resolution;
        vec2 offset = size / resolution * .28;
        vec3 colour = texture2D(tDiffuse, uv).rgb * .4;
        colour += texture2D(tDiffuse, uv + offset).rgb * .15;
        colour += texture2D(tDiffuse, uv - offset).rgb * .15;
        colour += texture2D(tDiffuse, uv + vec2(offset.x, -offset.y)).rgb * .15;
        colour += texture2D(tDiffuse, uv + vec2(-offset.x, offset.y)).rgb * .15;
        vec2 p = fract(grid) - .5;
        vec2 aa = .65 / size;
        float zero = box(p, vec2(.32, .39), aa)
          * (1. - box(p, vec2(.17, .24), aa));
        float one = box(p, vec2(.075, .39), aa);
        one = max(one, box(p - vec2(0., -.33), vec2(.27, .06), aa));
        one = max(one, box(p - vec2(-.10, .29), vec2(.17, .065), aa));
        float luma = dot(colour, vec3(.2126, .7152, .0722));
        float glyph = mix(one, zero, smoothstep(.13, .22, luma));
        float seed = fract(sin(cell.x * 127.1) * 43758.5453);
        // A soft highlight travels down a few columns; the text itself stays steady.
        float cycle = fract(cell.y / 65. + time * .075 + seed);
        float distanceToHead = min(cycle, 1. - cycle);
        float head = (1. - smoothstep(.01, .08, distanceToHead)) * step(.78, seed);
        vec3 ink = colour * (1.45 + head * .3);
        vec3 result = colour * .12 + ink * glyph;
        gl_FragColor = vec4(mix(texture2D(tDiffuse, vUv).rgb, result, strength), 1.);
      }
    `,
  });
}
