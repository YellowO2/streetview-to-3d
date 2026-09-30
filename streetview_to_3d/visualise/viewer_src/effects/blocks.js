// The far buildings (postprocess/buildings.py's solid, blocks.ply), drawn
// as what they are, triangles: past where points meet the scene there is
// nothing for points to do. Each wall vertex carries its place on its wall
// (facade: metres along, metres up from its foot, -1e4 on a roof), so the shader draws
// floors of windows from it (as postprocess/buildings.windows does on the
// near points), painted rather than printed: soft edges, each pane a little
// lighter or darker, a pale sill under it, the wall darker at its foot.
// Where a window is under a couple of pixels, the wall's average instead (a
// floor's band, then an evenly darker wall), so they do not shimmer far off.
import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

export const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // of a bay
  WINDOW_V = [0.3, 0.8], // of a floor
  WINDOW = [0.45, [0.05, 0.07, 0.1]], // a window: the wall this dark, plus this
  WINDOW_MIX = 0.6, // a window over its wall this much: suggested, not printed
  PANE = 0.25, // each pane this much lighter or darker, at most
  SILL = [0.06, 0.12], // a sill this high (of a floor) under a window, this much lighter
  FOOT = [0.8, 3]; // a wall this dark at its foot, as it was by this high
const loader = new PLYLoader();
loader.setCustomPropertyNameMapping({ facade: ['facade_u', 'facade_v'] });

// blocks.ply (a buffer) as triangles in the viewer's frame (flip: y up, z south).
export function parseBlocks(buffer, flip) {
  const geometry = loader.parse(buffer);
  if (!geometry.getAttribute('position')?.count || !geometry.index) {
    geometry.dispose();
    throw Error('blocks.ply has no triangles.');
  }
  geometry.applyMatrix4(flip);
  return geometry;
}

const f = (x) => x.toFixed(3);
const vec3 = (v) => `vec3(${v.map(f).join(',')})`;
const area = (WINDOW_U[1] - WINDOW_U[0]) * (WINDOW_V[1] - WINDOW_V[0]);

// The triangles (parseBlocks) as one solid mesh, windows drawn on its walls.
export function blocksMesh(geometry) {
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', 'attribute vec2 facade;\nvarying vec2 vFacade;\n#include <common>')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvFacade = facade;');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `varying vec2 vFacade;
        #include <common>
        // 1 inside [a, b], 0 outside, over a pixel's width w
        float band(float x, float a, float b, float w) {
          return smoothstep(a - w, a + w, x) * (1.0 - smoothstep(b - w, b + w, x));
        }`,
      )
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        if (vFacade.y > -1000.0) {
          vec2 cell = vFacade / vec2(${f(BAY_M)}, ${f(FLOOR_M)});
          vec2 q = fract(cell), px = fwidth(cell);
          float far = clamp(max(px.x, px.y) * 3.0 - 0.5, 0.0, 1.0);
          float up = step(0.5, vFacade.y);           // no window in the ground floor's first half metre
          float inside = band(q.x, ${f(WINDOW_U[0])}, ${f(WINDOW_U[1])}, px.x)
            * band(q.y, ${f(WINDOW_V[0])}, ${f(WINDOW_V[1])}, px.y);
          float sill = band(q.x, ${f(WINDOW_U[0] - 0.03)}, ${f(WINDOW_U[1] + 0.03)}, px.x)
            * band(q.y, ${f(WINDOW_V[0] - SILL[0])}, ${f(WINDOW_V[0])}, px.y) * (1.0 - far);
          float pane = fract(sin(dot(floor(cell), vec2(12.9898, 78.233))) * 43758.5453);
          vec3 wall = diffuseColor.rgb;
          vec3 glass = wall * ${f(WINDOW[0])} + ${vec3(WINDOW[1])};
          glass *= 1.0 + (pane - 0.5) * ${f(2 * PANE)} * (1.0 - far);
          float w = mix(inside, ${f(area)}, far) * up;
          diffuseColor.rgb = mix(wall * (1.0 + ${f(SILL[1])} * sill * up), glass, w * ${f(WINDOW_MIX)});
          diffuseColor.rgb *= mix(${f(FOOT[0])}, 1.0, smoothstep(0.0, ${f(FOOT[1])}, vFacade.y));
        }`,
      );
  };
  return new THREE.Mesh(geometry, material);
}
