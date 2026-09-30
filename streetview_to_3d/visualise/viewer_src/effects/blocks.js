import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// The far buildings (postprocess/buildings.py's solid, blocks.ply), drawn
// as what they are, triangles: past where points meet the scene there is
// nothing for points to do. Each wall vertex carries its place on its wall
// (facade: metres along, metres up, -1 on a roof), so the shader draws
// floors of windows from it (as postprocess/buildings.windows does on the
// near points) -- where a window is under a couple of pixels, the wall's
// average instead (a floor's band, then an evenly darker wall), so they do
// not shimmer far off.
export const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // of a bay
  WINDOW_V = [0.3, 0.8], // of a floor
  WINDOW = [0.45, [0.03, 0.05, 0.08]]; // a window: the wall this dark, plus this
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

// The triangles (parseBlocks) as one solid mesh, windows drawn on its walls.
export function blocksMesh(geometry) {
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', 'attribute vec2 facade;\nvarying vec2 vFacade;\n#include <common>')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvFacade = facade;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', 'varying vec2 vFacade;\n#include <common>')
      .replace(
        '#include <color_fragment>',
        `#include <color_fragment>
        if (vFacade.y > 0.5) {
          vec2 cell = vFacade / vec2(${f(BAY_M)}, ${f(FLOOR_M)});
          vec2 q = fract(cell);
          float inside = step(${f(WINDOW_U[0])}, q.x) * step(q.x, ${f(WINDOW_U[1])})
            * step(${f(WINDOW_V[0])}, q.y) * step(q.y, ${f(WINDOW_V[1])});
          vec2 px = fwidth(cell);
          float far = clamp(max(px.x, px.y) * 3.0 - 0.5, 0.0, 1.0);
          float w = mix(inside, ${f((WINDOW_U[1] - WINDOW_U[0]) * (WINDOW_V[1] - WINDOW_V[0]))}, far);
          diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * ${f(WINDOW[0])} + ${vec3(WINDOW[1])}, w * 0.85);
        }`,
      );
  };
  return new THREE.Mesh(geometry, material);
}
