import * as THREE from 'three';
import { f } from '@viewer/util';

// Holes shot in the world (gun.js): a 3D grid around the scene's foot, sampled smoothly so holes are round.
// A shot uploads only the layers (one per CELL_M south) it touched.
export const CELL_M = 0.6,
  SIZE = [256, 128, 256]; // cells east, up, south
const FLOOR_M = 15; // grid starts this far below the foot

const data = new Uint8Array(SIZE[0] * SIZE[1] * SIZE[2]);
const field = new THREE.DataArrayTexture(data, ...SIZE);
field.format = THREE.RedFormat;
field.minFilter = field.magFilter = THREE.LinearFilter;
field.unpackAlignment = 1;
field.needsUpdate = true;
let whole = true; // next upload sends the whole texture
field.onUpdate = () => (whole = false);

export const shot = {
  shotField: { value: field },
  shotCorner: { value: new THREE.Vector3() },
  shotOn: { value: 0 }, // 0 until the first shot: skip the lookup
};

// GLSL: shotAway(p), whether world point p is shot away
export const SHOT = `
  uniform highp sampler2DArray shotField;
  uniform vec3 shotCorner;
  uniform float shotOn;
  bool shotAway(vec3 p) {
    if (shotOn < .5) return false;
    vec3 c = (p - shotCorner) / ${f(CELL_M)} - .5; // in cells, from the first cell's centre
    if (any(lessThan(c, vec3(0.))) || any(greaterThan(c, vec3(${SIZE.map((n) => f(n - 1)).join(', ')}))))
      return false;
    vec2 uv = (c.xy + .5) / vec2(${f(SIZE[0])}, ${f(SIZE[1])});
    float z = floor(c.z);
    return mix(texture(shotField, vec3(uv, z)).r, texture(shotField, vec3(uv, min(z + 1., ${f(SIZE[2] - 1)}))).r,
      c.z - z) > .5;
  }
`;

// centre the grid on a scene's foot and clear it
export function placeShots(foot) {
  shot.shotCorner.value.set(
    foot.x - (SIZE[0] * CELL_M) / 2,
    foot.y - FLOOR_M,
    foot.z - (SIZE[2] * CELL_M) / 2,
  );
  clearShots();
}
export function clearShots() {
  data.fill(0);
  field.clearLayerUpdates();
  whole = true;
  field.needsUpdate = true;
  shot.shotOn.value = 0;
}

const ab = new THREE.Vector3(),
  ap = new THREE.Vector3(),
  p = new THREE.Vector3();
// shoot away everything within radius of segment a-b; false if it is all off the grid
export function carve(a, b, radius) {
  const corner = shot.shotCorner.value;
  const lo = [0, 1, 2].map((i) =>
      Math.max(
        0,
        Math.floor(
          (Math.min(a.getComponent(i), b.getComponent(i)) - radius - corner.getComponent(i)) /
            CELL_M,
        ),
      ),
    ),
    hi = [0, 1, 2].map((i) =>
      Math.min(
        SIZE[i] - 1,
        Math.ceil(
          (Math.max(a.getComponent(i), b.getComponent(i)) + radius - corner.getComponent(i)) /
            CELL_M,
        ),
      ),
    );
  if (lo.some((l, i) => l > hi[i])) return false;
  ab.subVectors(b, a);
  const span = Math.max(ab.lengthSq(), 1e-12);
  for (let z = lo[2]; z <= hi[2]; z++) {
    let touched = false;
    for (let y = lo[1]; y <= hi[1]; y++)
      for (let x = lo[0]; x <= hi[0]; x++) {
        p.set(x + 0.5, y + 0.5, z + 0.5)
          .multiplyScalar(CELL_M)
          .add(corner);
        ap.subVectors(p, a);
        const t = Math.min(1, Math.max(0, ap.dot(ab) / span));
        if (ap.addScaledVector(ab, -t).lengthSq() > radius * radius) continue;
        data[x + SIZE[0] * (y + SIZE[1] * z)] = 255;
        touched = true;
      }
    if (touched && !whole) field.addLayerUpdate(z);
  }
  field.needsUpdate = true;
  shot.shotOn.value = 1;
  return true;
}
