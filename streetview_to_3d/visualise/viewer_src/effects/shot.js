import * as THREE from 'three';

// Where the world has been shot away (gun.js): one rule every point follows,
// as the demos' (demo.js) -- the scene's points and the map's (points.js),
// the buildings' strokes (blocks.js), the water's (water.js), the land's
// patches (land.js) -- a point inside it is gone, wherever it stands.
//
// Held as a grid of cells round the scene's foot (CELL_M apart, SIZE of them:
// east, up from FLOOR_M under it, south), each how much is shot away there,
// read smoothly between cells, so a hole's edge is round, not blocky. A shot
// carves only the layers it passes (one each CELL_M south), sent on alone.
export const CELL_M = 0.6,
  SIZE = [256, 128, 256];
const FLOOR_M = 15;

const data = new Uint8Array(SIZE[0] * SIZE[1] * SIZE[2]);
const field = new THREE.DataArrayTexture(data, ...SIZE);
field.format = THREE.RedFormat;
field.minFilter = field.magFilter = THREE.LinearFilter;
field.unpackAlignment = 1;
field.needsUpdate = true;
let whole = true; // all of it to be sent: not layer by layer
field.onUpdate = () => (whole = false);

// shared by every shader the shots cut through
export const shot = {
  shotField: { value: field },
  shotCorner: { value: new THREE.Vector3() },
  shotOn: { value: 0 }, // none shot yet: nothing read
};
const f = (x) => x.toFixed(4);

// GLSL, vertex or fragment: whether world point p is shot away
export const SHOT = `
  uniform highp sampler2DArray shotField;
  uniform vec3 shotCorner;
  uniform float shotOn;
  bool shotAway(vec3 p) {
    if (shotOn < .5) return false;
    vec3 c = (p - shotCorner) / ${f(CELL_M)} - .5; // in cells, from the first's middle
    if (any(lessThan(c, vec3(0.))) || any(greaterThan(c, vec3(${SIZE.map((n) => f(n - 1)).join(', ')}))))
      return false;
    vec2 uv = (c.xy + .5) / vec2(${f(SIZE[0])}, ${f(SIZE[1])});
    float z = floor(c.z);
    return mix(texture(shotField, vec3(uv, z)).r, texture(shotField, vec3(uv, min(z + 1., ${f(SIZE[2] - 1)}))).r,
      c.z - z) > .5;
  }
`;

// the grid round a scene, its foot (the middle of its points, at their bottom); nothing shot
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
// shoot away everything within radius of the path a to b; false once it is all off the grid
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
