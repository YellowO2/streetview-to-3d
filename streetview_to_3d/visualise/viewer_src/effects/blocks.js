import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// The far buildings (postprocess/buildings.py's solid, blocks.ply) come as
// triangles -- a small file -- and are drawn as points, like everything
// else, so they take the same paint: scattered over each triangle as far
// apart as the terrain's points there (spacing), at random, so no grid
// shows. Each wall vertex carries its place on its wall (facade: metres
// along, metres up, -1 on a roof), so the points take floors of windows
// from it -- each way only where the points are close enough to draw it,
// else its average (a floor's band of windows, or an even darker wall).
// Behind every wall, and under every roof, sparser and darker layers
// (BEHIND_M): through the gaps between points, more building, not sky.
const FLOOR_M = 3.2,
  BAY_M = 3.0,
  WINDOW_U = [0.3, 0.7], // of a bay
  WINDOW_V = [0.3, 0.8], // of a floor
  WINDOW = [0.45, [0.03, 0.05, 0.08]], // a window: the wall this dark, plus this
  BEHIND_M = [0.5, 1.5],
  BEHIND_GAP = 2, // times sparser
  BEHIND_SHADE = 0.7;
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

// Deterministic, so a scene looks the same every time it opens.
function random(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// How much of a window at a place (metres) on a wall points `gap` apart
// show, one way: the window itself where they are close enough, else how
// much of the wall is window.
function across(x, [lo, hi], size, gap) {
  if (gap > size / 4) return hi - lo;
  const f = x / size - Math.floor(x / size);
  return f >= lo && f <= hi ? 1 : 0;
}

// Points over the triangles (parseBlocks), spacing(x, z) apart: a
// THREE.Points with position and color.
export function scatterBlocks(geometry, spacing) {
  const p = geometry.getAttribute('position'),
    c = geometry.getAttribute('color'),
    f = geometry.getAttribute('facade'),
    index = geometry.index.array;
  const rand = random(1),
    pos = [],
    col = [];
  const A = new THREE.Vector3(),
    B = new THREE.Vector3(),
    C = new THREE.Vector3(),
    e1 = new THREE.Vector3(),
    e2 = new THREE.Vector3(),
    n = new THREE.Vector3();
  const lerp = (attr, k, a, b, cc, u, v) =>
    attr.getComponent(a, k) * (1 - u - v) +
    attr.getComponent(b, k) * u +
    attr.getComponent(cc, k) * v;
  for (let t = 0; t < index.length; t += 3) {
    const a = index[t],
      b = index[t + 1],
      cc = index[t + 2];
    A.fromBufferAttribute(p, a);
    B.fromBufferAttribute(p, b);
    C.fromBufferAttribute(p, cc);
    e1.subVectors(B, A);
    e2.subVectors(C, A);
    n.crossVectors(e1, e2);
    const area = n.length() / 2;
    if (area < 1e-6) continue;
    const gap = spacing((A.x + B.x + C.x) / 3, (A.z + B.z + C.z) / 3);
    const wall = f.getY(a) >= 0 && f.getY(b) >= 0 && f.getY(cc) >= 0;
    // behind a wall: into the building (the outline runs counter-clockwise
    // seen from above, so its inside is to the left of a -> b); under a roof
    const along = Math.hypot(e1.x, e1.z) > 1e-6 ? e1 : e2;
    const inward = wall
      ? new THREE.Vector3(along.z, 0, -along.x).normalize()
      : n.y > 0
        ? new THREE.Vector3(0, -1, 0)
        : null;
    const layers = [[0, gap, 1]];
    if (inward) for (const d of BEHIND_M) layers.push([d, gap * BEHIND_GAP, BEHIND_SHADE]);
    for (const [depth, step, shade] of layers) {
      const count = Math.floor(area / (step * step) + rand());
      for (let k = 0; k < count; k++) {
        let u = rand(),
          v = rand();
        if (u + v > 1) {
          u = 1 - u;
          v = 1 - v;
        }
        let r = lerp(c, 0, a, b, cc, u, v),
          g = lerp(c, 1, a, b, cc, u, v),
          bl = lerp(c, 2, a, b, cc, u, v);
        if (wall && !depth) {
          const w =
            across(lerp(f, 0, a, b, cc, u, v), WINDOW_U, BAY_M, gap) *
            across(lerp(f, 1, a, b, cc, u, v), WINDOW_V, FLOOR_M, gap);
          const [dark, tint] = WINDOW;
          r += (r * dark + tint[0] - r) * w * 0.85;
          g += (g * dark + tint[1] - g) * w * 0.85;
          bl += (bl * dark + tint[2] - bl) * w * 0.85;
        }
        const x = lerp(p, 0, a, b, cc, u, v),
          y = lerp(p, 1, a, b, cc, u, v),
          z = lerp(p, 2, a, b, cc, u, v);
        pos.push(
          x + (inward ? inward.x * depth : 0),
          y + (inward ? inward.y * depth : 0),
          z + (inward ? inward.z * depth : 0),
        );
        col.push(r * shade, g * shade, bl * shade);
      }
    }
  }
  const points = new THREE.BufferGeometry();
  points.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  points.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  points.computeBoundingBox();
  points.computeBoundingSphere();
  return new THREE.Points(points, new THREE.PointsMaterial({ vertexColors: true }));
}
