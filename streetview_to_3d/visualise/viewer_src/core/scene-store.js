import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import {
  validateScene,
  placementMode,
  relativePath,
  scenePieces,
  gpsPlacement,
  SURROUNDINGS,
  LAND,
  ROADS,
  WATER,
  GOOGLE,
  BLOCKS,
  LIFE,
} from '@viewer/core/scene-format';
import { waterSurfaces } from '@viewer/world/water';
import { trafficPoints } from '@viewer/life/traffic';
import { birdPoints } from '@viewer/life/flock';
import { boatPoints } from '@viewer/life/boats';
import { duckPoints } from '@viewer/life/ducks';
import { catPoints } from '@viewer/life/cats';
import { landPoints } from '@viewer/world/land';
import { blockPoints, buildingPoints, RATIO } from '@viewer/world/blocks';
import { parseSurface } from '@viewer/world/scatter';
const flip = new THREE.Matrix4().makeScale(1, -1, -1);
const identity = new THREE.Matrix4();
function matrixRows(m) {
  return Array.from({ length: 4 }, (_, r) =>
    Array.from({ length: 4 }, (_, c) => m.elements[c * 4 + r]),
  );
}
function exportPlacement(original, correction) {
  return matrixRows(
    flip
      .clone()
      .multiply(correction)
      .multiply(flip)
      .multiply(new THREE.Matrix4().set(...original.flat())),
  );
}
export function dispose(group) {
  group?.traverse((o) => {
    if (o.isPoints) {
      o.geometry.dispose();
      o.material.dispose();
    } else if (o.userData.splat) o.dispose();
    else if (o.isMesh) {
      o.geometry.dispose();
      o.material.dispose();
      if (o.isReflector) o.dispose(); // its mirror render target
    }
  });
}
const loader = new PLYLoader();
loader.setCustomPropertyNameMapping({
  gap: ['gap'],
  kind: ['kind'],
  near: ['near'],
  sway: ['sway'], // per-point wind sway (fill: trees)
});
// land.ply triangles as the land's dabs (world/land.js); gapOf(x, z): point spacing there
function parseLand(buffer, gapOf) {
  const geometry = loader.parse(buffer);
  if (!geometry.getAttribute('position')?.count || !geometry.index) {
    geometry.dispose();
    throw Error(`${LAND}.ply has no triangles.`);
  }
  geometry.applyMatrix4(flip);
  const land = landPoints(geometry, gapOf);
  land.userData.surroundings = LAND;
  return land;
}
// blocks.ply, the far buildings DA3 never reaches, as dabs (world/blocks.js)
function parseBlocks(buffer) {
  const blocks = blockPoints(parseSurface(buffer, flip, `${BLOCKS}.ply`));
  blocks.userData.surroundings = BLOCKS;
  return blocks;
}
// roads.ply triangles as dabs, as the land's are
function parseRoads(buffer, gapOf) {
  const roads = landPoints(parseSurface(buffer, flip, `${ROADS}.ply`), gapOf);
  roads.userData.surroundings = ROADS;
  return roads;
}
export function parsePoints(buffer, transform) {
  const geometry = loader.parse(buffer);
  try {
    const p = geometry.getAttribute('position');
    if (!p?.count || !p.array.every(Number.isFinite)) throw Error('PLY has no valid points.');
    if (transform) geometry.applyMatrix4(new THREE.Matrix4().set(...transform.flat()));
    geometry.applyMatrix4(flip);
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    if (!Number.isFinite(geometry.boundingSphere.radius)) throw Error('Invalid point bounds.');
    const color = !!geometry.getAttribute('color');
    return new THREE.Points(
      geometry,
      new THREE.PointsMaterial({ vertexColors: color, color: color ? 0xffffff : 0xa9cbe3 }),
    );
  } catch (e) {
    geometry.dispose();
    throw e;
  }
}
// The one rule for a map point's size: DA3's points are the smallest in the world; a map point is
// as big on them and bigger the further from them, and its spacing follows (RATIO; postprocess
// seams.size_at, spacing_at).
const DA3_M = 0.17, // DA3's points as drawn in the default look
  GROW = 0.0154; // a map point's size grows this much per metre from DA3's points
const spacingAt = (d) => (DA3_M + GROW * d) / RATIO;
const REACH_M = 1200; // the scene's distance is known this far from the origin

// (x, z) -> metres from DA3's points: from the 1 m squares holding at least 3 of them (as
// postprocess terrain's edge), by a two-pass chamfer over the squares within REACH_M
function sceneDistance(group) {
  const n = 2 * REACH_M,
    count = new Uint8Array(n * n);
  group.traverse((o) => {
    if (o.userData.nodeIndex == null) return;
    const p = o.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      const x = Math.floor(p.getX(i)) + REACH_M,
        z = Math.floor(p.getZ(i)) + REACH_M;
      if (x >= 0 && x < n && z >= 0 && z < n && count[z * n + x] < 3) count[z * n + x]++;
    }
  });
  const least = count.includes(3) ? 3 : 1; // a scene of a few points: any square holding one
  const far = Float32Array.from(count, (c) => (c >= least ? 0 : Infinity));
  const pass = (from, to, step) => {
    for (let z = from; z !== to; z += step)
      for (let x = from; x !== to; x += step) {
        const i = z * n + x;
        let d = far[i];
        if (x - step >= 0 && x - step < n) d = Math.min(d, far[i - step] + 1);
        if (z - step >= 0 && z - step < n) {
          d = Math.min(d, far[i - step * n] + 1);
          if (x - step >= 0 && x - step < n) d = Math.min(d, far[i - step * n - step] + Math.SQRT2);
          if (x + step >= 0 && x + step < n) d = Math.min(d, far[i - step * n + step] + Math.SQRT2);
        }
        far[i] = d;
      }
  };
  pass(0, n, 1);
  pass(n - 1, -1, -1);
  const at = (v) => Math.min(Math.max(Math.floor(v) + REACH_M, 0), n - 1);
  return (x, z) => far[at(z) * n + at(x)];
}
// each placed node's camera position in the viewer frame
function cameraPoints(data) {
  return data.nodes
    .filter((n) => n.transform && n.position)
    .map((n) =>
      new THREE.Vector3(...n.position)
        .applyMatrix4(new THREE.Matrix4().set(...n.transform.flat()))
        .applyMatrix4(flip)
        .toArray(),
    );
}
// A Gaussian splat (.spz); Spark is imported only when one is opened.
async function parseSplat(buffer) {
  // Spark decodes in a worker, which Chrome won't start for a file:// page
  if (location.protocol === 'file:')
    throw Error(
      'Splats need the viewer served over http. In its folder run "python3 -m http.server", then open http://localhost:8000/viewer.html.',
    );
  const { SplatMesh } = await import('@sparkjsdev/spark');
  const mesh = new SplatMesh({ fileBytes: buffer });
  await mesh.initialized;
  if (!mesh.numSplats) {
    mesh.dispose();
    throw Error('Splat file has no splats.');
  }
  mesh.quaternion.set(1, 0, 0, 0); // Y-down to Y-up, as `flip` does for points
  mesh.userData.splat = true;
  return mesh;
}
async function readBuffer(source) {
  if (typeof source !== 'string') return source.arrayBuffer();
  const response = await fetch(source);
  if (!response.ok) throw Error(`Download failed (${response.status}).`);
  return response.arrayBuffer();
}
// Loads a scene off-screen; failures and superseded loads never touch the installed scene.
// The surroundings are Google's 3D Tiles (world/google-tiles.js), not the map-built land, roads
// and buildings, where the scene names its own (scene.json's google) or a Map Tiles API key is
// given (google: streamed, unless a google.ply lies beside the scene).
export async function loadAsset(
  source,
  resolve,
  progress,
  cancelled,
  { splat = false, google = null } = {},
) {
  const group = new THREE.Group();
  let data = null,
    placement = null;
  try {
    if (splat) group.add(await parseSplat(await readBuffer(source)));
    else if (!resolve) group.add(parsePoints(await readBuffer(source)));
    else {
      data = JSON.parse(new TextDecoder().decode(await readBuffer(source)));
      validateScene(data);
      placement = placementMode(data);
      const assets = data.nodes.flatMap((n, i) =>
        n.ply ? [{ i, source: resolve(relativePath(n.ply)) }] : [],
      );
      for (let j = 0; j < assets.length; j++) {
        if (cancelled()) {
          dispose(group);
          return null;
        }
        progress(`Loading point cloud ${j + 1} of ${assets.length}…`);
        const buffer = await readBuffer(assets[j].source);
        if (cancelled()) {
          dispose(group);
          return null;
        }
        const points = parsePoints(buffer, data.nodes[assets[j].i].transform);
        points.userData.nodeIndex = assets[j].i;
        group.add(points);
        await new Promise((r) => setTimeout(r, 0));
      }
      // surroundings are in world coordinates: placed scenes only
      const googled = placement === 'world' && (google || data[GOOGLE]);
      const mapped = placement === 'world' && !googled; // the map-built surroundings
      // a map point's spacing where its ply gives none, by its distance from DA3's points
      const far = placement === 'world' ? sceneDistance(group) : null;
      const gapOf = (x, z) => spacingAt(far(x, z));
      for (const key of mapped ? SURROUNDINGS : []) {
        if (!data[key]) continue;
        progress(`Loading the ${key}…`);
        const buffer = await readBuffer(resolve(relativePath(data[key])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        const built = buildingPoints(parsePoints(buffer).geometry, { gapAt: gapOf });
        built.userData.surroundings = key;
        group.add(built);
      }
      if (mapped && data[LAND]) {
        progress('Loading the land…');
        const buffer = await readBuffer(resolve(relativePath(data[LAND])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        group.add(parseLand(buffer, gapOf));
      }
      for (const key of mapped ? [ROADS, BLOCKS] : []) {
        if (!data[key]) continue;
        progress(`Loading the ${key === BLOCKS ? 'far buildings' : key}…`);
        const buffer = await readBuffer(resolve(relativePath(data[key])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        if (key === BLOCKS) group.add(parseBlocks(buffer));
        else group.add(parseRoads(buffer, gapOf));
        await new Promise((r) => setTimeout(r, 0));
      }
      if (googled) {
        progress('Loading the surroundings…');
        // the scene's own, else one saved beside it (?google-save), else asked of Google
        let buffer = null;
        try {
          buffer = await readBuffer(resolve(relativePath(data[GOOGLE] || 'google.ply')));
        } catch (e) {
          if (!google) throw e;
          const { googleTiles } = await import('@viewer/world/google-tiles');
          buffer = await googleTiles(google, data.center, cameraPoints(data), gapOf, progress);
        }
        if (cancelled()) {
          dispose(group);
          return null;
        }
        // dabs like the map's buildings, their photo colours left as lit
        const built = buildingPoints(parsePoints(buffer).geometry, { baked: true });
        built.userData.surroundings = GOOGLE;
        group.add(built);
        const head = new TextDecoder().decode(buffer.slice(0, 400));
        group.userData.credit = /comment credit (.*)\n/.exec(head)?.[1];
        group.userData.google = buffer;
      }
      if (placement === 'world' && data[WATER]) {
        progress('Loading the water…');
        const buffer = await readBuffer(resolve(relativePath(data[WATER])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        for (const water of waterSurfaces(JSON.parse(new TextDecoder().decode(buffer)), (e, n) =>
          gapOf(e, -n),
        )) {
          water.userData.surroundings = WATER;
          group.add(water);
        }
      }
      if (placement === 'world' && data[LIFE]) {
        progress('Loading what moves…');
        const buffer = await readBuffer(resolve(relativePath(data[LIFE])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        const life = JSON.parse(new TextDecoder().decode(buffer));
        const moving = [
          life.cars.roads.length ? trafficPoints(life.cars) : null,
          birdPoints(life.birds),
          boatPoints(life.boats),
          life.ducks && duckPoints(life.ducks),
          life.cats && catPoints(life.cats),
        ];
        for (const points of moving.filter(Boolean)) {
          points.userData.surroundings = LIFE;
          group.add(points);
        }
      }
    }
    if (cancelled()) {
      dispose(group);
      return null;
    }
    return { group, data, placement };
  } catch (e) {
    dispose(group);
    throw e;
  }
}
export class SceneStore {
  group = null;
  data = null;
  placement = null;
  name = 'No scene open';
  splat = null;
  nodes = new Map();
  corrections = [];
  undo = [];
  redo = [];
  checkpoint = '';
  exportRequested = false;
  install(asset, name) {
    dispose(this.group);
    Object.assign(this, asset);
    this.name = name;
    this.nodes = new Map();
    this.splat = null;
    this.group.traverse((o) => {
      if (o.isPoints && o.userData.nodeIndex != null) this.nodes.set(o.userData.nodeIndex, o);
      if (o.userData.splat) this.splat = o;
    });
    this.corrections = this.data?.nodes.map(() => new THREE.Matrix4()) || [];
    this.undo = [];
    this.redo = [];
    this.checkpoint = JSON.stringify(this.data);
    this.exportRequested = false;
  }
  snapshot() {
    return this.corrections.map((m) => m.clone());
  }
  exported() {
    const data = structuredClone(this.data);
    if (!data) return null;
    data.nodes.forEach((n, i) => {
      if (n.transform && !this.corrections[i].equals(identity))
        n.transform = exportPlacement(n.transform, this.corrections[i]);
    });
    return data;
  }
  get dirty() {
    return !!this.data && JSON.stringify(this.exported()) !== this.checkpoint;
  }
  markExportRequested() {
    this.checkpoint = JSON.stringify(this.exported());
    this.exportRequested = true;
  }
  apply() {
    this.nodes.forEach((o, i) => {
      o.matrixAutoUpdate = false;
      o.matrix.copy(this.corrections[i]);
      o.updateMatrixWorld(true);
    });
  }
  preview(delta, before, members) {
    const next = members.map((i) => [i, delta.clone().multiply(before[i])]);
    if (next.some(([, m]) => !m.elements.every(Number.isFinite)))
      throw Error('Adjustment is too large.');
    next.forEach(([i, m]) => (this.corrections[i] = m));
    this.apply();
  }
  commit(before) {
    if (
      before.every((m, i) =>
        m.elements.every((v, j) => Math.abs(v - this.corrections[i].elements[j]) < 1e-10),
      )
    )
      return;
    this.undo.push({ before, after: this.snapshot() });
    if (this.undo.length > 100) this.undo.shift();
    this.redo = [];
  }
  restore(snapshot) {
    this.corrections = snapshot.map((m) => m.clone());
    this.apply();
  }
  history(redo = false) {
    const from = redo ? this.redo : this.undo,
      to = redo ? this.undo : this.redo,
      entry = from.pop();
    if (entry) {
      this.restore(redo ? entry.after : entry.before);
      to.push(entry);
    }
  }
  resetPiece(members) {
    const before = this.snapshot();
    members.forEach((i) => this.corrections[i].identity());
    this.apply();
    this.commit(before);
  }
  box(members = null, visibleOnly = false) {
    const box = new THREE.Box3();
    // points only: a splat has no bounds (see app.js SPLAT_RADIUS)
    const objects = members
      ? members.map((i) => this.nodes.get(i)).filter(Boolean)
      : (this.group?.children || []).filter((o) => o.isPoints && !o.userData.surroundings);
    objects.forEach((o) => {
      o.updateMatrixWorld(true);
      if (!visibleOnly || o.visible)
        box.union(o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld));
    });
    return box;
  }
  prepareGPS(scale) {
    if (this.placement !== 'raw' || !Number.isFinite(scale) || scale <= 0)
      throw Error('Enter a positive scale.');
    const rows = gpsPlacement(this.data, scale),
      matrix = flip
        .clone()
        .multiply(new THREE.Matrix4().set(...rows.flat()))
        .multiply(flip);
    this.nodes.forEach((o) => {
      o.geometry.applyMatrix4(matrix);
      o.geometry.computeBoundingBox();
      o.geometry.computeBoundingSphere();
    });
    scenePieces(this.data)[0].forEach(
      (i) => (this.data.nodes[i].transform = rows.map((r) => [...r])),
    );
    this.placement = 'world'; // the GPS placement becomes the reset baseline
  }
}
