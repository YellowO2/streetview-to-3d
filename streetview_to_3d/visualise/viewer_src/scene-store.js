import * as THREE from 'three';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import {
  validateScene,
  placementMode,
  relativePath,
  scenePieces,
  gpsPlacement,
  SURROUNDINGS,
  WATER,
} from '@viewer/scene-format';
import { waterSurfaces, disposeWater } from '@viewer/effects/water';
const flip = new THREE.Matrix4().makeScale(1, -1, -1);
const identity = new THREE.Matrix4();
export function matrixRows(m) {
  return Array.from({ length: 4 }, (_, r) =>
    Array.from({ length: 4 }, (_, c) => m.elements[c * 4 + r]),
  );
}
export function exportPlacement(original, correction) {
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
      o.geometry.dispose(); // water: its surfaces share one material
      disposeWater(o.material);
    }
  });
}
const loader = new PLYLoader();
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
// postprocess/terrain.py spaces its points further apart the further they
// are from the nearest camera, out to 2 km (SPACING, as terrain.gap_at):
// drawn at one size they would be dust far out. So they are grouped by
// their spacing, each group drawn its own size in metres
// (userData.pointSize), just over that spacing.
const GAPS = [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20, Infinity];
const nearest = (x, z, places) => Math.min(...places.map(([a, b]) => Math.hypot(x - a, z - b)));
// 5 cm at a camera, GAP_PER of the distance to the nearest one more further
// out (terrain.gap_at); drawn as if it grew only SIZE_PER, so far points
// are sparser without growing as fast
const GAP0 = 0.05,
  GAP_PER = 0.018,
  SIZE_PER = 0.012;
const gapAt = (x, z, cams) => GAP0 + GAP_PER * nearest(x, z, cams);
export const SPACING = { terrain: gapAt, buildings: gapAt };
// Each placed node's camera, seen from above, in the viewer's frame.
export function cameraPlaces(data) {
  return data.nodes
    .filter((n) => n.transform && n.position)
    .map((n) => {
      const v = new THREE.Vector3(...n.position)
        .applyMatrix4(new THREE.Matrix4().set(...n.transform.flat()))
        .applyMatrix4(flip);
      return [v.x, v.z];
    });
}
export function terrainBands(points, spacing = SPACING.terrain, cams = [[0, 0]]) {
  const geometry = points.geometry;
  const p = geometry.getAttribute('position'),
    c = geometry.getAttribute('color');
  const group = new Uint8Array(p.count);
  for (let i = 0; i < p.count; i++) {
    const gap = spacing(p.getX(i), p.getZ(i), cams) * 0.999;
    while (gap > GAPS[group[i]]) group[i]++;
  }
  const bands = GAPS.flatMap((gap, g) => {
    const members = [];
    for (let i = 0; i < p.count; i++) if (group[i] === g) members.push(i);
    if (!members.length) return [];
    const part = new THREE.BufferGeometry();
    const pick = (attr) =>
      new THREE.Float32BufferAttribute(
        members.flatMap((i) => [attr.getX(i), attr.getY(i), attr.getZ(i)]),
        3,
      );
    part.setAttribute('position', pick(p));
    if (c) part.setAttribute('color', pick(c));
    part.computeBoundingBox();
    part.computeBoundingSphere();
    const band = new THREE.Points(part, points.material.clone());
    const spaced = Number.isFinite(gap) ? gap : 30;
    band.userData.pointSize = 1.5 * (GAP0 + ((spaced - GAP0) * SIZE_PER) / GAP_PER);
    return [band];
  });
  geometry.dispose();
  points.material.dispose();
  return bands;
}
// A Gaussian splat (.spz). Spark is imported only when one is opened, so a
// page showing point clouds never downloads it.
export async function parseSplat(buffer) {
  // Spark decodes in a worker, which Chrome will not start for a page opened
  // straight from disk -- it then fails with an unhelpful data-URL error.
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
  mesh.quaternion.set(1, 0, 0, 0); // the same Y-down to Y-up turn `flip` gives points
  mesh.userData.splat = true;
  return mesh;
}
export async function readBuffer(source) {
  if (typeof source !== 'string') return source.arrayBuffer();
  const response = await fetch(source);
  if (!response.ok) throw Error(`Download failed (${response.status}).`);
  return response.arrayBuffer();
}
// A load is staged off-screen; failures and superseded requests cannot destroy
// the currently installed scene. UI decides when to commit the result.
export async function loadAsset(source, resolve, progress, cancelled, { splat = false } = {}) {
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
      // in the world already, so only with a placed scene
      for (const key of placement === 'world' ? SURROUNDINGS : []) {
        if (!data[key]) continue;
        progress(`Loading the ${key}…`);
        const buffer = await readBuffer(resolve(relativePath(data[key])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        const points = parsePoints(buffer);
        const spacing = SPACING[key];
        for (const part of spacing ? terrainBands(points, spacing, cameraPlaces(data)) : [points]) {
          part.userData.surroundings = key;
          group.add(part);
        }
      }
      if (placement === 'world' && data[WATER]) {
        progress('Loading the water…');
        const buffer = await readBuffer(resolve(relativePath(data[WATER])));
        if (cancelled()) {
          dispose(group);
          return null;
        }
        for (const mesh of waterSurfaces(JSON.parse(new TextDecoder().decode(buffer)))) {
          mesh.userData.surroundings = WATER;
          group.add(mesh);
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
    // points only: a splat has no bounds to read (see app.js's splat view)
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
    this.placement = 'world'; // The prepared GPS placement becomes the reset baseline.
  }
}
