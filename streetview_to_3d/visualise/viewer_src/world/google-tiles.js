// Experimental: the surroundings from Google's Photorealistic 3D Tiles instead of the map-built
// land and buildings. The tiles around the scene are streamed once and laid with dabs as the
// land is; postprocess/world/google.py merges a saved copy into DA3's points. Export leaves
// them out.
import * as THREE from 'three';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { landMarks } from '@viewer/world/land';
import { RATIO } from '@viewer/world/blocks';

const LIBRARY = 'https://unpkg.com/3d-tiles-renderer@0.5.3/build/',
  DRACO = 'https://unpkg.com/three@0.178.0/examples/jsm/libs/draco/gltf/';
const RADIUS_M = 1000, // as far as the land reaches (postprocess terrain.RADIUS_M)
  CAM_H = 2.45, // a pano's camera above its ground (postprocess place.CAM_H)
  RIG_UP_M = 30, // the loading cameras this far above the scene
  RESOLUTION = 4096, // their screens: with ERROR, a tile's allowed error is about distance / 340
  ERROR = 12,
  SETTLED_M = 0.05, // heights fitted once a refit moves them less than this
  TIMEOUT_S = 180,
  ROW = 31; // bytes a point in the PLY

// viewer frame (east, up, south) from east-north-up
const FROM_ENU = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);

// The tiles within RADIUS_M of [lat, lon] as dabs, with the tiles' ground under the cameras
// (cams: [x, y, z] each, viewer frame) put CAM_H below them. gapOf(x, z): point spacing there.
// Resolves with a PLY's bytes.
export async function googleTiles(key, [lat, lon], cams, gapOf, progress = () => {}) {
  const [{ TilesRenderer }, { GoogleCloudAuthPlugin, GLTFExtensionsPlugin }] = await Promise.all([
    import(LIBRARY + 'index.three.js'),
    import(LIBRARY + 'index.plugins.js'),
  ]);
  const tiles = new TilesRenderer();
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: key }));
  tiles.registerPlugin(
    new GLTFExtensionsPlugin({ dracoLoader: new DRACOLoader().setDecoderPath(DRACO) }),
  );
  tiles.errorTarget = ERROR;
  // the scene's place on the globe at the origin; height is fitted below
  const rig = new THREE.Group();
  rig.matrixAutoUpdate = false;
  rig.add(tiles.group);
  let height = 0;
  const place = () => {
    const frame = tiles.ellipsoid.getEastNorthUpFrame(
      THREE.MathUtils.degToRad(lat),
      THREE.MathUtils.degToRad(lon),
      height,
      new THREE.Matrix4(),
    );
    rig.matrix.copy(FROM_ENU).multiply(frame.invert());
    rig.updateMatrixWorld(true);
  };
  place();
  // cameras looking out and down from above the scene's middle: detail falls with distance
  const middle = cams
    .reduce((s, c) => s.add(new THREE.Vector3(...c)), new THREE.Vector3())
    .divideScalar(cams.length);
  const looks = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 0, 1],
    [0, 0, -1],
    [0, -1, 0],
  ].map((way) => {
    const camera = new THREE.PerspectiveCamera(90, 1, 1, 2 * RADIUS_M);
    camera.position.copy(middle).setY(middle.y + RIG_UP_M);
    camera.up.set(0, way[1] ? 0 : 1, way[1] ? -1 : 0);
    camera.lookAt(camera.position.clone().add(new THREE.Vector3(...way)));
    camera.updateMatrixWorld(true);
    tiles.setCamera(camera);
    tiles.setResolution(camera, RESOLUTION, RESOLUTION);
    return camera;
  });

  // the tiles' ground height under each camera, by a ray straight down
  const ray = new THREE.Raycaster();
  const fit = () => {
    const off = [];
    for (const [x, y, z] of cams) {
      ray.set(new THREE.Vector3(x, 1e4, z), new THREE.Vector3(0, -1, 0));
      const hit = ray.intersectObject(tiles.group, true)[0];
      if (hit) off.push(hit.point.y - (y - CAM_H));
    }
    if (!off.length) return Infinity;
    off.sort((a, b) => a - b);
    const move = off[off.length >> 1];
    height += move; // a higher origin lowers the tiles
    place();
    return Math.abs(move);
  };

  // load until nothing is pending and the heights have settled
  const started = performance.now();
  await new Promise((resolve, reject) => {
    let idle = 0;
    const step = () => {
      for (const camera of looks) camera.updateMatrixWorld(true);
      tiles.update();
      const { downloading, parsing, queued = 0 } = tiles.stats;
      progress(`Loading Google 3D Tiles… ${tiles.visibleTiles.size} tiles`);
      idle = downloading + parsing + queued === 0 && tiles.visibleTiles.size ? idle + 1 : 0;
      if (idle > 30 && fit() < SETTLED_M) return resolve();
      if (idle > 30) idle = 0; // heights moved: the cameras see other tiles now
      if (performance.now() - started > TIMEOUT_S * 1000)
        return reject(Error('Google 3D Tiles took too long to load.'));
      setTimeout(step, 30);
    };
    step();
  });

  // each tile's triangles laid with dabs as the land's are (land.landMarks), coloured from its
  // photo; a tile at a time, kept as packed arrays: a city's tiles make many millions
  const meshes = [];
  tiles.group.updateMatrixWorld(true);
  tiles.group.traverseVisible((o) => o.isMesh && meshes.push(o));
  const colour = new THREE.Color(),
    chunks = [];
  for (const [n, mesh] of meshes.entries()) {
    if (n % 10 === 0) {
      progress(`Turning the tiles into points… ${n} of ${meshes.length}`);
      await new Promise((r) => setTimeout(r, 0));
    }
    const geometry = mesh.geometry.clone().applyMatrix4(mesh.matrixWorld);
    const position = geometry.getAttribute('position'),
      uv = geometry.getAttribute('uv'),
      pixels = imageOf(mesh.material.map);
    if (!geometry.index) geometry.setIndex([...Array(position.count).keys()]);
    const made = landMarks(geometry, gapOf, {
      smooth: false,
      colourAt: (a, b, c, [wa, wb, wc]) => {
        if (!pixels || !uv) return mesh.material.color.toArray();
        const s = wa * uv.getX(a) + wb * uv.getX(b) + wc * uv.getX(c),
          r = wa * uv.getY(a) + wb * uv.getY(b) + wc * uv.getY(c);
        const px = Math.min(pixels.width - 1, Math.max(0, Math.floor(s * pixels.width))),
          py = Math.min(pixels.height - 1, Math.max(0, Math.floor(r * pixels.height))),
          at = 4 * (py * pixels.width + px);
        return colour
          .setRGB(
            pixels.data[at] / 255,
            pixels.data[at + 1] / 255,
            pixels.data[at + 2] / 255,
            THREE.SRGBColorSpace,
          )
          .toArray();
      },
    });
    // within RADIUS_M, as the PLY's rows: x y z, red green blue, gap, normal (world frame: -y, -z)
    const rows = new DataView(new ArrayBuffer(ROW * made.dab.length));
    let count = 0;
    for (let i = 0; i < made.dab.length; i++) {
      const [x, y, z] = [0, 1, 2].map((d) => made.centre[3 * i + d]);
      if (Math.hypot(x - middle.x, z - middle.z) > RADIUS_M) continue;
      const at = ROW * count++;
      for (let d = 0; d < 3; d++) {
        const flip = d ? -1 : 1;
        rows.setFloat32(at + 4 * d, flip * made.centre[3 * i + d], true);
        rows.setFloat32(at + 19 + 4 * d, flip * made.facing[3 * i + d], true);
      }
      colour.fromArray(made.tint, 3 * i).convertLinearToSRGB();
      for (const [d, v] of colour.toArray().entries())
        rows.setUint8(at + 12 + d, Math.round(Math.min(Math.max(v, 0), 1) * 255));
      rows.setFloat32(at + 15, made.dab[i] / RATIO, true); // its spacing
    }
    chunks.push(new Uint8Array(rows.buffer, 0, ROW * count));
  }
  const credit = tiles
    .getAttributions([])
    .map((x) => x.value)
    .join('; ');
  tiles.dispose();
  // a binary PLY as buildings.ply is, Google's attribution in a header comment
  const count = chunks.reduce((n, c) => n + c.length, 0) / ROW;
  const head = new TextEncoder().encode(
    `ply\nformat binary_little_endian 1.0\ncomment credit ${credit}\nelement vertex ${count}\n` +
      'property float x\nproperty float y\nproperty float z\n' +
      'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float gap\n' +
      'property float nx\nproperty float ny\nproperty float nz\nend_header\n',
  );
  const out = new Uint8Array(head.length + ROW * count);
  out.set(head);
  let at = head.length;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.length;
  }
  return out.buffer;
}

// a texture's pixels, read once
const canvas = document.createElement('canvas'),
  context = canvas.getContext('2d', { willReadFrequently: true });
function imageOf(map) {
  const image = map?.image;
  if (!image?.width) return null;
  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, image.width, image.height);
}
