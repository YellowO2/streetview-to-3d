import * as THREE from 'three';

export const START_HEIGHT = 1; // Metres above the capture centre, in viewer space.

// Choose the panorama nearest the scene's GPS centre. Its stored camera position
// must pass through the same placement and Y/Z conversion as the PLY geometry.
export function panoramaStart(data) {
  const nodes = data?.nodes?.filter(
    (n) => n.ply && n.position?.length === 3 && n.position.every(Number.isFinite),
  );
  if (!nodes?.length) return null;
  const center = data.center;
  const distance = (n) =>
    Number.isFinite(n.pano?.lat) && Number.isFinite(n.pano?.lon) && center
      ? (n.pano.lat - center[0]) ** 2 +
        ((n.pano.lon - center[1]) * Math.cos((center[0] * Math.PI) / 180)) ** 2
      : Infinity;
  const node = nodes.reduce((a, b) => (distance(b) < distance(a) ? b : a));
  const position = new THREE.Vector3(...node.position);
  if (node.transform) position.applyMatrix4(new THREE.Matrix4().set(...node.transform.flat()));
  position.multiply(new THREE.Vector3(1, -1, -1));
  position.y += START_HEIGHT;
  const heading = Number.isFinite(node.pano?.heading) ? node.pano.heading : 0;
  const target = position.clone().add(new THREE.Vector3(Math.sin(heading), 0, -Math.cos(heading)));
  return { position, target };
}
