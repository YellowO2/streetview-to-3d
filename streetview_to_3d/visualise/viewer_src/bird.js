import * as THREE from 'three';

export function createBird() {
  // A deliberately simple, low-poly placeholder bird. Local forward is -Z.
  const bird = new THREE.Group();
  const plumage = new THREE.MeshStandardMaterial({
    color: 0xe8eee1,
    roughness: 0.85,
    flatShading: true,
  });
  const wingMaterial = new THREE.MeshStandardMaterial({
    color: 0x8da995,
    roughness: 0.85,
    flatShading: true,
  });
  const beakMaterial = new THREE.MeshStandardMaterial({
    color: 0xf4b75f,
    roughness: 0.7,
    flatShading: true,
  });
  function ellipsoid(parent, material, position, scale) {
    const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), material);
    mesh.position.set(...position);
    mesh.scale.set(...scale);
    parent.add(mesh);
    return mesh;
  }
  ellipsoid(bird, plumage, [0, 0, 0], [0.32, 0.28, 0.62]);
  ellipsoid(bird, plumage, [0, 0.2, -0.48], [0.25, 0.25, 0.28]);
  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.32, 4), beakMaterial);
  beak.rotation.x = -Math.PI / 2;
  beak.position.set(0, 0.16, -0.83);
  bird.add(beak);
  const eyes = new THREE.MeshBasicMaterial({ color: 0x162328 });
  for (const side of [-1, 1])
    ellipsoid(bird, eyes, [side * 0.205, 0.26, -0.61], [0.045, 0.045, 0.045]);
  ellipsoid(bird, wingMaterial, [0, 0.04, 0.61], [0.25, 0.075, 0.3]);
  const wings = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.22, 0.04, 0);
    bird.add(pivot);
    ellipsoid(pivot, wingMaterial, [side * 0.48, 0, 0.08], [0.67, 0.065, 0.27]);
    return pivot;
  });
  bird.visible = false;
  return { bird, wings };
}
