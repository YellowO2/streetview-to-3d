import * as THREE from "three";
import { SplatMesh } from "@sparkjsdev/spark";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class SceneManager {
  constructor() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1,
      1000,
    );
    // 2. Position the camera ever so slightly off-center (so the controls have a target to look at)
    this.camera.position.set(0, 0, 0.1);

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);

    // 3. Initialize OrbitControls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableZoom = false; // Disable zooming since you are "inside" a sphere
    this.controls.enablePan = false; // Disable panning
    this.controls.enableDamping = true; // Add a nice smooth glide when dragging
    this.controls.dampingFactor = 0.05;
  }

  addPanorama(imageUrl) {
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(imageUrl);
    const geometry = new THREE.SphereGeometry(100, 60, 40);
    geometry.scale(-1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    this.sphere = new THREE.Mesh(geometry, material);
    this.scene.add(this.sphere);
  }

  startRenderLoop() {
    // Seen as the main loop of the scene
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }
}
