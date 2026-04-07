import * as THREE from "three";
import { SplatMesh, SparkControls } from "@sparkjsdev/spark";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ARButton } from "three/addons/webxr/ARButton.js";

export class SceneManager {
  constructor() {
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.1, //limit for rendering. Meaning if anything is closer than 0.1 units, it won't be rendered
      1000,
    );
    this.camera.position.set(0, 0, 0); //start at center

    this.renderer = new THREE.WebGLRenderer();
    this.renderer.xr.enabled = true;
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(this.renderer.domElement);
    document.body.appendChild(ARButton.createButton(this.renderer));

    this.controls = new SparkControls({
      canvas: this.renderer.domElement,
    }); //read sparks docs for more.
  }

  addPanorama(imageUrl) {
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load(imageUrl);
    const geometry = new THREE.SphereGeometry(100, 60, 40); //i should change this to match actual size in the future
    geometry.scale(-1, 1, 1);
    const material = new THREE.MeshBasicMaterial({ map: texture });
    this.sphere = new THREE.Mesh(geometry, material);
    this.scene.add(this.sphere);
  }

  startRenderLoop() {
    // Seen as the main loop of the scene
    this.renderer.setAnimationLoop(() => {
      this.controls.update(this.camera);
      this.renderer.render(this.scene, this.camera);
    });
  }
}
