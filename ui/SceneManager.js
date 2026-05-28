import * as THREE from "three";
import { SplatMesh, SparkControls } from "@sparkjsdev/spark";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class SceneManager {
  constructor() {
    this.scene = new THREE.Scene();

    this.container = document.getElementById("sceneContainer");
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;

    this.camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000);
    this.camera.position.set(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(w, h);
    this.container.appendChild(this.renderer.domElement);

    this.controls = new SparkControls({ canvas: this.renderer.domElement });

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
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

  addMesh(meshUrl, position = { x: 0, y: 0, z: -3 }) {
    const mesh = new SplatMesh({ url: meshUrl });
    mesh.quaternion.set(1, 0, 0, 0);
    mesh.position.set(position.x, position.y, position.z);
    this.scene.add(mesh);
    return mesh;
  }

  resetCamera() {
    this.camera.position.set(0, 0, 0);
    this.camera.rotation.set(0, 0, 0);
    this.camera.quaternion.identity();
  }

  startRenderLoop() {
    // Seen as the main loop of the scene
    this._renderTick = () => {
      this.controls.update(this.camera);
      this.renderer.render(this.scene, this.camera);
    };
    this.renderer.setAnimationLoop(this._renderTick);
  }

  pauseRenderLoop() {
    this.renderer.setAnimationLoop(null);
  }

  resumeRenderLoop() {
    if (this._renderTick) {
      this.renderer.setAnimationLoop(this._renderTick);
    }
  }
}
