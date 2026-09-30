import * as THREE from 'three';
import { createStyles } from '@viewer/effects/controller';
export function createViewport(host) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#11171e');
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  // drawn at most 1.5 pixels a screen point: the paint style's softness hides it, and a
  // Retina screen's 2 shades 1.8 times the pixels
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', '3D point cloud viewport');
  host.prepend(canvas);
  scene.add(new THREE.HemisphereLight(0xe7f1ff, 0x63714c, 2.6));
  const sun = new THREE.DirectionalLight(0xffefd5, 2.2);
  sun.position.set(3, 5, 4);
  scene.add(sun);
  const styles = createStyles(scene, camera, renderer);
  const resize = () => {
    const w = Math.max(host.clientWidth, 1),
      h = Math.max(host.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    styles.resize();
  };
  new ResizeObserver(resize).observe(host);
  resize();
  const highlight = new THREE.Box3Helper(new THREE.Box3(), 0xc2e6ad);
  highlight.material.depthTest = false;
  highlight.material.transparent = true;
  highlight.renderOrder = 10;
  highlight.visible = false;
  highlight.userData.styleOverlay = true;
  scene.add(highlight);
  return {
    scene,
    camera,
    renderer,
    canvas,
    highlight,
    styles,
  };
}
