import * as THREE from 'three';
export function createViewport(host) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#11171e');
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 10000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('aria-label', '3D point cloud viewport');
  host.prepend(canvas);
  scene.add(new THREE.HemisphereLight(0xe7f1ff, 0x63714c, 2.6));
  const sun = new THREE.DirectionalLight(0xffefd5, 2.2);
  sun.position.set(3, 5, 4);
  scene.add(sun);
  const resize = () => {
    const w = Math.max(host.clientWidth, 1),
      h = Math.max(host.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  };
  new ResizeObserver(resize).observe(host);
  resize();
  const highlight = new THREE.Box3Helper(new THREE.Box3(), 0xc2e6ad);
  highlight.material.depthTest = false;
  highlight.material.transparent = true;
  highlight.renderOrder = 10;
  highlight.visible = false;
  scene.add(highlight);
  const ray = new THREE.Raycaster();
  return {
    scene,
    camera,
    renderer,
    canvas,
    highlight,
    pick(event, store, pointSize) {
      if (!store.group) return null;
      const r = canvas.getBoundingClientRect();
      ray.params.Points.threshold = pointSize * 2;
      ray.setFromCamera(
        new THREE.Vector2(
          ((event.clientX - r.left) / r.width) * 2 - 1,
          1 - ((event.clientY - r.top) / r.height) * 2,
        ),
        camera,
      );
      return (
        ray.intersectObjects(
          store.group.children.filter((o) => o.visible),
          false,
        )[0] || null
      );
    },
  };
}
