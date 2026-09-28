import * as THREE from 'three';

// Shared lifecycle for temporary geometry styles; source data always stays authoritative.
export function createGeometryStyle(scene, build, disposeShared = () => {}) {
  const root = new THREE.Group();
  scene.add(root);
  let sources = [],
    meshes = [],
    cachedSize = null;
  function clear() {
    meshes.forEach((mesh) => {
      mesh.userData.disposeStyle?.();
    });
    root.clear();
    meshes = [];
    cachedSize = null;
  }
  return {
    configure(group) {
      clear();
      sources = [];
      group?.traverse((object) => {
        if (object.isPoints) sources.push(object);
      });
    },
    render(active, size, draw) {
      root.visible = active && sources.length > 0;
      if (!root.visible) return draw();
      if (cachedSize !== size) {
        clear();
        for (const source of sources) {
          const mesh = build(source, size, sources.length);
          mesh.matrixAutoUpdate = false;
          root.add(mesh);
          meshes.push(mesh);
        }
        cachedSize = size;
      }
      const visibility = sources.map((source) => source.visible);
      sources.forEach((source, i) => {
        source.updateWorldMatrix(true, false);
        meshes[i].matrix.copy(source.matrixWorld);
        let visible = true;
        for (let parent = source; parent; parent = parent.parent) visible &&= parent.visible;
        meshes[i].visible = visible;
        source.visible = false;
      });
      try {
        return draw();
      } finally {
        sources.forEach((source, i) => {
          source.visible = visibility[i];
        });
      }
    },
    dispose() {
      clear();
      root.removeFromParent();
      disposeShared();
    },
  };
}
