import { tickWater } from '@viewer/effects/water';
import { tickMoving } from '@viewer/effects/moving';
import { playDemo, stopDemo, tickDemo } from '@viewer/effects/demo';
import { placeShots } from '@viewer/effects/shot';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { createMatrixPass, MATRIX_CELL_SIZE } from '@viewer/effects/matrix';
import { createAnimePass } from '@viewer/effects/anime';
import { createDitherPass, ditherCellSize } from '@viewer/effects/dither';
import { createEnvironment } from '@viewer/effects/environment';
import { pointMotion } from '@viewer/effects/points';

import { STYLE_DEFAULTS, normalizeStyle } from '@viewer/effects/presets';

export function createStyles(scene, camera, renderer) {
  const environment = createEnvironment(scene);
  const originalBackground = scene.background;
  let style = 'original',
    settings = { ...STYLE_DEFAULTS.original },
    composer,
    anime,
    dither,
    matrix;
  const matrixBackground = new THREE.Color(0x15191d);
  // A placed scene is in metres, so the look (float, scan) is one
  // fixed size, the small Stockholm scene's radius, whatever
  // the scene's extent; the reveal still sweeps the scene's own radius.
  const LOOK_M = 33;
  let asset = null,
    radius = 1,
    look = 1,
    time = 0,
    revealStart = null,
    entries = [];
  const center = new THREE.Vector3();
  function initialize() {
    if (composer) return;
    composer = new EffectComposer(renderer);
    for (const target of [composer.renderTarget1, composer.renderTarget2])
      target.depthTexture = new THREE.DepthTexture(target.width, target.height);
    anime = createAnimePass();
    dither = createDitherPass();
    matrix = createMatrixPass();
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(anime);
    composer.addPass(dither);
    composer.addPass(matrix);
    composer.addPass(new OutputPass());
  }
  // the scene's own points' middle, at their bottom
  function foot() {
    const box = asset?.box();
    if (!box || box.isEmpty()) return null;
    const at = box.getCenter(new THREE.Vector3());
    at.y = box.min.y;
    return at;
  }
  function resize() {
    if (!composer) return;
    const size = renderer.getSize(new THREE.Vector2()),
      ratio = renderer.getPixelRatio();
    composer.setPixelRatio(ratio);
    composer.setSize(size.x, size.y);
    anime.uniforms.texel.value.set(1 / (size.x * ratio), 1 / (size.y * ratio));
    dither.uniforms.resolution.value.set(size.x * ratio, size.y * ratio);
    matrix.uniforms.resolution.value.set(size.x * ratio, size.y * ratio);
    matrix.uniforms.cellSize.value = MATRIX_CELL_SIZE * ratio;
  }
  function updateMotion(editing) {
    for (const { uniforms, animated } of entries) {
      uniforms.styleDensity.value = settings.density / 100;
      uniforms.stylePointScale.value = style === 'paint' ? 1.2 : 1;
      uniforms.styleRound.value = style === 'paint' ? 1 : 0;
      uniforms.styleRadius.value = radius;
      uniforms.styleLook.value = look;
      uniforms.styleCenter.value.copy(center);
      uniforms.styleTime.value = time;
      uniforms.styleFloat.value =
        style !== 'original' && !animated && !editing && settings.floating ? settings.amount : 0;
      uniforms.styleScan.value = style !== 'original' && !editing && settings.scan ? 1 : 0;
      uniforms.styleReveal.value =
        animated || editing || revealStart === null || style === 'original'
          ? 1
          : Math.min(1, (time - revealStart) / 4);
    }
  }
  return {
    set(next, options) {
      next = normalizeStyle(next);
      if (!(next in STYLE_DEFAULTS)) return;
      style = next;
      settings = { ...STYLE_DEFAULTS[next], ...options };
      if (style !== 'original') {
        initialize();
        resize();
      }
      updateMotion(false);
    },
    configure(store, r) {
      asset = store;
      radius = r;
      look = store.placement === 'world' ? LOOK_M : r;
      const box = store.box();
      box.isEmpty() ? center.set(0, 0, 0) : box.getCenter(center);
      entries = [];
      stopDemo();
      placeShots(foot() ?? new THREE.Vector3()); // the world whole, shots kept round it
      store.group?.traverse((object) => {
        if (object.userData.pointStyle) {
          // the buildings' strokes: those near DA3 drawn as its points are
          entries.push({ uniforms: object.material.uniforms, animated: false });
        } else if (object.isPoints) {
          entries.push({ uniforms: pointMotion(object), animated: false });
        }
      });
      // Animated world objects share the point look, but retain their own motion.
      const materials = new Set(entries.map(({ uniforms }) => uniforms));
      scene.traverse((object) => {
        if (!object.isPoints || !object.userData.styleAnimated) return;
        const uniforms = pointMotion(object);
        if (!materials.has(uniforms)) {
          entries.push({ uniforms, animated: true });
          materials.add(uniforms);
        }
      });
      revealStart = null;
      updateMotion(false);
    },
    reveal() {
      if (entries.length) revealStart = time;
    },
    // a demo (demo.js) round the scene's foot
    demo(name) {
      const at = foot();
      if (at) playDemo(name, at);
    },
    resize,
    render(dt, editing = false) {
      if (!editing) time += dt;
      updateMotion(editing);
      if (!editing) tickWater(dt);
      if (!editing) tickMoving(dt);
      if (editing) stopDemo();
      else tickDemo(dt);
      environment.update(
        style === 'paint' ? 'anime' : style,
        camera,
        !!asset?.group && settings.atmosphere && style !== 'matrix',
        !!asset?.group && !asset.splat,
        time,
      );
      scene.background = style === 'matrix' ? matrixBackground : originalBackground;
      if (style === 'original' || !asset?.group) {
        renderer.render(scene, camera);
        return;
      }
      anime.enabled = style === 'paint';
      dither.enabled = style === 'dither';
      matrix.enabled = style === 'matrix';
      matrix.uniforms.time.value = time;
      matrix.uniforms.strength.value = settings.strength;
      const depth = composer.readBuffer.depthTexture;
      // Spark blends transparent Gaussians without reliable surface depth.
      // Grade its colour normally; do not interpret the background's depth as a splat surface.
      const useDepth = asset.splat ? 0 : 1;
      anime.uniforms.tDepth.value = dither.uniforms.tDepth.value = depth;
      anime.uniforms.useDepth.value = dither.uniforms.useDepth.value = useDepth;
      anime.uniforms.strength.value = dither.uniforms.strength.value = settings.strength;
      const viewportSize = renderer.getSize(new THREE.Vector2());
      dither.uniforms.pixelSize.value = ditherCellSize(
        viewportSize.x,
        viewportSize.y,
        renderer.getPixelRatio(),
        settings.pixels,
      );
      dither.uniforms.near.value = camera.near;
      dither.uniforms.far.value = camera.far;
      // Render editor handles after grading so their axis colours stay legible.
      const overlays = scene.children.filter((o) => o.userData.styleOverlay && o.visible);
      overlays.forEach((o) => (o.visible = false));
      try {
        composer.render(dt);
      } finally {
        overlays.forEach((o) => (o.visible = true));
      }
      if (overlays.length) {
        const visible = scene.children.map((o) => o.visible);
        const background = scene.background,
          autoClear = renderer.autoClear;
        try {
          scene.children.forEach((o) => (o.visible = overlays.includes(o)));
          scene.background = null;
          renderer.autoClear = false;
          renderer.clearDepth();
          renderer.render(scene, camera);
        } finally {
          scene.children.forEach((o, i) => (o.visible = visible[i]));
          scene.background = background;
          renderer.autoClear = autoClear;
        }
      }
    },
    dispose() {
      environment.dispose();
      composer?.passes.forEach((pass) => pass.dispose?.());
      composer?.dispose();
      scene.background = originalBackground;
    },
  };
}
