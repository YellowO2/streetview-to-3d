import { createWater } from '@viewer/effects/water';
import { createVoxels } from '@viewer/effects/voxel';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { createAnimePass } from '@viewer/effects/anime';
import { createDitherPass, ditherCellSize } from '@viewer/effects/dither';
import { createEnvironment } from '@viewer/effects/environment';
import { pointMotion } from '@viewer/effects/points';

import { STYLE_DEFAULTS, normalizeStyle } from '@viewer/effects/presets';

export function createStyles(scene, camera, renderer) {
  const voxels = createVoxels(scene);
  const water = createWater(scene);
  const environment = createEnvironment(scene);
  const originalBackground = scene.background;
  let style = 'original',
    settings = { ...STYLE_DEFAULTS.original },
    composer,
    anime,
    dither;
  let asset = null,
    radius = 1,
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
    composer.addPass(new RenderPass(scene, camera));
    composer.addPass(anime);
    composer.addPass(dither);
    composer.addPass(new OutputPass());
  }
  function resize() {
    if (!composer) return;
    const size = renderer.getSize(new THREE.Vector2()),
      ratio = renderer.getPixelRatio();
    composer.setPixelRatio(ratio);
    composer.setSize(size.x, size.y);
    anime.uniforms.texel.value.set(1 / (size.x * ratio), 1 / (size.y * ratio));
    dither.uniforms.resolution.value.set(size.x * ratio, size.y * ratio);
  }
  function updateMotion(editing) {
    for (const uniforms of entries) {
      uniforms.styleDensity.value = settings.density / 100;
      uniforms.stylePointScale.value = style === 'paint' ? 1.2 : 1;
      uniforms.styleRound.value = style === 'paint' ? 1 : 0;
      uniforms.styleRadius.value = radius;
      uniforms.styleCenter.value.copy(center);
      uniforms.styleTime.value = time;
      uniforms.styleFloat.value =
        style !== 'original' && !editing && settings.floating ? settings.amount : 0;
      uniforms.styleScan.value = style !== 'original' && !editing && settings.scan ? 1 : 0;
      uniforms.styleReveal.value =
        editing || revealStart === null || style === 'original'
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
      voxels.configure(store.group);
      radius = r;
      const box = store.box();
      box.isEmpty() ? center.set(0, 0, 0) : box.getCenter(center);
      entries = [];
      store.group?.traverse((object) => {
        if (object.isPoints) {
          entries.push(pointMotion(object));
        }
      });
      revealStart = null;
      environment.configure(box, radius);
      water.configure(box, radius);
      updateMotion(false);
    },
    reveal() {
      if (entries.length) revealStart = time;
    },
    resize,
    render(dt, editing = false) {
      return voxels.render(style === 'voxel', radius * 0.008 * (settings.blocks || 1), () => {
        if (!editing) time += dt;
        updateMotion(editing);
        water.update(
          !!asset?.group && !!settings.water && !asset.splat,
          settings.waterLevel ?? 0.1,
          dt,
          editing || settings.stillWater,
        );
        environment.update(
          settings.water && !asset?.splat
            ? 'anime'
            : ['paint', 'voxel'].includes(style)
              ? 'anime'
              : style,
          camera,
          !!asset?.group && (settings.atmosphere || (settings.water && !asset.splat)),
        );
        scene.background = originalBackground;
        if (style === 'original' || !asset?.group) {
          renderer.render(scene, camera);
          return;
        }
        anime.enabled = style === 'paint';
        dither.enabled = style === 'dither';
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
        dither.uniforms.fogDistance.value = radius;
        dither.uniforms.fogAmount.value = settings.atmosphere ? 1 : 0;
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
            fog = scene.fog,
            autoClear = renderer.autoClear;
          try {
            scene.children.forEach((o) => (o.visible = overlays.includes(o)));
            scene.background = null;
            scene.fog = null;
            renderer.autoClear = false;
            renderer.clearDepth();
            renderer.render(scene, camera);
          } finally {
            scene.children.forEach((o, i) => (o.visible = visible[i]));
            scene.background = background;
            scene.fog = fog;
            renderer.autoClear = autoClear;
          }
        }
      });
    },
    dispose() {
      voxels.dispose();
      environment.dispose();
      water.dispose();
      composer?.passes.forEach((pass) => pass.dispose?.());
      composer?.dispose();
      scene.background = originalBackground;
    },
  };
}
