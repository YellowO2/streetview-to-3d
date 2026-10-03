import { tickWater } from '@viewer/world/water';
import { tickMoving } from '@viewer/life/moving';
import { playDemo, stopDemo, tickDemo } from '@viewer/style/demo';
import { placeShots } from '@viewer/flight/shot';
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { createAnimePass } from '@viewer/style/anime';
import { createDitherPass, ditherCellSize } from '@viewer/style/dither';
import { createEnvironment } from '@viewer/world/environment';
import { pointMotion } from '@viewer/style/points';
import { setGlyphs } from '@viewer/style/glyphs';
import { STYLE_DEFAULTS, normalizeStyle } from '@viewer/style/presets';

// Style pipeline: sky and clouds, per-frame point uniforms, demos, and the postprocess passes.
// A placed scene is in metres, so the float and scan scale is fixed (the small Stockholm scene's radius).
const LOOK_M = 33;

export function createStyles(scene, camera, renderer) {
  const environment = createEnvironment(scene);
  const originalBackground = scene.background;
  let style = 'original',
    settings = { ...STYLE_DEFAULTS.original },
    composer,
    anime,
    dither;
  // Characters is painted like Soft paint
  const painted = () => style === 'paint' || style === 'characters';
  let asset = null,
    look = 1,
    time = 0,
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
  // the scene's own points' centre, at their bottom
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
  }
  function updateMotion(editing) {
    for (const { uniforms, animated } of entries) {
      uniforms.styleDensity.value = settings.density / 100;
      uniforms.stylePointScale.value = painted() ? 1.2 : 1;
      uniforms.styleRound.value = painted() ? 1 : 0;
      uniforms.styleLook.value = look;
      uniforms.styleCenter.value.copy(center);
      uniforms.styleTime.value = time;
      uniforms.styleFloat.value =
        style !== 'original' && !animated && !editing && settings.floating ? settings.amount : 0;
      uniforms.styleScan.value = style !== 'original' && !editing && settings.scan ? 1 : 0;
    }
  }
  return {
    set(next, options) {
      next = normalizeStyle(next);
      if (!(next in STYLE_DEFAULTS)) return;
      style = next;
      settings = { ...STYLE_DEFAULTS[next], ...options };
      setGlyphs(style === 'characters' ? settings.characters : '', settings.characterSize);
      if (style !== 'original') {
        initialize();
        resize();
      }
      updateMotion(false);
    },
    configure(store, r) {
      asset = store;
      look = store.placement === 'world' ? LOOK_M : r;
      const box = store.box();
      box.isEmpty() ? center.set(0, 0, 0) : box.getCenter(center);
      entries = [];
      stopDemo();
      placeShots(foot() ?? new THREE.Vector3());
      store.group?.traverse((object) => {
        if (object.userData.pointStyle) {
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
      updateMotion(false);
    },
    // play a demo (demo.js) around the scene's foot
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
        painted() ? 'anime' : style,
        camera,
        !!asset?.group && settings.atmosphere,
        !!asset?.group && !asset.splat,
        time,
      );
      scene.background = originalBackground;
      if (style === 'original' || !asset?.group) {
        renderer.render(scene, camera);
        return;
      }
      anime.enabled = painted();
      dither.enabled = style === 'dither';
      const depth = composer.readBuffer.depthTexture;
      // splats have no reliable surface depth: grade colour without the depth mask
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
