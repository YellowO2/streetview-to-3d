import { ditherCellSize } from '@viewer/effects/dither';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { JSDOM } from 'jsdom';
import { viewerTemplate } from './fixture.mjs';
import { createStyleControls } from '@viewer/effects/ui';
import { createEnvironment } from '@viewer/effects/environment';
import { pointMotion } from '@viewer/effects/points';

test('style controls retain per-preset settings and expose only supported splat controls', () => {
  const dom = new JSDOM(viewerTemplate());
  globalThis.document = dom.window.document;
  const calls = [];
  const ui = createStyleControls({ style: (...args) => calls.push(args) });
  const $ = (id) => document.getElementById(id);
  ui.render({ group: {}, splat: null }, false);
  ui.select('paint');
  assert.equal($('point-density').value, '100');
  assert.equal($('style-float').value, '0.3');
  assert.equal(2 ** Number($('point-size').value), 1);
  $('point-density').value = '60';
  $('point-density').dispatchEvent(new dom.window.Event('input'));
  assert.equal(calls.at(-1)[1].density, 60);
  assert.equal($('style-floating').checked, true);
  $('style-strength').value = '.37';
  $('style-strength').dispatchEvent(new dom.window.Event('input'));
  ui.select('dither');
  assert.equal($('point-density').value, '100');
  assert.equal($('style-pixels-label').hidden, false);
  assert.equal($('style-floating').checked, false);
  ui.select('paint');
  assert.equal($('style-strength').value, '0.37');
  assert.equal($('point-density').value, '60');
  ui.render({ group: {}, splat: {} }, false);
  assert($('style-point-controls').hidden);
  assert(!$('style-splat-note').hidden);
  ui.render({ group: {}, splat: {} }, true);
  assert($('visual-style').disabled);
  assert($('style-strength').disabled);
  ui.select('original');
  assert($('style-controls').hidden);
  assert.equal(calls.at(-1)[0], 'original');
  dom.window.close();
});

test('environment supplies only the sky, leaving scene terrain untouched', () => {
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera();
  const terrain = new THREE.Group();
  scene.add(terrain);
  const environment = createEnvironment(scene);
  environment.update('dither', camera);
  assert.equal(scene.fog, null);
  assert.equal(scene.children.length, 2);
  assert(terrain.visible);
  environment.update('original', camera);
  environment.dispose();
  assert.deepEqual(scene.children, [terrain]);
});

test('point motion keeps source coordinates and material identity intact', () => {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([1, 2, 3, 4, 5, 6], 3));
  const material = new THREE.PointsMaterial();
  const cloud = new THREE.Points(geometry, material);
  const positions = geometry.attributes.position.array.slice();
  const first = pointMotion(cloud),
    second = pointMotion(cloud);
  assert.equal(first, second);
  first.styleFloat.value = 1;
  first.styleTime.value = 100;
  assert.equal(cloud.material, material);
  assert.deepEqual(geometry.attributes.position.array, positions);
  geometry.dispose();
  material.dispose();
});

test('dither density is bounded across viewport sizes and pixel ratios', () => {
  for (const [width, height] of [
    [640, 480],
    [1920, 1080],
    [3840, 2160],
    [800, 2400],
  ]) {
    for (const ratio of [1, 2]) {
      const size = ditherCellSize(width, height, ratio, 1);
      assert(size >= 3 * ratio);
      assert((Math.max(width, height) * ratio) / size <= 480);
      assert.equal(size / ratio, ditherCellSize(width, height, 1, 1));
    }
  }
  assert.equal(ditherCellSize(640, 480, 2, 8), 16);
});

test('bird materials compose with shared point styling and stable particle density', async () => {
  const { createBird } = await import('@viewer/bird');
  const model = createBird();
  let cloud;
  model.bird.traverse((o) => {
    if (o.isPoints && !cloud) cloud = o;
  });
  assert(cloud.userData.styleAnimated);
  const uniforms = pointMotion(cloud);
  uniforms.stylePointScale.value = 1.2;
  uniforms.styleDensity.value = 0.9;
  const shader = {
    uniforms: {},
    vertexShader: THREE.ShaderLib.points.vertexShader,
    fragmentShader: THREE.ShaderLib.points.fragmentShader,
  };
  cloud.material.onBeforeCompile(shader);
  assert.match(shader.vertexShader, /float phase = styleSeed/);
  assert.match(shader.vertexShader, /vBirdAlpha = birdAlpha/);
  assert.match(shader.fragmentShader, /vBirdAlpha/);
  assert.equal(shader.uniforms.stylePointScale.value, 1.2);
  assert.equal(shader.uniforms.styleDensity.value, 0.9);
  model.dispose();
});
