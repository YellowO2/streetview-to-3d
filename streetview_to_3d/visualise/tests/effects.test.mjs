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
  assert.equal($('point-density').value, '90');
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
  ui.select('voxel');
  ui.render({ group: {}, splat: {} }, false);
  assert.equal($('visual-style').value, 'original');
  assert(document.querySelector('option[value="voxel"]').disabled);
  ui.select('paint');
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

test('mist terrain stays fixed when camera moves and is excluded from capture geometry', () => {
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera();
  const environment = createEnvironment(scene);
  const box = new THREE.Box3(new THREE.Vector3(10, -2, 20), new THREE.Vector3(30, 8, 40));
  environment.configure(box, 15);
  environment.update('dither', camera);
  const anchor = environment.terrain.position.clone();
  const matrices = environment.terrain.children.map((o) =>
    o.geometry.attributes.position.array.slice(),
  );
  camera.position.set(100, 200, 300);
  camera.rotation.y = 1.5;
  environment.update('dither', camera);
  assert(environment.terrain.position.equals(anchor));
  assert(scene.fog);
  assert(environment.terrain.visible);
  environment.terrain.children.forEach((o, i) => {
    assert.deepEqual(o.geometry.attributes.position.array, matrices[i]);
    assert(!o.isPoints);
  });
  environment.update('original', camera);
  assert.equal(scene.fog, null);
  assert(!environment.terrain.visible);
  environment.dispose();
  assert.equal(scene.children.length, 0);
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
