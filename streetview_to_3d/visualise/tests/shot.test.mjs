import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { shot, placeShots, clearShots, carve, CELL_M, SIZE } from '@viewer/effects/shot';
import { createGun, RADIUS, SPEED } from '@viewer/gun';
import { pointMotion } from '@viewer/effects/points';
import { blockPoints } from '@viewer/effects/blocks';
import { landPoints } from '@viewer/effects/land';

const cell = (p) => {
  const c = p.clone().sub(shot.shotCorner.value).divideScalar(CELL_M).floor();
  return shot.shotField.value.image.data[c.x + SIZE[0] * (c.y + SIZE[1] * c.z)];
};

test('a shot cuts away what is within its radius of its path, and is mended', () => {
  placeShots(new THREE.Vector3(5, 0, 5));
  assert.equal(shot.shotOn.value, 0);
  const a = new THREE.Vector3(0, 2, 0),
    b = new THREE.Vector3(10, 2, 0);
  assert(carve(a, b, 1));
  assert.equal(shot.shotOn.value, 1);
  assert.equal(cell(new THREE.Vector3(5, 2.3, 0.3)), 255); // on its path
  assert.equal(cell(new THREE.Vector3(5, 4, 0)), 0); // 2 m off it
  assert.equal(cell(new THREE.Vector3(14, 2, 0)), 0); // past its end
  assert(!carve(new THREE.Vector3(1e4, 0, 0), new THREE.Vector3(1e4 + 5, 0, 0), 1)); // off the grid
  clearShots();
  assert.equal(cell(new THREE.Vector3(5, 2.3, 0.3)), 0);
  assert.equal(shot.shotOn.value, 0);
});

test('every kind of point is shot away by the one rule, the bird not', () => {
  const square = () => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 9, 9, 0, 9], 3));
    g.setIndex([0, 1, 2]);
    return g;
  };
  const compiled = (object) => {
    pointMotion(object);
    const shader = {
      uniforms: {},
      vertexShader: THREE.ShaderLib.points.vertexShader,
      fragmentShader: THREE.ShaderLib.points.fragmentShader,
    };
    object.material.onBeforeCompile(shader);
    return shader;
  };
  const world = compiled(new THREE.Points(square(), new THREE.PointsMaterial()));
  assert.match(world.vertexShader, /shotAway\(stylePosition \+ styleCenter\)/);
  assert.equal(world.uniforms.shotField, shot.shotField);
  assert.match(blockPoints(square()).material.vertexShader, /shotAway\(centre\)/);
  assert.match(landPoints(square(), () => 1).material.vertexShader, /shotAway\(centre\)/);
  const bird = new THREE.Points(square(), new THREE.PointsMaterial());
  bird.userData.styleAnimated = true;
  assert.doesNotMatch(compiled(bird).vertexShader, /shotAway/);
});

test('the gun shoots a ball ahead, which cuts its way through and is gone', () => {
  const scene = new THREE.Scene();
  placeShots(new THREE.Vector3());
  const gun = createGun(scene);
  const eye = new THREE.Vector3(0, 2, 0),
    way = new THREE.Vector3(0, 0, -1);
  assert(gun.fire(eye, way));
  assert(!gun.fire(eye, way)); // not again so soon
  const [first] = gun.shots;
  assert(first.ball.visible && first.ball.userData.styleAnimated);
  for (let t = 0; t < 0.5; t += 0.02) gun.update(0.02);
  assert(Math.abs(first.ball.position.z + 1.6 + SPEED * 0.5) < SPEED * 0.03);
  assert.equal(cell(new THREE.Vector3(0, 2 + RADIUS / 2, -8)), 255);
  for (let t = 0; t < 5; t += 0.05) gun.update(0.05);
  assert(!first.ball.visible);
  gun.dispose();
  clearShots();
});
