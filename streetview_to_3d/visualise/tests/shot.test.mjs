import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { shot, placeShots, clearShots, carve, CELL_M, SIZE } from '@viewer/flight/shot';
import { createGun, RADIUS, SPEED, AIM_M, BIG, CHARGE_S } from '@viewer/flight/gun';
import { pointMotion } from '@viewer/style/points';
import { blockPoints } from '@viewer/world/blocks';
import { landPoints } from '@viewer/world/land';

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
  assert.equal(first.radius, RADIUS);
  assert(Math.abs(first.ball.position.z + 1.6 + SPEED * 0.5) < SPEED * 0.03);
  assert.equal(cell(new THREE.Vector3(0, 2 + RADIUS / 2, -8)), 255);
  for (let t = 0; t < 5; t += 0.05) gun.update(0.05);
  assert(!first.ball.visible);
  gun.dispose();
  clearShots();
});

test('held, the gun is at the lower right of the eye and shoots from its muzzle toward where the eye looks', () => {
  const scene = new THREE.Scene();
  placeShots(new THREE.Vector3());
  const gun = createGun(scene);
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 2, 0); // looking along -z
  gun.hold(camera, true);
  assert(gun.held.visible && gun.held.userData.styleAnimated);
  assert(gun.held.position.x > 0 && gun.held.position.y < 2 && gun.held.position.z < 0);
  assert(gun.fire(camera.position, new THREE.Vector3(0, 0, -1)));
  const [shot] = gun.shots;
  assert(shot.ball.position.distanceTo(camera.position) < 1); // at the muzzle, not 1.6 m ahead
  const aim = camera.position.clone().add(new THREE.Vector3(0, 0, -AIM_M));
  const toward = aim.sub(shot.ball.position).normalize();
  assert(shot.way.dot(toward) > 0.9999); // at where the eye looks
  gun.hold(camera, false);
  assert(!gun.held.visible);
  gun.dispose();
  clearShots();
});

test('a click shoots as ever; held longer, the shot goes bigger, its hole as big', () => {
  const scene = new THREE.Scene();
  placeShots(new THREE.Vector3());
  const gun = createGun(scene);
  const eye = new THREE.Vector3(0, 2, 0),
    way = new THREE.Vector3(0, 0, -1);
  gun.press();
  assert(gun.release(eye, way)); // a click
  const [click, big] = gun.shots;
  assert.equal(click.radius, RADIUS);
  for (let t = 0; t < 0.3; t += 0.05) gun.update(0.05);
  gun.press();
  for (let t = 0; t < CHARGE_S + 1; t += 0.05) gun.update(0.05); // held past CHARGE_S
  assert(gun.release(eye, way));
  assert.equal(big.radius, BIG);
  assert.equal(big.ball.scale.x, BIG / RADIUS);
  for (let t = 0; t < 0.5; t += 0.02) gun.update(0.02);
  assert.equal(cell(new THREE.Vector3(BIG * 0.8, 2, -10)), 255); // cut as wide as it
  gun.dispose();
  clearShots();
});
