import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as THREE from 'three';
import { demo, DEMOS, playDemo, tickDemo } from '@viewer/effects/demo';
import { pointMotion } from '@viewer/effects/points';
import { blockPoints } from '@viewer/effects/blocks';
import { landSurface } from '@viewer/effects/land';
import { waterSurfaces } from '@viewer/effects/water';

const points = () => {
  const shader = {
    uniforms: {},
    vertexShader: THREE.ShaderLib.points.vertexShader,
    fragmentShader: THREE.ShaderLib.points.fragmentShader,
  };
  return shader;
};
const square = (y) => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([0, y, 0, 0, y, 9, 9, y, 9], 3));
  g.setIndex([0, 1, 2]);
  return g;
};

test('every kind of point follows the demos, one rule, the bird not', () => {
  const cloud = new THREE.Points(square(0), new THREE.PointsMaterial());
  pointMotion(cloud);
  const shader = points();
  cloud.material.onBeforeCompile(shader);
  assert.match(shader.vertexShader, /demoed\(\(modelMatrix/);
  assert.equal(shader.uniforms.demoKind, demo.demoKind);

  const blocks = blockPoints(square(0));
  assert.match(blocks.material.vertexShader, /demoed\(centre/);
  assert.equal(blocks.material.uniforms.demoT, demo.demoT);

  const land = landSurface(square(0), () => 1);
  assert.match(land.material.vertexShader, /demoed\(w\.xyz/);
  assert.match(land.material.fragmentShader, /demoShown\(world\)/);
  assert.equal(land.material.uniforms.demoCentre, demo.demoCentre);

  const [water] = waterSurfaces(
    {
      surfaces: [
        {
          level: 0,
          outer: [
            [0, 0],
            [9, 0],
            [9, 9],
          ],
          holes: [],
        },
      ],
    },
    () => 1,
  );
  const dabs = water.children.find((c) => c.isPoints);
  assert.match(dabs.material.vertexShader, /demoed\(world\.xyz/);
  assert.equal(dabs.material.uniforms.demoKind, demo.demoKind);
  water.dispose();

  const bird = new THREE.Points(square(0), new THREE.PointsMaterial());
  bird.userData.styleAnimated = true;
  pointMotion(bird);
  const birdShader = points();
  bird.material.onBeforeCompile(birdShader);
  assert.doesNotMatch(birdShader.vertexShader, /demoed/);
});

test('a demo plays round its centre and ends', () => {
  const centre = new THREE.Vector3(1, 2, 3);
  for (const [name, [kind, length]] of Object.entries(DEMOS)) {
    playDemo(name, centre);
    assert.equal(demo.demoKind.value, kind);
    assert(demo.demoCentre.value.equals(centre));
    for (let t = 0; t < length - 0.1; t += 0.05) tickDemo(0.05);
    assert.equal(demo.demoKind.value, kind);
    tickDemo(0.2);
    assert.equal(demo.demoKind.value, 0);
  }
  playDemo('nothing', centre);
  assert.equal(demo.demoKind.value, 0);
});
