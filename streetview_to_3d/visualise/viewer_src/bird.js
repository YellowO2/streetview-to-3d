import { createBirdPlume } from '@viewer/bird-plume';
import { createBirdTrails } from '@viewer/bird-trails';
import * as THREE from 'three';

// A stylised jewel bird, inspired by birds-of-paradise plumage rather than a species model.
// Feather-shaped point ribbons, articulated shoulders, delayed feather/tail motion.
export function createBird() {
  const bird = new THREE.Group();
  bird.name = 'Jewel point-cloud bird';
  const clouds = [];
  const material = new THREE.PointsMaterial({
    vertexColors: true,
    size: 0.012,
    transparent: true,
    depthWrite: false,
    sizeAttenuation: true,
  });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader =
      'attribute float birdAlpha; varying float vBirdAlpha;\n' +
      shader.vertexShader.replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\n vBirdAlpha = birdAlpha;',
      );
    shader.fragmentShader = 'varying float vBirdAlpha;\n' + shader.fragmentShader;

    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\n float rim = length(gl_PointCoord-.5)*2.; if(rim>=1.) discard;',
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      'diffuseColor.a *= vBirdAlpha * (1. - smoothstep(.35, 1., rim));\n #include <opaque_fragment>',
    );
  };
  material.customProgramCacheKey = () => 'jewel-bird-plume-v5';
  function points(parent, vertices, colors) {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(vertices, 3).setUsage(THREE.DynamicDrawUsage),
    );
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const p = new THREE.Points(g, material);

    p.userData.styleAnimated = true;
    // Stable density sampling while the particle positions flow through the plume.
    g.setAttribute(
      'styleSeed',
      new THREE.Float32BufferAttribute(
        Array.from({ length: vertices.length / 3 }, (_, i) => ((i * 73) % 997) / 997),
        1,
      ),
    );
    p.frustumCulled = false;
    clouds.push({ points: p, rest: Float32Array.from(vertices) });
    parent.add(p);
    return p;
  }
  function body(center, scale, count, base, accent) {
    const positions = [],
      colors = [],
      a = new THREE.Color(base),
      b = new THREE.Color(accent);
    for (let i = 0; i < count; i++) {
      const y = 1 - (2 * (i + 0.5)) / count,
        r = Math.sqrt(1 - y * y),
        theta = i * 2.39996323;
      positions.push(
        center[0] + Math.cos(theta) * r * scale[0],
        center[1] + y * scale[1],
        center[2] + Math.sin(theta) * r * scale[2],
      );
      const c = a.clone().lerp(b, Math.max(0, y) * 0.8 + 0.15 * (0.5 + 0.5 * Math.sin(theta * 3)));
      colors.push(c.r, c.g, c.b);
    }
    points(bird, positions, colors);
  }
  body([0, 0, 0], [0.16, 0.17, 0.43], 700, '#123e59', '#1bd8bd');
  body([0, 0.13, -0.4], [0.135, 0.145, 0.18], 320, '#1553a0', '#54f0d1');

  body([0, -0.08, -0.25], [0.13, 0.09, 0.19], 160, '#ff8644', '#ffe284');
  function feather(parent, base, tip, width, hue) {
    const pivot = new THREE.Group();
    pivot.position.set(...base);
    parent.add(pivot);
    const vertices = [],
      colors = [],
      colour = new THREE.Color();
    const dx = tip[0] - base[0],
      dy = tip[1] - base[1],
      dz = tip[2] - base[2],
      length = Math.hypot(dx, dz);
    for (let row = 0; row < 30; row++) {
      const t = (row + 0.5) / 30,
        w = width * Math.pow(Math.sin(Math.PI * t), 0.65);
      for (let col = -2; col <= 2; col++) {
        const jitter = Math.sin(row * 73.31 + col * 13.17);
        const cross = (col + jitter * 0.4) / 2;
        vertices.push(
          dx * t - (dz / length) * w * cross,
          dy * t + 0.045 * Math.sin(Math.PI * t) + 0.025 * Math.abs(cross),
          dz * t + (dx / length) * w * cross,
        );
        // Turquoise roots, sapphire/violet tips; warm gold shafts.
        colour.setHSL(hue + t * 0.16, 0.8, 0.4 + 0.18 * Math.sin(Math.PI * t));
        if (col === 0) colour.set('#e9cb83');
        colors.push(colour.r, colour.g, colour.b);
      }
    }
    points(pivot, vertices, colors);
    return pivot;
  }
  const wings = [],
    feathers = [];
  for (const side of [-1, 1]) {
    const shoulder = new THREE.Group();
    shoulder.position.set(side * 0.1, 0.02, -0.12);
    bird.add(shoulder);
    wings.push(shoulder);
    for (let i = 0; i < 12; i++) {
      const t = i / 11;
      const f = feather(
        shoulder,
        [side * (0.03 + t * 0.52), 0, t * 0.2],
        [side * (0.42 + t * 0.85), -0.02, 0.52 - t * 0.3],
        0.065,
        0.46 + t * 0.025,
      );
      feathers.push({ pivot: f, side, t });
    }
  }
  const tails = [];
  for (let i = -2; i <= 2; i++)
    tails.push(
      feather(
        bird,
        [i * 0.035, -0.025, 0.3],
        [i * 0.15, -0.13, 1.35 + (2 - Math.abs(i)) * 0.22],
        0.06,
        0.49 + Math.abs(i) * 0.025,
      ),
    );
  const plume = createBirdPlume(bird, clouds);
  const trails = createBirdTrails(
    bird,
    clouds.map((cloud) => cloud.points),
    clouds.map((cloud) => cloud.rest),
  );
  let phase = 0,
    rate = 4;
  bird.visible = false;
  return {
    bird,
    wings,
    resetTrails() {
      trails.reset();
      plume.reset();
    },
    animate(dt, moving) {
      dt = Math.min(Math.max(dt, 0), 0.05);
      rate += ((moving ? 4.8 : 3.2) - rate) * (1 - Math.exp(-3 * dt));
      phase += dt * rate;

      const beat = Math.sin(phase);
      wings.forEach((wing, i) => {
        const side = i === 0 ? -1 : 1;
        wing.rotation.z = side * (0.12 + beat * 0.38);
        wing.rotation.y = side * (0.08 + Math.cos(phase) * 0.12);
        wing.rotation.x = Math.cos(phase - 0.3) * 0.08;
      });
      for (const { pivot, side, t } of feathers) {
        pivot.rotation.z = side * Math.sin(phase - 0.4 - t * 0.45) * 0.16 * t;
        pivot.rotation.y = side * (0.5 + 0.5 * Math.cos(phase)) * t * 0.15;
      }
      tails.forEach((tail, i) => {
        tail.rotation.x = Math.sin(phase - 0.9 - i * 0.12) * 0.065;
        tail.rotation.y = Math.sin(phase * 0.55 - i * 0.25) * 0.04;
      });
      plume.update(dt);
      trails.update(dt);
    },
    dispose() {
      trails.dispose();
      clouds.forEach(({ points }) => points.geometry.dispose());
      material.dispose();
      bird.removeFromParent();
    },
  };
}
