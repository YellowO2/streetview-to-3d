import * as THREE from 'three';

// Three low-detail terrain rings anchored to the capture, not the screen.
// A camera-centred dome supplies a rotation-aware horizon without a texture;
// in the plain view it is a clear blue sky behind every placed scene.
export function createEnvironment(scene) {
  const terrain = new THREE.Group();
  terrain.name = 'Style environment';
  const resources = [];
  for (let layer = 0; layer < 3; layer++) {
    const positions = [],
      indices = [],
      segments = 192;
    const distance = 5 + layer * 5;
    for (let ring = 0; ring < 3; ring++) {
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        const ridge =
          0.9 +
          0.45 * Math.sin(angle * 3 + layer * 1.7) +
          0.25 * Math.sin(angle * 7 - layer) +
          0.12 * Math.cos(angle * 13);
        const radius = distance + ring * 2.5;
        const height = ring === 1 ? ridge * (1 + layer * 0.6) : -0.4;
        positions.push(Math.cos(angle) * radius, height, Math.sin(angle) * radius);
        if (ring < 2 && i < segments) {
          const a = ring * (segments + 1) + i,
            b = a + segments + 1;
          indices.push(a, b, a + 1, b, b + 1, a + 1);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const material = new THREE.MeshBasicMaterial({
      color: ['#788a8e', '#93a5a8', '#b2c0c1'][layer],
      side: THREE.DoubleSide,
    });
    terrain.add(new THREE.Mesh(geometry, material));
    resources.push(geometry, material);
  }
  const groundGeometry = new THREE.CircleGeometry(80, 96);
  const groundMaterial = new THREE.MeshBasicMaterial({ color: '#b8c5c5', side: THREE.DoubleSide });
  const ground = new THREE.Mesh(groundGeometry, groundMaterial);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.45;
  terrain.add(ground);
  resources.push(groundGeometry, groundMaterial);
  const skyMaterial = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: { anime: { value: 0 }, blue: { value: 0 } },
    vertexShader: `
      varying vec3 direction;
      void main() {
        direction = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 direction;
      uniform float anime, blue;
      void main() {
        float h = normalize(direction).y;
        vec3 horizon = mix(mix(vec3(.58,.66,.67), vec3(.78,.94,1.0), anime), vec3(.80,.88,.96), blue);
        vec3 zenith = mix(mix(vec3(.30,.43,.49), vec3(.13,.55,1.0), anime), vec3(.33,.55,.86), blue);
        vec3 below = mix(mix(vec3(.47,.56,.57), vec3(.65,.84,.91), anime), vec3(.64,.68,.70), blue);
        vec3 colour = mix(horizon, zenith, smoothstep(0.0, .9, h));
        colour = mix(colour, below, (1.0 - smoothstep(-.6, 0.0, h)));
        gl_FragColor = vec4(colour, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }
    `,
  });
  const skyGeometry = new THREE.SphereGeometry(1, 32, 16);
  const sky = new THREE.Mesh(skyGeometry, skyMaterial);
  sky.frustumCulled = false;
  sky.renderOrder = -100;
  scene.add(terrain, sky);
  terrain.visible = sky.visible = false;
  const fog = new THREE.Fog('#c7d3d4', 5, 30);
  return {
    configure(box, radius) {
      const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
      terrain.position.copy(center);
      terrain.position.y = box.isEmpty() ? -radius * 0.1 : box.min.y - radius * 0.05;
      terrain.scale.setScalar(radius);
      fog.near = radius * 2.5;
      fog.far = radius * 22;
    },
    // plainSky: a placed scene is open, so the plain view gets the blue sky
    update(style, camera, enabled = true, plainSky = false) {
      sky.visible = style === 'original' ? plainSky : enabled;
      skyMaterial.uniforms.blue.value = style === 'original' ? 1 : 0;
      terrain.visible = enabled && style === 'dither';
      scene.fog = terrain.visible ? fog : null;
      skyMaterial.uniforms.anime.value = style === 'anime' ? 1 : 0;
      sky.position.copy(camera.position);
      sky.scale.setScalar(camera.far * 0.5);
    },
    dispose() {
      scene.remove(terrain, sky);
      [...resources, skyGeometry, skyMaterial].forEach((resource) => resource.dispose());
      scene.fog = null;
    },
    terrain,
  };
}
