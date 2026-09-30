import * as THREE from 'three';

// A camera-centred sky dome; terrain belongs to the scene.
export function createEnvironment(scene) {
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
  scene.add(sky);
  sky.visible = false;
  return {
    // plainSky: a placed scene is open, so the plain view gets the blue sky;
    update(style, camera, enabled = true, plainSky = false) {
      sky.visible = style === 'original' ? plainSky : enabled;
      skyMaterial.uniforms.blue.value = style === 'original' ? 1 : 0;
      skyMaterial.uniforms.anime.value = style === 'anime' ? 1 : 0;
      sky.position.copy(camera.position);
      sky.scale.setScalar(camera.far * 0.5);
    },
    dispose() {
      scene.remove(sky);
      [skyGeometry, skyMaterial].forEach((resource) => resource.dispose());
    },
  };
}
