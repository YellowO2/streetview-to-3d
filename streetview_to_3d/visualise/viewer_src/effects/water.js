import * as THREE from 'three';

// A world-anchored opaque water surface. Analytic sky reflection, no scene render target.
export function createWater(scene) {
  const geometry = new THREE.PlaneGeometry(1, 1);
  geometry.rotateX(-Math.PI / 2);
  const material = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    uniforms: { time: { value: 0 }, scale: { value: 1 }, anchor: { value: new THREE.Vector3() } },
    vertexShader: `varying vec3 world;
      void main() { world = (modelMatrix * vec4(position,1.)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(world,1.); }`,
    fragmentShader: `uniform float time, scale; uniform vec3 anchor; varying vec3 world;
      void main() {
        vec2 p = (world.xz-anchor.xz)/scale;
        float t = time*.35;
        vec2 slope = vec2(cos(p.x*31.+p.y*13.+t)*.035 + cos(p.x*63.-p.y*27.-t*.8)*.018,
                          cos(p.x*13.+p.y*29.+t*.7)*.035 + sin(p.x*41.+p.y*53.+t)*.012);
        slope *= 1. - smoothstep(2.,12.,length(p));
        vec3 n = normalize(vec3(-slope.x,1.,-slope.y));
        vec3 v = normalize(cameraPosition-world);
        if (v.y < 0.) n = -n;
        vec3 reflected = reflect(-v,n);
        vec3 sky = mix(vec3(.78,.94,1.),vec3(.13,.55,1.),smoothstep(0.,.9,reflected.y));
        float fresnel = .035 + .965*pow(1.-max(dot(n,v),0.),5.);
        vec3 colour = mix(vec3(.025,.22,.26),sky,fresnel);
        vec3 sun = normalize(vec3(-.4,.65,.5));
        float glint = pow(max(dot(reflected,sun),0.),180.);
        colour += vec3(1.,.87,.61)*glint*.75;
        float distanceFade = smoothstep(10.,35.,length(p));
        colour = mix(colour,vec3(.78,.94,1.),distanceFade);
        gl_FragColor = vec4(colour,1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'Flood water';
  mesh.visible = false;
  scene.add(mesh);
  let base = 0,
    radius = 1,
    time = 0;
  return {
    mesh,
    configure(box, r) {
      radius = Math.max(r, 0.001);
      const center = box.isEmpty() ? new THREE.Vector3() : box.getCenter(new THREE.Vector3());
      mesh.position.copy(center);
      base = box.isEmpty() ? 0 : box.min.y;
      mesh.scale.setScalar(radius * 100);
      material.uniforms.scale.value = radius;
      material.uniforms.anchor.value.copy(center);
    },
    update(enabled, level, dt, paused = false) {
      mesh.visible = enabled;
      mesh.position.y = base + radius * level;
      if (enabled && !paused) time += Math.min(dt, 0.1);
      material.uniforms.time.value = time;
    },
    dispose() {
      mesh.removeFromParent();
      geometry.dispose();
      material.dispose();
    },
  };
}
