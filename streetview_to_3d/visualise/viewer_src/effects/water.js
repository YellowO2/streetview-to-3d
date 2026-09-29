import * as THREE from 'three';

// The scene's water (postprocess/water.py, water.json): each body one flat
// shape at its own level, a few hundred triangles where points would be
// tens of thousands. Analytic sky reflection, soft ripples and a sun glint;
// no reflection of the scene, no render target. One material for all, its
// time moved on by tickWater.
const material = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  uniforms: { time: { value: 0 } },
  vertexShader: `varying vec3 world;
    void main() { world = (modelMatrix * vec4(position,1.)).xyz;
      gl_Position = projectionMatrix * viewMatrix * vec4(world,1.); }`,
  fragmentShader: `uniform float time; varying vec3 world;
    void main() {
      vec2 p = world.xz / 6.;
      float t = time*.35;
      vec2 slope = vec2(cos(p.x*3.1+p.y*1.3+t)*.035 + cos(p.x*6.3-p.y*2.7-t*.8)*.018,
                        cos(p.x*1.3+p.y*2.9+t*.7)*.035 + sin(p.x*4.1+p.y*5.3+t)*.012);
      float away = length(cameraPosition-world);
      slope *= 1. - smoothstep(40.,250.,away);   // ripples finer than a pixel only shimmer
      vec3 n = normalize(vec3(-slope.x,1.,-slope.y));
      vec3 v = normalize(cameraPosition-world);
      if (v.y < 0.) n = -n;
      vec3 reflected = reflect(-v,n);
      vec3 sky = mix(vec3(.78,.94,1.),vec3(.25,.6,.95),smoothstep(0.,.9,reflected.y));
      float fresnel = .06 + .94*pow(1.-max(dot(n,v),0.),5.);
      vec3 colour = mix(vec3(.08,.33,.42),sky,fresnel);
      vec3 sun = normalize(vec3(-.4,.65,.5));
      colour += vec3(1.,.87,.61)*pow(max(dot(reflected,sun),0.),180.)*.75;
      gl_FragColor = vec4(colour,1.);
      #include <tonemapping_fragment>
      #include <colorspace_fragment>
    }`,
});

export function tickWater(dt) {
  material.uniforms.time.value += Math.min(dt, 0.1);
}

// water.json's surfaces, east/north metres from the scene's centre, as
// meshes in the viewer's frame (x east, y up, z south: the points' flip).
export function waterSurfaces(data) {
  const ring = (points) => points.map(([e, n]) => new THREE.Vector2(e, n));
  return (data?.surfaces || []).flatMap((s) => {
    if (!Number.isFinite(s.level) || !(s.outer?.length >= 3)) return [];
    const shape = new THREE.Shape(ring(s.outer));
    shape.holes = (s.holes || []).filter((h) => h.length >= 3).map((h) => new THREE.Path(ring(h)));
    const geometry = new THREE.ShapeGeometry(shape);
    // shape (east, north) -> viewer (east, level, -north)
    geometry.rotateX(Math.PI / 2);
    geometry.scale(1, 1, -1);
    geometry.translate(0, s.level, 0);
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'Water';
    return [mesh];
  });
}
