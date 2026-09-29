import * as THREE from 'three';

// The scene's water (postprocess/water.py, water.json): each body one flat
// shape at its own level, a few hundred triangles, over land that goes on
// under it -- the shore is wherever the two cross, nothing cut. Turquoise,
// light and clear near the shore (water.json's "shore": metres from dry
// land), deeper out, soft moving highlights, fading into the haze far
// off. No reflection of the scene, no render target. One material per
// scene, its time moved on by tickWater.
const materials = new Set();

function waterMaterial(shore) {
  const size = shore?.size || 1;
  const metres = new Uint8Array(size * size).fill(255);
  if (shore?.metres?.length === size * size) metres.set(shore.metres);
  const texture = new THREE.DataTexture(
    metres,
    size,
    size,
    THREE.RedFormat,
    THREE.UnsignedByteType,
  );
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  const material = new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    transparent: true,
    depthWrite: false,
    uniforms: {
      time: { value: 0 },
      shore: { value: texture },
      shoreLo: { value: shore ? shore.lo : 0 },
      shoreSpan: { value: shore ? shore.cell * size : 1 },
    },
    vertexShader: `varying vec3 world;
      void main() { world = (modelMatrix * vec4(position,1.)).xyz;
        gl_Position = projectionMatrix * viewMatrix * vec4(world,1.); }`,
    fragmentShader: `uniform float time, shoreLo, shoreSpan; uniform sampler2D shore; varying vec3 world;
      void main() {
        vec2 en = vec2(world.x, -world.z);                       // east, north
        float fromShore = texture2D(shore, (en - shoreLo) / shoreSpan).r * 255.;
        float deep = smoothstep(0., 14., fromShore);
        vec2 p = en / 7.;
        float t = time * .5;
        float a = sin(p.x * 1.7 + sin(p.y * 1.3 + t) * 1.6 + t);
        float b = sin(p.y * 1.9 + sin(p.x * 1.1 - t * .7) * 1.6 - t * .8);
        float away = length(cameraPosition - world);
        float glint = smoothstep(.55, .95, a * b) * (1. - smoothstep(60., 400., away));
        vec3 colour = mix(vec3(.56, .9, .8), vec3(.2, .62, .66), deep);
        vec3 v = normalize(cameraPosition - world);
        colour = mix(colour, vec3(.8, .95, .98), .35 * pow(1. - abs(v.y), 3.));   // the sky, low
        colour += vec3(.9, 1., .95) * glint * .22;
        colour = mix(colour, vec3(.84, .93, .94), smoothstep(250., 1100., away)); // into the haze
        float alpha = mix(.4, .94, deep);
        gl_FragColor = vec4(colour, alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
  });
  materials.add(material);
  return material;
}

export function tickWater(dt) {
  for (const m of materials) m.uniforms.time.value += Math.min(dt, 0.1);
}

export function disposeWater(material) {
  if (!materials.delete(material)) return; // its other surfaces already did
  material.uniforms.shore.value.dispose();
  material.dispose();
}

// water.json's surfaces, east/north metres from the scene's centre, as
// meshes in the viewer's frame (x east, y up, z south: the points' flip),
// sharing one material.
export function waterSurfaces(data) {
  const surfaces = (data?.surfaces || []).filter(
    (s) => Number.isFinite(s.level) && s.outer?.length >= 3,
  );
  if (!surfaces.length) return [];
  const material = waterMaterial(data.shore);
  const ring = (points) => points.map(([e, n]) => new THREE.Vector2(e, n));
  return surfaces.map((s) => {
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
    return mesh;
  });
}
