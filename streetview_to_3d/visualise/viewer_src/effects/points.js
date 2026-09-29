import { Vector3 } from 'three';

// Patch the existing material once; never replace geometry or alter exported positions.
const patched = new WeakMap();
export function pointMotion(object) {
  const material = object.material;
  if (patched.has(material)) return patched.get(material);
  const stableSeed = !!object.geometry.getAttribute('styleSeed');
  const uniforms = {
    styleDensity: { value: 1 },
    stylePointScale: { value: 1 },
    styleRound: { value: 0 },
    styleTime: { value: 0 },
    styleFloat: { value: 0 },
    styleScan: { value: 0 },
    styleRadius: { value: 1 },
    styleLook: { value: 1 },
    styleCenter: { value: new Vector3() },
    styleReveal: { value: 1 },
  };
  const before = material.onBeforeCompile,
    key = material.customProgramCacheKey.bind(material);
  const originalKey = key();
  material.onBeforeCompile = (shader, renderer) => {
    before.call(material, shader, renderer);
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader =
      `
      ${stableSeed ? 'attribute float styleSeed;' : ''}
      uniform float styleTime, styleFloat, styleLook, styleDensity, stylePointScale;
      uniform vec3 styleCenter;
      varying vec3 stylePosition;
    ` + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      '#include <begin_vertex>',
      `
      #include <begin_vertex>
      stylePosition = (modelMatrix * vec4(position, 1.0)).xyz - styleCenter;
      float phase = ${stableSeed ? 'styleSeed' : 'fract(sin(dot(position, vec3(12.9898,78.233,37.719))) * 43758.5453)'} * 6.2831853;
      vec3 drift = vec3(sin(styleTime*.55+phase)*.45, sin(styleTime*.8+phase)*.65,
        cos(styleTime*.5+phase)*.45);
      transformed += drift * styleLook * .004 * styleFloat;
    `,
    );
    shader.vertexShader = shader.vertexShader.replace(
      '#include <logdepthbuf_vertex>',
      `
      gl_PointSize *= stylePointScale * (1.0 + sin(styleTime*.8+phase)*.14*styleFloat);
      if (phase / 6.2831853 >= styleDensity) { gl_Position = vec4(2.,2.,2.,1.); gl_PointSize = 0.; }
      #include <logdepthbuf_vertex>
    `,
    );
    shader.fragmentShader =
      `
      varying vec3 stylePosition;
      ${stableSeed ? 'attribute float styleSeed;' : ''}
      uniform float styleTime, styleRadius, styleLook, styleScan, styleReveal, styleRound;
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <clipping_planes_fragment>',
      `
      #include <clipping_planes_fragment>
      if (styleRound > .5 && length(gl_PointCoord - .5) > .5) discard;
      if (styleReveal < .999 && length(stylePosition) > styleRadius * styleReveal * 1.5) discard;
    `,
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      `
      float wave = pow(.5+.5*sin(length(stylePosition.xz)/styleLook*16.0-styleTime*.8),18.0)*styleScan;
      outgoingLight = mix(outgoingLight, vec3(.65,.86,.76), wave*.25);
      #include <opaque_fragment>
    `,
    );
  };
  material.customProgramCacheKey = () => originalKey + ':viewer-point-motion-v3:' + stableSeed;
  material.needsUpdate = true;
  patched.set(material, uniforms);
  return uniforms;
}
