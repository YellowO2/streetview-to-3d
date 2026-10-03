import { Vector3 } from 'three';
import { worldUniforms, WORLD_VERTEX, WORLD_FRAGMENT, cutPoint } from '@viewer/style/world-points';

// Patches a point material once for the styles' float, scan, density and wind sway;
// geometry and exported positions are never touched.
const SWAY_M = 0.12, // most a tree point moves in the wind
  WIND = [0.8, 0.6]; // wind direction (x, z)
const patched = new WeakMap();

// the uniforms the styles drive (controller.js); shared by blocks.js so points near DA3 move alike
export const styleUniforms = () => ({
  styleDensity: { value: 1 },
  stylePointScale: { value: 1 },
  styleRound: { value: 0 },
  styleTime: { value: 0 },
  styleFloat: { value: 0 },
  styleScan: { value: 0 },
  styleLook: { value: 1 },
  styleCenter: { value: new Vector3() },
});

export function pointMotion(object) {
  const material = object.material;
  if (patched.has(material)) return patched.get(material);
  const stableSeed = !!object.geometry.getAttribute('styleSeed');
  const swaying = !!object.geometry.getAttribute('sway'); // 0-255 per point (fill)
  const uniforms = styleUniforms();
  // points with their own motion (water) are left alone
  if (object.userData.ownMotion) {
    patched.set(material, uniforms);
    return uniforms;
  }
  // world points take the demos, shots and characters; the bird, gun and shots don't
  const demoed = !object.userData.styleAnimated;
  const before = material.onBeforeCompile,
    key = material.customProgramCacheKey.bind(material);
  const originalKey = key();
  material.onBeforeCompile = (shader, renderer) => {
    before.call(material, shader, renderer);
    Object.assign(shader.uniforms, uniforms, demoed ? worldUniforms : {});
    shader.vertexShader =
      `
      ${demoed ? WORLD_VERTEX : ''}
      ${stableSeed ? 'attribute float styleSeed;' : ''}
      ${swaying ? 'attribute float sway;' : ''}
      uniform float styleTime, styleFloat, styleLook, styleDensity, stylePointScale;
      uniform vec3 styleCenter;
      varying vec3 stylePosition;
      varying float styleGlyph;
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
      styleGlyph = phase / 6.2831853;
      ${
        swaying
          ? `float gust = dot(position.xz, vec2(.07, .045)) - styleTime * 1.1;
      transformed.xz += vec2(${WIND}) * sway / 255. * ${SWAY_M.toFixed(3)} * (sin(gust) + .35 * sin(gust * 2.3 + 1.7));`
          : ''
      }
      float demoIn = 1.;
      ${demoed ? 'if (shotAway(stylePosition + styleCenter)) demoIn = 0.;' : ''}
    `,
    );
    if (demoed)
      shader.vertexShader = shader.vertexShader.replace(
        '#include <project_vertex>',
        `
      #include <project_vertex>
      float demoHere;
      mvPosition = viewMatrix * vec4(demoed((modelMatrix * vec4(transformed, 1.)).xyz,
        phase / 6.2831853, demoHere), 1.);
      gl_Position = projectionMatrix * mvPosition;
      demoIn = min(demoIn, demoHere);
    `,
      );
    shader.vertexShader = shader.vertexShader.replace(
      '#include <logdepthbuf_vertex>',
      `
      gl_PointSize *= stylePointScale * (1.0 + sin(styleTime*.8+phase)*.14*styleFloat);
      ${demoed ? 'gl_PointSize = worldSize(gl_PointSize, phase / 6.2831853, phase / 6.2831853);' : ''}
      if (phase / 6.2831853 >= styleDensity || demoIn < .5 || gl_PointSize == 0.) { gl_Position = vec4(2.,2.,2.,1.); gl_PointSize = 0.; }
      #include <logdepthbuf_vertex>
    `,
    );
    shader.fragmentShader =
      `
      varying vec3 stylePosition;
      varying float styleGlyph;
      uniform float styleTime, styleLook, styleScan, styleRound;
      ${demoed ? WORLD_FRAGMENT : ''}
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <clipping_planes_fragment>',
      `
      #include <clipping_planes_fragment>
      ${cutPoint(demoed && 'styleGlyph', 'styleRound > .5 && length(gl_PointCoord - .5) > .5')}
    `,
    );
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <opaque_fragment>',
      `
      float wave = pow(.5+.5*sin(length(stylePosition.xz)/styleLook*16.0-styleTime*.8),18.0)*styleScan;
      outgoingLight = mix(outgoingLight, vec3(.65,.86,.76), wave*.25);
      outgoingLight *= pointShade;
      #include <opaque_fragment>
    `,
    );
  };
  material.customProgramCacheKey = () =>
    originalKey + ':viewer-point-motion-v11:' + stableSeed + demoed + swaying;
  material.needsUpdate = true;
  patched.set(material, uniforms);
  return uniforms;
}
