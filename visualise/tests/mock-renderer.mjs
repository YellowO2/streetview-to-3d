// Full application logic runs against real Three.js maths/controls and jsdom.
// Only GPU drawing is substituted; this is not a visual/browser test.
export * from '../node_modules/three/build/three.module.js';
export class WebGLRenderer {
  constructor() {
    this.domElement = document.createElement('canvas');
  }
  setPixelRatio() {}
  setSize() {}
  setAnimationLoop(callback) {
    this.loop = callback;
  }
  render() {}
}
