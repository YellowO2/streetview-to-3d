// Full application logic runs against real Three.js maths/controls and jsdom.
// Only GPU drawing is substituted; this is not a visual/browser test.
export * from '../node_modules/three/build/three.module.js';
export class WebGLRenderer {
  constructor() {
    this.domElement = document.createElement('canvas');
  }
  setPixelRatio(value) {
    this.pixelRatio = value;
  }
  getPixelRatio() {
    return this.pixelRatio || 1;
  }
  setSize(width, height) {
    this.width = width;
    this.height = height;
  }
  getSize(target) {
    return target.set(this.width || 1, this.height || 1);
  }
  setAnimationLoop(callback) {
    this.loop = callback;
  }
  render() {}
}
