// Small helpers shared by the viewer's shaders and point builders.

export const SUN = [3, 5, 4]; // direction toward the sun, viewer frame

// GLSL number formatters
export const f = (x) => x.toFixed(4);
export const v3 = (v) => `vec3(${v.map(f).join(', ')})`;

// Deterministic pseudo-random value in [0, 1) for a grid cell.
export function hash(i, j, k) {
  const s = Math.sin(i * 12.9898 + j * 78.233 + k * 37.719) * 43758.5453;
  return s - Math.floor(s);
}

// count points spread evenly over the unit sphere (golden-angle spiral), as [x, y, z]
export function fibonacciSphere(count) {
  return Array.from({ length: count }, (_, i) => {
    const y = 1 - (2 * (i + 0.5)) / count,
      ring = Math.sqrt(1 - y * y),
      theta = i * 2.39996323;
    return [Math.cos(theta) * ring, y, Math.sin(theta) * ring];
  });
}

// GLSL for a PointsMaterial fragment: discards outside the round point; r is 0 at its centre, 1 at its edge
export const DISC = `float r = dot(gl_PointCoord * 2. - 1., gl_PointCoord * 2. - 1.);
  if (r > 1.) discard;`;
