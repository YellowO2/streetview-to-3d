// Scene.py compatibility and GPS math. No DOM or renderer dependencies.
export function scenePieces(data, threshold = 0) {
  const parent = data.nodes.map((_, i) => i);
  function find(i) {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  }
  for (const edge of data.edges) {
    const rates = [edge.keep_a, edge.keep_b].filter((k) => k && k[1]).map((k) => k[0] / k[1]);
    const confidence = rates.length ? Math.min(...rates) : 0;
    if (confidence >= threshold) parent[find(edge.a)] = find(edge.b);
  }
  const groups = new Map();
  data.nodes.forEach((node, i) => {
    if (node.position == null) return;
    const root = find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(i);
  });
  return [...groups.values()];
}
export function relativePath(path) {
  if (
    typeof path !== 'string' ||
    !path ||
    path.startsWith('/') ||
    path.includes('\\') ||
    path.includes(':') ||
    path.split('/').some((p) => !p || p === '..' || p === '.')
  ) {
    throw new Error('PLY paths must be relative to scene.json, without “..”.');
  }
  return path;
}
export function validateScene(data) {
  if (
    !data ||
    !Array.isArray(data.nodes) ||
    !Array.isArray(data.edges) ||
    !Array.isArray(data.center) ||
    data.center.length !== 2 ||
    !data.center.every(Number.isFinite)
  ) {
    throw new Error('Expected a scene.py scene: center, nodes and edges.');
  }
  data.nodes.forEach((node, i) => {
    if (!node || !node.pano) throw new Error(`Node ${i} has no panorama metadata.`);
    if (
      node.position != null &&
      (!Array.isArray(node.position) ||
        node.position.length !== 3 ||
        !node.position.every(Number.isFinite))
    )
      throw new Error(`Node ${i} has an invalid position.`);
    if (node.ply) relativePath(node.ply);
    if (node.transform != null) {
      const t = node.transform;
      if (
        !Array.isArray(t) ||
        t.length !== 4 ||
        !t.every((row) => Array.isArray(row) && row.length === 4 && row.every(Number.isFinite)) ||
        t[3].some((v, j) => Math.abs(v - (j === 3 ? 1 : 0)) > 1e-8)
      )
        throw new Error(`Node ${i} has an invalid 4×4 transform.`);
    }
  });
  for (const edge of data.edges) {
    if (
      !edge ||
      ![edge.a, edge.b].every((i) => Number.isInteger(i) && i >= 0 && i < data.nodes.length)
    )
      throw new Error('An edge references a missing node.');
    for (const k of [edge.keep_a, edge.keep_b]) {
      if (
        k != null &&
        (!Array.isArray(k) ||
          k.length !== 2 ||
          !k.every(Number.isFinite) ||
          k[0] < 0 ||
          k[1] < k[0])
      )
        throw new Error('An edge has invalid confidence counts.');
    }
  }
}
export function placementMode(data) {
  const indices = data.nodes.map((n, i) => (n.ply ? i : -1)).filter((i) => i >= 0);
  if (!indices.length) throw new Error('This scene has no PLY files.');
  if (indices.every((i) => data.nodes[i].transform != null)) return 'world';
  const groups = scenePieces(data);
  if (
    indices.every((i) => data.nodes[i].transform == null) &&
    groups.length === 1 &&
    indices.every((i) => groups[0].includes(i))
  )
    return 'raw';
  throw new Error(
    'This scene contains separate or partially placed pieces. Run postprocess.pipeline to save their transforms first.',
  );
}
export function gpsPlacement(data, scale) {
  const members = scenePieces(data)[0];
  const pairs = members.map((i) => {
    const n = data.nodes[i],
      lat = n.pano.lat,
      lon = n.pano.lon;
    if (![lat, lon].every(Number.isFinite)) throw new Error('GPS coordinates are missing.');
    return {
      src: [n.position[0] * scale, n.position[2] * scale],
      dst: [
        (lon - data.center[1]) * 111320 * Math.cos((data.center[0] * Math.PI) / 180),
        (lat - data.center[0]) * 111320,
      ],
    };
  });
  if (pairs.length < 2)
    throw new Error(
      'At least two camera positions are needed to fit heading. Use the Python alignment pipeline for this scene.',
    );
  const mean = (key) =>
    [0, 1].map((axis) => pairs.reduce((sum, p) => sum + p[key][axis], 0) / pairs.length);
  const a = mean('src'),
    b = mean('dst');
  let dot = 0,
    cross = 0,
    spread = 0;
  for (const p of pairs) {
    const x = p.src[0] - a[0],
      z = p.src[1] - a[1],
      e = p.dst[0] - b[0],
      n = p.dst[1] - b[1];
    dot += x * e + z * n;
    cross += x * n - z * e;
    spread += x * x + z * z;
  }
  if (spread < 1e-8 || Math.hypot(dot, cross) < 1e-8)
    throw new Error('The camera track is too short to determine heading.');
  const angle = Math.atan2(cross, dot),
    c = Math.cos(angle),
    s = Math.sin(angle);
  return [
    [scale * c, 0, -scale * s, b[0] - c * a[0] + s * a[1]],
    [0, scale, 0, 0],
    [scale * s, 0, scale * c, b[1] - s * a[0] - c * a[1]],
    [0, 0, 0, 1],
  ];
}
