// Local file discovery is isolated from rendering and interaction state.
export function fileEntries(files) {
  return [...files].map((file) => ({ file, path: file.webkitRelativePath || file.name }));
}
export function resolveEntries(entries) {
  const manifests = entries.filter((e) => e.path.split('/').pop() === 'scene.json');
  if (manifests.length > 1) throw Error('Choose one scene folder at a time.');
  if (manifests.length) {
    const manifest = manifests[0],
      root = manifest.path.slice(0, -10),
      files = new Map(entries.map((e) => [e.path, e.file]));
    return {
      source: manifest.file,
      name: root.split('/').filter(Boolean).pop() || 'Scene',
      resolve: (path) => {
        const file = files.get(root + path);
        if (!file) throw Error(`Missing ${path}. Open the whole scene folder.`);
        return file;
      },
    };
  }
  if (entries.length === 1 && /\.ply$/i.test(entries[0].path))
    return { source: entries[0].file, name: entries[0].file.name };
  throw Error('Choose a single PLY, or scene.json together with its PLY files.');
}
async function walk(entry, prefix = '') {
  if (entry.isFile)
    return [{ file: await new Promise((r, j) => entry.file(r, j)), path: prefix + entry.name }];
  if (!entry.isDirectory) return [];
  const reader = entry.createReader(),
    result = [];
  for (;;) {
    const batch = await new Promise((r, j) => reader.readEntries(r, j));
    if (!batch.length) break;
    for (const child of batch) result.push(...(await walk(child, prefix + entry.name + '/')));
  }
  return result;
}
export async function droppedFiles(transfer) {
  // Capture entries synchronously, before the browser invalidates DataTransfer.
  const entries = [...transfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean),
    fallback = fileEntries(transfer.files),
    files = [];
  for (const entry of entries) files.push(...(await walk(entry)));
  return entries.length ? files : fallback;
}
