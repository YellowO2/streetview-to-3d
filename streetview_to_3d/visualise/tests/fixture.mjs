import { readFileSync } from 'node:fs';
export function viewerTemplate() {
  const read = (name) => readFileSync(new URL(`../viewer_src/${name}`, import.meta.url), 'utf8');
  return read('template.html').replace('<!-- SCENE_MANAGER -->', read('scene-manager.html'));
}
