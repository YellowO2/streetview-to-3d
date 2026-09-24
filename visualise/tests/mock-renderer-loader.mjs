export function resolve(specifier, context, nextResolve) {
  if (specifier === 'three')
    return nextResolve(new URL('./mock-renderer.mjs', import.meta.url).href, context);
  return nextResolve(specifier, context);
}
