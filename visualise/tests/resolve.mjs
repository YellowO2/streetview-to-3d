export function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('@viewer/'))
    return nextResolve(
      new URL(`../viewer_src/${specifier.slice(8)}.js`, import.meta.url).href,
      context,
    );
  return nextResolve(specifier, context);
}
