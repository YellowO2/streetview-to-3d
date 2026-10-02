import { STYLE_DEFAULTS, normalizeStyle } from '@viewer/effects/presets';
const $ = (id) => document.getElementById(id);
export function createStyleControls(actions) {
  const saved = structuredClone(STYLE_DEFAULTS);
  const reducedMotion = !!document.defaultView?.matchMedia?.('(prefers-reduced-motion: reduce)')
    .matches;
  if (reducedMotion) Object.values(saved).forEach((preset) => (preset.floating = false));
  let current = 'original',
    splat = false,
    blocked = false,
    loaded = false;
  function value() {
    return {
      density: Number($('point-density').value),
      pointSize: 2 ** Number($('point-size').value),
      strength: Number($('style-strength').value),
      pixels: Number($('style-pixels').value),
      floating: $('style-floating').checked,
      amount: Number($('style-float').value),
      scan: $('style-scan').checked,
      atmosphere: $('style-atmosphere').checked,
    };
  }
  function render() {
    $('point-density').disabled = blocked || !loaded || splat;
    $('point-density-value').textContent = $('point-density').value + '%';
    const original = current === 'original';
    $('style-controls').hidden = original;
    $('style-pixels-label').hidden = current !== 'dither';
    $('style-point-controls').hidden = splat;
    $('style-splat-note').hidden = !splat;
    $('visual-style').disabled = blocked;
    for (const control of document.querySelectorAll(
      '#style-controls input, #style-controls button',
    ))
      control.disabled = blocked || !loaded || original;
    $('style-strength-value').textContent = Math.round($('style-strength').value * 100) + '%';
    $('style-pixels-value').textContent = $('style-pixels').value + 'px';
    $('style-float-value').textContent = Math.round($('style-float').value * 100) + '%';
  }
  function apply() {
    saved[current] = value();
    render();
    actions.style?.(current, saved[current]);
  }
  function select(next) {
    next = normalizeStyle(next);
    if (!(next in saved)) next = 'original';
    saved[current] = value();
    current = next;
    $('visual-style').value = next;
    const preset = saved[next];
    $('point-density').value = preset.density;
    $('point-size').value = Math.log2(preset.pointSize);
    $('point-size').oninput?.();
    $('style-strength').value = preset.strength;
    $('style-pixels').value = preset.pixels;
    $('style-floating').checked = preset.floating;
    $('style-float').value = preset.amount;
    $('style-scan').checked = preset.scan;
    $('style-atmosphere').checked = preset.atmosphere;
    render();
    actions.style?.(next, preset);
  }
  $('point-density').oninput = apply;
  $('visual-style').onchange = () => select($('visual-style').value);
  for (const input of document.querySelectorAll('#style-controls input')) input.oninput = apply;
  $('style-reveal').onclick = () => actions.reveal?.();
  $('style-rise').onclick = () => actions.demo?.('rise');
  $('style-swirl').onclick = () => actions.demo?.('swirl');
  $('style-gentle').onclick = () => actions.demo?.('gentle');
  return {
    select,
    render(store, disabled) {
      splat = !!store.splat;
      blocked = disabled;
      loaded = !!store.group;
      render();
    },
  };
}
