import * as THREE from 'three';

// Knobs for tuning a look by eye: each a uniform ({ value }), shared by
// whatever shaders take them. A page opened with ?tune shows them in a panel,
// top right, a slider each (numbers) or a colour picker (colours); "Copy"
// puts their values on the clipboard, to set as the defaults in the code.
let panels = 0;

// numbers: { name: [value, lowest, highest] }, colours: { name: '#hex' } ->
// { name: uniform } (a colour's a THREE.Color)
export function tunable(title, numbers, colours = {}) {
  const knobs = {
    ...Object.fromEntries(Object.entries(numbers).map(([k, [v]]) => [k, { value: v }])),
    ...Object.fromEntries(
      Object.entries(colours).map(([k, hex]) => [k, { value: new THREE.Color(hex) }]),
    ),
  };
  if (/[?&]tune\b/.test(globalThis.location?.search || '') && globalThis.document)
    panel(title, numbers, colours, knobs);
  return knobs;
}

function panel(title, numbers, colours, knobs) {
  const box = document.createElement('div');
  box.style.cssText =
    `position:fixed;top:${80 + 40 * panels++}px;right:12px;z-index:50;background:#1b232cee;` +
    'color:#dde;padding:8px 10px;font:12px system-ui;border-radius:8px;max-height:80vh;' +
    'overflow:auto;width:250px';
  box.innerHTML = `<b>${title}</b>`;
  for (const [k, [v, lo, hi]] of Object.entries(numbers)) {
    const row = document.createElement('label');
    row.style.cssText =
      'display:grid;grid-template-columns:80px 1fr 44px;gap:6px;align-items:center';
    row.innerHTML = `<span>${k}</span><input type="range" min="${lo}" max="${hi}" step="any" value="${v}"><span>${v}</span>`;
    const [input, shown] = [row.children[1], row.children[2]];
    input.oninput = () => {
      knobs[k].value = Number(input.value);
      shown.textContent = Number(input.value).toPrecision(3);
    };
    box.append(row);
  }
  for (const [k, hex] of Object.entries(colours)) {
    const row = document.createElement('label');
    row.style.cssText =
      'display:flex;justify-content:space-between;align-items:center;margin-top:4px';
    row.innerHTML = `<span>${k}</span><input type="color" value="${hex}">`;
    row.children[1].oninput = (e) => knobs[k].value.set(e.target.value);
    box.append(row);
  }
  const copy = document.createElement('button');
  copy.textContent = 'Copy';
  copy.style.marginTop = '6px';
  copy.onclick = () => {
    const values = Object.fromEntries(
      Object.entries(knobs).map(([k, u]) => [
        k,
        u.value.isColor ? '#' + u.value.getHexString() : u.value,
      ]),
    );
    navigator.clipboard?.writeText(JSON.stringify(values, null, 1));
  };
  box.append(copy);
  document.body.append(box);
}
