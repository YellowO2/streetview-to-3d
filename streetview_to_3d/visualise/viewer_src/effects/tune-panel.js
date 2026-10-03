// Uniforms for tuning a look by eye. With ?tune in the URL each gets a slider in a panel;
// "Copy" puts the values on the clipboard to paste back as defaults.
let panels = 0;

// numbers: { name: [value, min, max] } -> { name: { value } }
export function tunable(title, numbers) {
  const knobs = Object.fromEntries(Object.entries(numbers).map(([k, [v]]) => [k, { value: v }]));
  if (/[?&]tune\b/.test(globalThis.location?.search || '') && globalThis.document)
    panel(title, numbers, knobs);
  return knobs;
}

function panel(title, numbers, knobs) {
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
  const copy = document.createElement('button');
  copy.textContent = 'Copy';
  copy.style.marginTop = '6px';
  copy.onclick = () => {
    const values = Object.fromEntries(Object.entries(knobs).map(([k, u]) => [k, u.value]));
    navigator.clipboard?.writeText(JSON.stringify(values, null, 1));
  };
  box.append(copy);
  document.body.append(box);
}
