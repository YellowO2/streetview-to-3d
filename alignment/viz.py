"""Reusable 2D scatter visualizer for GPS-alignment work: node positions
colored by an arbitrary group field, with an optional grey reference layer
underneath (e.g. true GPS shape), a hoverable/clickable legend, and an
optional click-to-select mode.

Used for every "does this positioning look right" / "which segment am I
pointing at" check in this project (per-chunk, per-date, per-island,
before/after drift-correction, gap-region picking, ...) so the
visualization code only needs writing once -- this used to be two
separate near-duplicate scripts (this one, and a click-to-select-only
variant) before they were merged here as one opt-in mode.

Usage:
    python -m alignment.viz --in /tmp/gps_pieces_15c.json \
        --pos-field fitted_en --ref-field real_en --group-field island \
        --title "Per-island fit vs GPS" --out /tmp/viz.html

    # click-to-select mode (whole group toggles on click, persisted via
    # the artifact's own db capability -- declare capabilities: {db: {}}
    # when publishing):
    python -m alignment.viz --in /tmp/gap_data.json --selectable \
        --pos-field fitted_en --group-field island --out /tmp/picker.html
"""
import argparse
import colorsys
import json


def group_color(i, n):
    h = (i / n) % 1.0
    r, g, b = colorsys.hls_to_rgb(h, 0.52, 0.62)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --surface-1: #fcfcfb; --page: #f9f9f7; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --text-muted: #898781; --gridline: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --accent: #eb6834; --real-dot: #6d6c66; --select: #2f6fed;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
      --text-muted: #898781; --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --accent: #d95926; --real-dot: #9c9a92; --select: #6ea0ff;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --text-muted: #898781; --gridline: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --accent: #d95926; --real-dot: #9c9a92; --select: #6ea0ff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--page); color: var(--text-primary);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .app { display: flex; flex-direction: column; height: 100vh; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); background: var(--surface-1);
    display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: 0.01em; }
  .subtitle { color: var(--text-secondary); font-size: 12.5px; margin-top: 2px; max-width: 640px; }
  .body { flex: 1; display: flex; min-height: 0; }
  .canvas-wrap { flex: 1; position: relative; overflow: hidden; background: var(--surface-1); }
  svg { width: 100%; height: 100%; display: block; cursor: grab; }
  svg.dragging { cursor: grabbing; }
  .sidebar { width: 300px; border-left: 1px solid var(--border); background: var(--surface-1);
    padding: 16px; overflow-y: auto; flex-shrink: 0; display: flex; flex-direction: column; gap: 10px; }
  .sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin: 0 0 10px; }
  .legend-note { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--text-secondary); font-size: 12.5px; margin-bottom: 6px; cursor: pointer; padding: 3px 4px; border-radius: 5px; }
  .legend-note:hover { background: var(--gridline); }
  .legend-note .l { display: flex; align-items: center; gap: 8px; }
  .legend-note .med { font: 600 11.5px ui-monospace, monospace; color: var(--text-muted); font-variant-numeric: tabular-nums; }
  .dot-sample { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  .hint { color: var(--text-muted); font-size: 11.5px; margin-top: 14px; line-height: 1.5; }
  .tooltip { position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
    padding: 7px 10px; border-radius: 6px; font-size: 12px; line-height: 1.5; z-index: 10;
    opacity: 0; transition: opacity 0.08s; max-width: 240px; font-family: ui-monospace, monospace; }
  .tooltip .k { opacity: 0.6; }
  .zoom-controls { position: absolute; right: 16px; bottom: 16px; display: flex; flex-direction: column; gap: 6px; }
  .zoom-btn { width: 28px; height: 28px; border-radius: 6px; border: 1px solid var(--border); background: var(--surface-1);
    color: var(--text-primary); font-size: 15px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
  .zoom-btn:hover { background: var(--gridline); }
  .btn { border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
    border-radius: 6px; padding: 7px 12px; font-size: 12.5px; cursor: pointer; }
  .btn:hover { background: var(--gridline); }
  .count { font: 600 20px ui-monospace, monospace; }
  .sel-list { font: 11px/1.55 ui-monospace, monospace; color: var(--text-secondary);
    flex: 1 1 auto; min-height: 180px; overflow-y: auto;
    background: var(--page); border: 1px solid var(--border); border-radius: 6px; padding: 8px;
    white-space: pre-wrap; overflow-wrap: anywhere; user-select: text; }
  .status { font-size: 11.5px; color: var(--text-muted); }
</style>
</head>
<body>
<div class="app">
  <header>
    <div>
      <h1>__TITLE__</h1>
      <div class="subtitle">__SUBTITLE__</div>
    </div>
  </header>
  <div class="body">
    <div class="canvas-wrap">
      <svg id="svg" viewBox="0 0 1000 1000"></svg>
      <div class="zoom-controls">
        <button class="zoom-btn" id="zoom-in">+</button>
        <button class="zoom-btn" id="zoom-out">&#8722;</button>
        <button class="zoom-btn" id="zoom-reset" title="reset">&#8635;</button>
      </div>
    </div>
    <div class="sidebar">
      __SELECT_PANEL__
      __REF_LEGEND__
      <h2 style="margin-top:20px">__GROUP_LABEL__ (median error)</h2>
      <div id="date-legend"></div>
      <div class="hint">__HINT__</div>
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __DATA__;
const DATES = __DATES__;
const DATE_COLORS = __COLORS__;
const HAS_REF = __HAS_REF__;
const SELECTABLE = __SELECTABLE__;

const svg = document.getElementById('svg');
const tooltip = document.getElementById('tooltip');

const xs = DATA.flatMap(d => HAS_REF ? [d.pos[0], d.ref[0]] : [d.pos[0]]);
const ys = DATA.flatMap(d => HAS_REF ? [d.pos[1], d.ref[1]] : [d.pos[1]]);
const minX = Math.min(...xs), maxX = Math.max(...xs);
const minY = Math.min(...ys), maxY = Math.max(...ys);
const pad = Math.max(maxX - minX, maxY - minY) * 0.08 + 1;
const vbX = minX - pad, vbY = -(maxY + pad);
const vbW = (maxX - minX) + pad * 2, vbH = (maxY - minY) + pad * 2;
svg.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`);

const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }
const root = el('g', {id: 'root'});
svg.appendChild(root);

const pxPerUnit = svg.clientWidth / vbW || 1;
const dotR = Math.max(1.4, 3.2 / pxPerUnit);
const lineW = Math.max(0.3, 0.6 / pxPerUnit);
const ringR = dotR * 1.9;

if (HAS_REF) {
  const refGroup = el('g', {id: 'reference'});
  root.appendChild(refGroup);
  DATA.forEach(d => {
    refGroup.appendChild(el('circle', {cx: d.ref[0], cy: -d.ref[1], r: dotR * 0.6, fill: 'var(--real-dot)', opacity: 0.45}));
  });
}

// -- selection state (only meaningful when SELECTABLE) --
const selected = new Set();
const byGroup = new Map();
DATA.forEach(d => { if (!byGroup.has(d.group)) byGroup.set(d.group, []); byGroup.get(d.group).push(d.key); });
let db = null;
let writeTimer = null;
const dotEls = new Map();

DATA.forEach(d => {
  const [fx, fy] = [d.pos[0], -d.pos[1]];
  const color = DATE_COLORS[d.group_idx % DATE_COLORS.length];
  let dot;
  if (SELECTABLE) {
    const g = el('g', {});
    const ring = el('circle', {cx: fx, cy: fy, r: ringR, fill: 'none', stroke: 'var(--select)', 'stroke-width': dotR * 0.5, opacity: 0});
    dot = el('circle', {cx: fx, cy: fy, r: dotR, fill: color, stroke: 'var(--surface-1)', 'stroke-width': lineW * 0.6, style: 'cursor:pointer'});
    g.appendChild(ring);
    g.appendChild(dot);
    dotEls.set(d.key, ring);
    root.appendChild(g);
    dot.addEventListener('click', (ev) => { ev.stopPropagation(); toggleGroup(d.group); });
  } else {
    dot = el('circle', {cx: fx, cy: fy, r: dotR, fill: color, 'data-date': d.group_idx, stroke: 'var(--surface-1)', 'stroke-width': lineW * 0.6});
    root.appendChild(dot);
  }
  dot.addEventListener('mouseenter', (ev) => showTooltip(ev, d));
  dot.addEventListener('mousemove', positionTooltip);
  dot.addEventListener('mouseleave', hideTooltip);
});

function showTooltip(ev, d) {
  tooltip.innerHTML = `<div><span class="k">group</span> ${d.group}</div><div><span class="k">label</span> ${d.label}</div><div><span class="k">error</span> ${(d.residual_m ?? 0).toFixed(1)}m</div>`;
  tooltip.style.opacity = 1;
  positionTooltip(ev);
}
function positionTooltip(ev) { tooltip.style.left = (ev.clientX + 14) + 'px'; tooltip.style.top = (ev.clientY + 14) + 'px'; }
function hideTooltip() { tooltip.style.opacity = 0; }

const legend = document.getElementById('date-legend');
DATES.forEach((info, i) => {
  const row = document.createElement('div');
  row.className = 'legend-note';
  row.dataset.date = i;
  row.innerHTML = `<span class="l"><span class="dot-sample" style="background:${DATE_COLORS[i % DATE_COLORS.length]}"></span>${info.group} (${info.n}n)</span><span class="med">${info.median.toFixed(0)}m</span>`;
  if (!SELECTABLE) {
    row.addEventListener('click', () => {
      document.querySelectorAll('[data-date]').forEach(node => {
        if (node.classList.contains('legend-note')) return;
        node.style.opacity = (String(node.dataset.date) === String(i)) ? '1' : '0.05';
      });
    });
  } else {
    row.addEventListener('click', () => toggleGroup(info.group));
  }
  legend.appendChild(row);
});
if (!SELECTABLE) {
  svg.addEventListener('click', () => document.querySelectorAll('[data-date]').forEach(n => { if (!n.classList.contains('legend-note')) n.style.opacity = ''; }));
}

// -- selection logic (SELECTABLE mode only) --
function toggleGroup(group) {
  const keys = byGroup.get(group) || [];
  const allSelected = keys.every(k => selected.has(k));
  keys.forEach(k => {
    if (allSelected) selected.delete(k); else selected.add(k);
    dotEls.get(k).setAttribute('opacity', selected.has(k) ? 1 : 0);
  });
  renderSelection();
  scheduleSave();
}

function renderSelection() {
  const countEl = document.getElementById('count');
  const selListEl = document.getElementById('sel-list');
  if (!countEl) return;
  const selectedGroups = [...byGroup.keys()].filter(g => byGroup.get(g).every(k => selected.has(k)) && byGroup.get(g).some(k => selected.has(k)));
  countEl.textContent = selectedGroups.length;
  if (!selectedGroups.length) { selListEl.textContent = '(none)'; return; }
  const chunkOf = new Map(DATA.map(d => [d.key, d.chunk_id || '?']));
  selListEl.textContent = selectedGroups.map(g => {
    const keys = byGroup.get(g);
    const chunks = [...new Set(keys.map(k => chunkOf.get(k)))].sort();
    return `${g}  (${keys.length} node(s), chunk(s): ${chunks.join(', ')})`;
  }).join('\n');
}

function scheduleSave() {
  clearTimeout(writeTimer);
  writeTimer = setTimeout(saveSelection, 400);
}

async function saveSelection() {
  const statusEl = document.getElementById('status');
  if (!db) { if (statusEl) statusEl.textContent = 'db unavailable -- selection not persisted, use the list at left manually.'; return; }
  const items = DATA.filter(d => selected.has(d.key)).map(d => ({key: d.key, chunk_id: d.chunk_id, group: d.group}));
  try {
    await db.doc('selection/current').set({items, updated_at: new Date().toISOString()});
    if (statusEl) statusEl.textContent = `saved (${items.length})`;
  } catch (e) {
    if (statusEl) statusEl.textContent = 'save failed: ' + (e && e.code || e);
  }
}

if (SELECTABLE) {
  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    selected.forEach(k => dotEls.get(k).setAttribute('opacity', 0));
    selected.clear();
    renderSelection();
    scheduleSave();
  });

  (async () => {
    const statusEl = document.getElementById('status');
    if (!window.claude || !window.claude.use) { if (statusEl) statusEl.textContent = 'db unavailable in this view.'; return; }
    db = await window.claude.use('db');
    if (!db) { if (statusEl) statusEl.textContent = 'db unavailable in this view.'; return; }
    try {
      const snap = await db.doc('selection/current').get();
      if (snap.exists) {
        const data = snap.data();
        (data.items || []).forEach(it => {
          selected.add(it.key);
          const ringEl = dotEls.get(it.key);
          if (ringEl) ringEl.setAttribute('opacity', 1);
        });
        renderSelection();
        if (statusEl) statusEl.textContent = `loaded (${selected.size})`;
      }
    } catch (e) {
      if (statusEl) statusEl.textContent = 'load failed: ' + (e && e.code || e);
    }
  })();
}

let view = {x: vbX, y: vbY, w: vbW, h: vbH};
function applyView() { svg.setAttribute('viewBox', `${view.x} ${view.y} ${view.w} ${view.h}`); }
let dragging = false, lastX = 0, lastY = 0;
svg.addEventListener('mousedown', (ev) => { dragging = true; lastX = ev.clientX; lastY = ev.clientY; svg.classList.add('dragging'); });
window.addEventListener('mouseup', () => { dragging = false; svg.classList.remove('dragging'); });
window.addEventListener('mousemove', (ev) => {
  if (!dragging) return;
  const scale = view.w / svg.clientWidth;
  view.x -= (ev.clientX - lastX) * scale; view.y -= (ev.clientY - lastY) * scale;
  lastX = ev.clientX; lastY = ev.clientY; applyView();
});
function zoom(factor, cx, cy) {
  view.x = cx - (cx - view.x) * factor; view.y = cy - (cy - view.y) * factor;
  view.w *= factor; view.h *= factor; applyView();
}
svg.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  const rect = svg.getBoundingClientRect();
  const px = view.x + (ev.clientX - rect.left) / rect.width * view.w;
  const py = view.y + (ev.clientY - rect.top) / rect.height * view.h;
  zoom(ev.deltaY > 0 ? 1.12 : 0.89, px, py);
}, {passive: false});
document.getElementById('zoom-in').addEventListener('click', () => zoom(0.8, view.x + view.w / 2, view.y + view.h / 2));
document.getElementById('zoom-out').addEventListener('click', () => zoom(1.25, view.x + view.w / 2, view.y + view.h / 2));
document.getElementById('zoom-reset').addEventListener('click', () => { view = {x: vbX, y: vbY, w: vbW, h: vbH}; applyView(); });
</script>
</body>
</html>
"""

SELECT_PANEL = """<div><div class="count" id="count">0</div><div class="hint">segment(s) selected</div></div>
      <button class="btn" id="clear-btn">Clear selection</button>
      <h2 style="margin-top:10px">Selected segments</h2>
      <div class="sel-list" id="sel-list">(none)</div>
      <div class="status" id="status"></div>"""


def build(data_path, pos_field, ref_field, group_field, label_field, residual_field,
          title, subtitle, out_path, highlight_group=None, highlight_color="#ffffff",
          selectable=False, chunk_field="chunk_id"):
    with open(data_path) as f:
        raw = json.load(f)

    groups = sorted({str(d[group_field]) for d in raw})
    n_groups = len(groups)

    points = []
    for d in raw:
        p = {
            "pos": d[pos_field],
            "group": d[group_field],
            "label": d.get(label_field, ""),
            "residual_m": d.get(residual_field, 0),
            "key": d.get(label_field, ""),
            "chunk_id": d.get(chunk_field),
        }
        if ref_field:
            p["ref"] = d[ref_field]
        points.append(p)

    group_info = []
    for g in groups:
        members = [p for p in points if str(p["group"]) == g]
        residuals = sorted(p["residual_m"] for p in members)
        median = residuals[len(residuals) // 2] if residuals else 0
        group_info.append({"group": g, "n": len(members), "median": median})
    group_info.sort(key=lambda x: -x["median"])
    order = {info["group"]: i for i, info in enumerate(group_info)}
    for p in points:
        p["group_idx"] = order[str(p["group"])]

    colors = [group_color(i, n_groups) for i in range(n_groups)]
    if highlight_group is not None and highlight_group in order:
        colors[order[highlight_group]] = highlight_color

    ref_legend = (
        '<h2>Reference</h2>'
        '<div class="legend-note" style="cursor:default"><span class="l">'
        '<span class="dot-sample" style="background:var(--real-dot)"></span> true GPS shape</span></div>'
    ) if ref_field else ""

    hint = ("Click any dot to select its whole segment, click again to unselect. "
            "Drag to pan, scroll/+/− to zoom.") if selectable else \
           "Drag to pan, scroll/pinch or +/− to zoom. Click a group to isolate it, empty space to clear."

    html = TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__GROUP_LABEL__", group_field)
    html = html.replace("__REF_LEGEND__", ref_legend)
    html = html.replace("__SELECT_PANEL__", SELECT_PANEL if selectable else "")
    html = html.replace("__HINT__", hint)
    html = html.replace("__DATA__", json.dumps(points))
    html = html.replace("__DATES__", json.dumps(group_info))
    html = html.replace("__COLORS__", json.dumps(colors))
    html = html.replace("__HAS_REF__", "true" if ref_field else "false")
    html = html.replace("__SELECTABLE__", "true" if selectable else "false")

    with open(out_path, "w") as f:
        f.write(html)
    print(f"wrote {out_path} ({len(html) / 1e3:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--pos-field", default="fitted_en", help="key holding [x,y] to plot")
    parser.add_argument("--ref-field", default="real_en", help="key holding [x,y] grey reference layer, empty to disable")
    parser.add_argument("--group-field", default="island", help="key to color/group/legend/select by")
    parser.add_argument("--label-field", default="key", help="key for tooltip label / selection key")
    parser.add_argument("--chunk-field", default="chunk_id", help="key for chunk id shown in selection list")
    parser.add_argument("--residual-field", default="residual_m")
    parser.add_argument("--title", default="GPS alignment viz")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--highlight-group", default=None, help="force this group value to a fixed color regardless of sort order")
    parser.add_argument("--highlight-color", default="#ffffff")
    parser.add_argument("--selectable", action="store_true",
                         help="click-to-select-whole-group mode instead of click-to-isolate; persists via the artifact's own db capability")
    args = parser.parse_args()
    build(args.in_path, args.pos_field, args.ref_field or None, args.group_field,
          args.label_field, args.residual_field, args.title, args.subtitle, args.out,
          args.highlight_group, args.highlight_color, args.selectable, args.chunk_field)


if __name__ == "__main__":
    main()
