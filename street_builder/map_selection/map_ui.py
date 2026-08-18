"""Builds the Leaflet map HTML for the street-builder node picker.

Pure string-building — no Gradio here. The map runs inside a sandboxed
iframe (same pattern app.py already uses for its single-pano map/viewers),
so clicking a marker can't call back into Python directly; instead it
posts a window message that a page-level listener (injected via
gr.Blocks(head=...), see tab.py) relays into a hidden Gradio textbox.

Google-only: Street View's own coverage graph (pano.links) gives real
street topology. Rather than showing every node fetched for the loaded
area (a scattered, cluttered cloud), only the chain built so far plus
whichever nodes are directly linked to its current end get shown — those
are the only markers that make sense to click next anyway.
"""
import json

from viewers import iframe as _iframe

_SUGGESTED_COLOR = "#ff9800"
_SELECTED_COLOR = "#00c853"
_EDGE_COLOR = "#9aa0a6"

_MESSAGE_TYPE = "street_builder_node_click"


def build_picker_map(lat, lon, nodes, edges, selected_keys, selected_edges, zoom=17, view=None):
    """nodes: list of {key, source, id, lat, lon, heading} for the whole loaded
    area (only the relevant subset -- see module docstring -- actually gets
    rendered). edges: list of (key_a, key_b) pairs from Street View's coverage
    graph. selected_keys: node keys selected so far (a graph, not necessarily
    a simple chain -- branches and loops are both possible). selected_edges:
    (key_a, key_b) pairs actually confirmed by a click, drawn as the
    highlighted graph instead of assuming selected_keys' list order traces a
    single path (it doesn't, once branches exist).
    view: optional (lat, lon, zoom) to open the map at instead of (lat, lon, zoom)
    above -- used to restore whatever pan/zoom the map was at before a click
    triggered a rebuild, since Gradio replaces the iframe wholesale on every
    update and a fresh Leaflet map otherwise has no memory of that.
    """
    view_lat, view_lon, view_zoom = view if view else (lat, lon, zoom)

    by_key = {n["key"]: n for n in nodes}
    selected_set = set(selected_keys)
    # Frontier = neighbors of ANY selected node, not just the most recent one
    # -- an earlier node's unvisited branch stays visible/clickable even
    # after the chain has moved past it.
    next_keys = set()
    for a, b in edges:
        if a in selected_set and b not in selected_set:
            next_keys.add(b)
        elif b in selected_set and a not in selected_set:
            next_keys.add(a)

    visible_keys = selected_set | next_keys
    visible_nodes = [by_key[k] for k in visible_keys if k in by_key]
    visible_edges = [e for e in edges if e[0] in visible_keys and e[1] in visible_keys]

    nodes_json = json.dumps(visible_nodes)
    edges_json = json.dumps(visible_edges)
    selected_json = json.dumps(selected_keys)
    selected_edges_json = json.dumps(selected_edges)

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{margin:0;height:100%;width:100%}}
.sb-empty{{position:absolute;top:8px;left:50%;transform:translateX(-50%);z-index:1000;
background:rgba(0,0,0,.75);color:#fff;font:12px sans-serif;padding:4px 10px;border-radius:6px}}
.sb-order{{background:{_SELECTED_COLOR};color:#fff;border:none;font-weight:600}}
.sb-order::before{{border-top-color:{_SELECTED_COLOR}}}</style>
</head><body>
<div id="map"></div>
<script>
// NODES/EDGES here are already filtered down to the selected chain plus
// its current end's direct neighbors -- see build_picker_map in map_ui.py.
var NODES = {nodes_json};
var EDGES = {edges_json};
var SELECTED = {selected_json};
var SELECTED_EDGES = {selected_edges_json};
var SUGGESTED_COLOR = {json.dumps(_SUGGESTED_COLOR)};
var SELECTED_COLOR = {json.dumps(_SELECTED_COLOR)};
var EDGE_COLOR = {json.dumps(_EDGE_COLOR)};

var m = L.map('map').setView([{view_lat},{view_lon}], {view_zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);

if (NODES.length === 0) {{
  var empty = document.createElement('div');
  empty.className = 'sb-empty';
  empty.textContent = 'No Street View coverage found here.';
  document.body.appendChild(empty);
}}

var byKey = {{}};
NODES.forEach(function(n) {{ byKey[n.key] = n; }});

EDGES.forEach(function(e) {{
  var a = byKey[e[0]], b = byKey[e[1]];
  if (!a || !b) return;
  L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {{color: EDGE_COLOR, weight: 2, opacity: 0.7}}).addTo(m);
}});

NODES.forEach(function(n) {{
  var order = SELECTED.indexOf(n.key);
  var isSelected = order !== -1;
  var color = isSelected ? SELECTED_COLOR : SUGGESTED_COLOR;
  var marker = L.circleMarker([n.lat, n.lon], {{
    radius: isSelected ? 10 : 8,
    color: color,
    fillColor: color,
    fillOpacity: 0.85,
    weight: isSelected ? 3 : 2,
  }}).addTo(m);
  var label = n.id;
  if (isSelected) {{
    label = '#' + (order + 1) + ' · ' + label;
    marker.bindTooltip(String(order + 1), {{permanent: true, direction: 'top', className: 'sb-order'}});
  }} else {{
    label = 'next · ' + label;
  }}
  marker.bindPopup(label);
  marker.on('click', function() {{
    console.log('[street_builder] marker clicked, posting to parent:', n.key);
    var c = m.getCenter();
    window.parent.postMessage({{
      type: {json.dumps(_MESSAGE_TYPE)},
      key: n.key, source: n.source, id: n.id, lat: n.lat, lon: n.lon, heading: n.heading,
      view: {{lat: c.lat, lon: c.lng, zoom: m.getZoom()}},
      _ts: Date.now(),
    }}, '*');
  }});
}});

// Confirmed edges drawn individually (not one polyline through SELECTED's
// list order) -- correct for a branching or loop-closing graph, where
// list order doesn't trace a single path.
SELECTED_EDGES.forEach(function(e) {{
  var a = byKey[e[0]], b = byKey[e[1]];
  if (!a || !b) return;
  L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {{color: SELECTED_COLOR, weight: 4}}).addTo(m);
}});
</script>
</body></html>"""
    return _iframe(doc)
