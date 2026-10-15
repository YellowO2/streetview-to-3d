"""Lay out the map section's widgets, empty."""
import json

import gradio as gr

from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import MESSAGE_TYPES, map_html, summary_markdown

BRIDGE_ELEM_ID = "map_bridge"

# hidden by CSS, not visible=False, which can leave it out of the DOM
BRIDGE_CSS = f"#{BRIDGE_ELEM_ID} {{ position: fixed !important; width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; overflow: hidden !important; }}"

BRIDGE_HEAD_SCRIPT = f"""
<script>
window.addEventListener('message', function(ev) {{
  if (!ev.data || {json.dumps(list(MESSAGE_TYPES))}.indexOf(ev.data.type) === -1) return;
  var el = document.querySelector('#{BRIDGE_ELEM_ID} textarea, #{BRIDGE_ELEM_ID} input');
  if (!el) {{
    console.error('[map] bridge element #{BRIDGE_ELEM_ID} not found in DOM');
    return;
  }}
  el.value = JSON.stringify(ev.data);
  el.dispatchEvent(new Event('input', {{bubbles: true}}));
  el.dispatchEvent(new Event('change', {{bubbles: true}}));
}});
</script>
"""


def _empty_state():
    return {"lat": None, "lon": None, "nodes": [], "edges": [], "selected": [], "selected_edges": [], "view": None,
            "radius_m": None, "preview_center": None, "area": None}


def map_widgets():
    """Build the map section's widgets, inside a gr.Blocks, showing an empty map."""
    state = gr.State(_empty_state())

    # one row: the location loads as it is pasted, so the only button is Expand
    with gr.Row(equal_height=True):
        area_input = gr.Textbox(
            placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
            show_label=False,
            container=False,
            scale=4,
        )
        radius_input = gr.Textbox(
            placeholder="Radius in metres (e.g. 50)",
            show_label=False,
            container=False,
            scale=2,
        )
        expand = gr.Button("Expand area", scale=1, min_width=100)

    map_view = gr.HTML(map_html(_empty_state()), elem_classes="no-pad")
    # hidden by BRIDGE_CSS
    bridge = gr.Textbox(elem_id=BRIDGE_ELEM_ID, show_label=False, container=False)

    with gr.Row(equal_height=True):
        selection_view = gr.Markdown(summary_markdown(_empty_state()), elem_id="map_selection")
        with gr.Column(scale=0, min_width=140):
            clear = gr.Button("Clear selection")

    return MapWidgets(state, area_input, radius_input, expand, map_view, bridge, selection_view, clear)
