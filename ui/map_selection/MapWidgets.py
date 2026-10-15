"""The map section's widgets, which its controls wire to."""
from dataclasses import dataclass

import gradio as gr


@dataclass(frozen=True)
class MapWidgets:
    state: gr.State             # the loaded nodes and the selection (map_widgets._empty_state)
    area_input: gr.Textbox      # a Google Maps URL or lat,lon
    radius_input: gr.Textbox    # metres round it, for Expand
    expand: gr.Button
    map_view: gr.HTML
    bridge: gr.Textbox          # relays the map iframe's messages (BRIDGE_HEAD_SCRIPT)
    selection_view: gr.Markdown
    clear: gr.Button
