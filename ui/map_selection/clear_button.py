"""Clear the selection back to its start node."""
from ui.map_selection.MapWidgets import MapWidgets
from ui.map_selection.tools.map_view import map_html, summary_markdown


def _handle_clear(state):
    # back to just the start node (which comes from the location, not a click)
    start = [state["nodes"][0]["key"]] if state.get("nodes") else []
    state = {**state, "selected": start, "selected_edges": []}
    return map_html(state), summary_markdown(state), state


def clear_button(widgets: MapWidgets):
    """Wire the Clear selection button."""
    widgets.clear.click(
        fn=_handle_clear,
        inputs=[widgets.state],
        outputs=[widgets.map_view, widgets.selection_view, widgets.state],
        api_name="handle_clear",
    )
