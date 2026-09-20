"""The Gradio app: pick an area on a map, run the pipeline, look at the result.

    tab.py            the three buttons -- prepare, reconstruct, place
    map_selection/    the map itself, and turning clicks into a selection
    viewers.py        the point-cloud viewer and download links

Nothing here computes anything; each button calls into the stage that does.
"""
