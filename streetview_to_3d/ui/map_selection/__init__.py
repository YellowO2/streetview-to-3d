"""The map: loading an area, clicking nodes, and turning that into a route.

The map runs in a sandboxed iframe, so a click cannot call Python directly
-- see tab.py for the postMessage bridge that carries it across.
"""
