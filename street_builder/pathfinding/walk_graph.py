"""Search build_graph's candidate graph from start to end (GPU, not built yet).

Plan: prefer edges near a ~10m target hop, not the biggest jump available.
Fall back to smaller/larger candidates only if the target-size hop fails.
"""
