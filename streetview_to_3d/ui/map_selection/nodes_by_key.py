"""Look up the map's loaded nodes by their key."""


def nodes_by_key(state):
    """{key: node dict} of every node the map state has loaded."""
    return {n["key"]: n for n in state["nodes"]}
