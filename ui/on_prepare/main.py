"""Gather and download the panoramas along a traced route, without the GPU."""
from ui.on_prepare.ready_status import ready_status


def on_prepare(route, *, prepare):
    """The Corridor prepare (the engine's gather_panoramas) gathers along route, and what to say."""
    corridor = prepare(*route)
    status = ready_status(corridor)
    return corridor, status
