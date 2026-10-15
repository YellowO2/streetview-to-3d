"""Gather and download the panoramas along a traced route, without the GPU."""
from ui.prepare_corridor.ready_status import ready_status


def prepare_corridor(route, *, prepare):
    """The Corridor prepare (the engine's panos) gathers along route, and what to say."""
    corridor = prepare(*route)
    status = ready_status(corridor)
    return corridor, status
