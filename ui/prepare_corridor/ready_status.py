"""Say how many panoramas are ready, and what to press next."""


def ready_status(corridor):
    """The status line once corridor is prepared."""
    n = sum(len(c) for g in corridor.date_graphs for c in g["dot_candidates"].values())
    return f"<p>{n} panoramas ready. Now click button 2.</p>"
