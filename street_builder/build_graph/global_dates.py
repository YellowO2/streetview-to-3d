"""Pick which real capture date covers which dot, for the WHOLE corridor
at once, from metadata alone (no GPU, no images downloaded) -- see
street_builder/ARCHITECTURE.md section 2.2 for where this fits.

Why this exists: the old per-chunk design let each chunk independently
rank its own top-N dates by LOCAL coverage. Two adjacent chunks can
easily land on different "best" dates even when both have real data on
a shared date that's merely their second- or third-best locally -- and
since a chunk's own boundary node then only has data on ITS chosen date,
not its neighbor's, cross-chunk bridging keeps failing on a real,
recurring pattern (confirmed empirically multiple times this session).
Picking dates for the corridor as a whole removes that mismatch by
construction: every chunk that's part of the same run gets the same
date at their shared boundary.

Also fixes a subtler problem: even within ONE chunk, walking a single
date in isolation leaves every dot that date doesn't cover as a genuine
gap (no flood-past-empty-dot fallback anymore -- see walk_graph.py). A
date's own coverage is frequently fragmented (several separate gaps),
each one splitting what should be one connected piece into several.
build_date_cover patches those gaps BEFORE the walk ever runs, by
borrowing whichever other date's own contiguous run best spans the gap
-- not just the missing dots, but the whole run it belongs to, since
that minimizes how many times the assignment switches dates along the
corridor (each switch is a cross-date pairwise test, riskier than a
same-date one).
"""
from services.geo import haversine_m


def dots_with_date(buckets, date):
    """{dot_index, ...} -- every dot that has at least one real candidate
    for this specific date."""
    return {i for i, bucket in buckets.items() if any(n["date"] == date for n in bucket)}


def connected_components(dot_set, adjacency):
    """Splits dot_set into its maximal connected components, walking only
    through adjacency edges that stay inside dot_set. Returns a list of
    sets. Pure graph bookkeeping, no metadata/GPU involved."""
    remaining = set(dot_set)
    components = []
    while remaining:
        seed = next(iter(remaining))
        stack = [seed]
        seen = {seed}
        while stack:
            d = stack.pop()
            for nb in adjacency.get(d, []):
                if nb in remaining and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(seen)
        remaining -= seen
    return components


def build_date_cover(points, adjacency, buckets, ranked_dates):
    """Assigns each reachable dot (has >=1 real candidate, for ANY date)
    to exactly one date, minimizing how many times the assignment
    switches date along the corridor -- see this module's own docstring
    for why that matters more than just "most dots covered."

    Algorithm: take the best-ranked date's own coverage as the starting
    assignment (usually fragmented into several disconnected runs, since
    no single date covers everything). For each remaining gap (a maximal
    connected run of still-unassigned reachable dots), find whichever
    OTHER date's own natural contiguous run (computed independently, not
    restricted to just the gap) overlaps the gap the most -- then adopt
    that date for the run's ENTIRE span, not just the gap dots, even
    overwriting dots the current assignment already had. That's the
    "take the whole c-d-e-f-h stretch, not just the xd-xe gap" behavior:
    it collapses what would've been several short alternating-date
    stretches into fewer, longer same-date ones. Repeats until no gap can
    be improved further. A dot with real candidates but on a date whose
    run never gets picked simply stays covered by whichever date DID win
    its region -- every reachable dot ends up assigned to SOME date as
    long as at least one date reaches it at all.

    Returns {dot_index: date}. Dots with zero real candidates for ANY
    date are absent -- nothing to assign, same as today's behavior (the
    walk just never reaches a dot with nothing there)."""
    date_components = {date: connected_components(dots_with_date(buckets, date), adjacency) for date in ranked_dates}

    reachable = set()
    for comps in date_components.values():
        for c in comps:
            reachable |= c
    if not reachable:
        return {}

    assigned = {}
    if ranked_dates:
        best_date = ranked_dates[0]
        for comp in date_components[best_date]:
            for d in comp:
                assigned[d] = best_date

    while True:
        unassigned = reachable - set(assigned)
        if not unassigned:
            break
        gaps = connected_components(unassigned, adjacency)

        improved = False
        for gap in gaps:
            best_choice = None  # (overlap_size, date, component)
            for date, comps in date_components.items():
                for comp in comps:
                    overlap = len(comp & gap)
                    if overlap == 0:
                        continue
                    if best_choice is None or overlap > best_choice[0]:
                        best_choice = (overlap, date, comp)
            if best_choice is None:
                continue  # this gap has zero real candidates on any date -- stays unassigned
            _, date, comp = best_choice
            for d in comp:
                assigned[d] = date
            improved = True

        if not improved:
            break  # remaining gaps are genuinely uncoverable by any date

    return assigned


def split_cover_into_chunks(points, adjacency, cover, chunk_size=20):
    """Split a whole-corridor date cover into chunks of roughly
    chunk_size dots each, analogous to
    map_selection.candidates.split_into_chunks but grown over the
    GLOBAL date-cover's dot graph instead of the raw selection graph --
    a chunk's own local BFS only ever steps into a neighbor dot that's
    both unassigned AND on the SAME assigned date as the chunk's seed,
    so every chunk is single-date BY CONSTRUCTION.

    That's the actual fix over the old flow: chunks used to get cut
    geographically first (chunk_size nodes off the raw selection graph)
    and only look up dates afterward, so a single chunk could straddle a
    date seam internally -- relying on the walk's own self-bridge pass
    (see pipeline_runner.py) to patch it back together inside one GPU
    call. Cutting along date-cover region boundaries instead means a
    cross-date seam only ever happens AT a chunk boundary, exactly where
    cross-chunk bridging (join_segments.bridge_incremental_gpu) already
    handles it.

    Returns (chunks, known_adjacent_chunk_pairs).
    chunks: [{"chunk_id": str, "dots": [dot_index, ...], "date": str,
    "start": (lat, lon), "goals": [(lat, lon), ...]}, ...] -- dots[0] is
    always the chunk's own start.
    known_adjacent_chunk_pairs: [(chunk_id_a, chunk_id_b), ...] -- every
    pair of chunks connected by at least one real adjacency edge, same
    date or not (a different-date pair is exactly a real seam -- still
    needs bridging, same as before)."""
    unassigned = set(cover)
    raw_chunks = []  # [[dots, date], ...]
    for seed in sorted(cover):
        if seed not in unassigned:
            continue
        date = cover[seed]
        chunk_dots = []
        queue = [seed]
        queued = {seed}
        while queue and len(chunk_dots) < chunk_size:
            d = queue.pop(0)
            if d not in unassigned:
                continue
            chunk_dots.append(d)
            unassigned.discard(d)
            for nb in adjacency.get(d, []):
                if nb in unassigned and nb not in queued and cover.get(nb) == date:
                    queued.add(nb)
                    queue.append(nb)
        if chunk_dots:
            raw_chunks.append([chunk_dots, date])

    # A chunk with <2 dots can't stand alone (prepare_pathfind_from_cover_chunk
    # needs a start + at least 1 goal) -- fold it into a same-date
    # neighbor chunk if one exists, else any adjacent chunk at all (a
    # genuine, unavoidable single-dot seam -- same last-resort fallback
    # split_into_chunks uses for the raw graph).
    dot_to_chunk_idx = {d: i for i, (dots, _) in enumerate(raw_chunks) for d in dots}
    for i, (dots, date) in enumerate(raw_chunks):
        if len(dots) >= 2 or not dots:
            continue
        target_same_date, target_any = None, None
        for d in dots:
            for nb in adjacency.get(d, []):
                j = dot_to_chunk_idx.get(nb)
                if j is None or j == i:
                    continue
                if raw_chunks[j][1] == date:
                    target_same_date = j
                    break
                target_any = j if target_any is None else target_any
            if target_same_date is not None:
                break
        target = target_same_date if target_same_date is not None else target_any
        if target is not None:
            raw_chunks[target][0].extend(dots)
            for d in dots:
                dot_to_chunk_idx[d] = target
            raw_chunks[i][0] = []
        else:
            print(f"split_cover_into_chunks: dropping isolated dot(s) with no real adjacency to anything: {dots}")
            raw_chunks[i][0] = []
    raw_chunks = [(dots, date) for dots, date in raw_chunks if dots]

    dot_to_chunk_id = {}
    for i, (dots, _) in enumerate(raw_chunks):
        chunk_id = f"chunk{i}"
        for d in dots:
            dot_to_chunk_id[d] = chunk_id

    chunks = []
    for i, (dots, date) in enumerate(raw_chunks):
        chunk_points = [points[d] for d in dots]
        chunks.append({
            "chunk_id": f"chunk{i}",
            "dots": dots,
            "date": date,
            "start": tuple(chunk_points[0]),
            "goals": [tuple(p) for p in chunk_points[1:]],
        })

    adjacent_pairs = set()
    for d, ca in dot_to_chunk_id.items():
        for nb in adjacency.get(d, []):
            cb = dot_to_chunk_id.get(nb)
            if cb and cb != ca:
                adjacent_pairs.add(frozenset((ca, cb)))
    known_adjacent_chunk_pairs = [tuple(sorted(pair)) for pair in adjacent_pairs]

    return chunks, known_adjacent_chunk_pairs


def build_global_date_graphs(points, adjacency, buckets, ranked_dates, cover=None):
    """Converts a build_date_cover assignment into the SAME
    {dot_index: [candidates]} shape build_corridor_graphs' date_graphs
    already use -- one dict, not one per date, since every dot now has
    exactly one assigned date rather than living in N isolated parallel
    graphs. cover: pass an already-computed build_date_cover result to
    reuse it (e.g. after inspecting it); computed fresh otherwise.

    Returns dot_candidates: {dot_index: [{key, source, id, lat, lon,
    date}, ...]} -- each dot's bucket, capped to just its ASSIGNED date's
    own candidates (still every real candidate for that date at that dot,
    not just one -- callers cap further per their own top-K rule, same
    as build_graph.build_corridor_graphs' TOP_PANOS_PER_DOT)."""
    if cover is None:
        cover = build_date_cover(points, adjacency, buckets, ranked_dates)
    dot_candidates = {}
    for dot, date in cover.items():
        same_date = [n for n in buckets.get(dot, []) if n["date"] == date]
        if same_date:
            dot_candidates[dot] = same_date
    return dot_candidates
