"""Split the walking graph into corridors -- maximal one-road stretches.

A corridor is a maximal run of degree-2 dots -- the graph's own definition
of "no choice about where to go next" -- capped at each end by a junction
(degree >= 3) or a dead end (degree 1). Junction dots belong to every
corridor that meets there, so corridors overlap by exactly one dot and the
network can be reassembled from them later.

This is a road inventory, NOT a way to divide a reconstruction into
alignable runs. On NTU it yields 622 corridors with a median length of
20 m, while a single reconstructed piece was measured spanning up to 28 of
them and 444 corridors hold only one piece. Pieces are regions grown by
GPS-fit merging, so they straddle junctions freely; the road and the piece
are different shapes, and no cut of this graph makes one contain the other.
"""
import argparse
import json
import math

from paths import FETCHED_GRAPH
from postprocess.gps_fit.fit import real_en

# How straight two corridors must meet to be one road through a junction.
# A road can bend at a junction, but a right-angle turn into a side street
# is a different road, and pairing those would thread one frame around a
# corner where "left" flips sides.
STRAIGHT_ENOUGH_DEG = 50.0


def _metres(points):
    """lat/lon -> the shared GLOBAL_ORIGIN metre frame.

    The same frame `load_pieces` puts camera positions in, so road
    polylines and piece cameras are directly comparable without any
    further transform.
    """
    return [real_en(lat, lon) for lat, lon in points]


def _length(xy, nodes):
    return sum(math.dist(xy[a], xy[b]) for a, b in zip(nodes, nodes[1:]))


def decompose(adjacency):
    """[[node ids along a corridor], ...] covering every edge exactly once.

    Corridors are returned as ordered walks. Two corridors may share an end
    dot (a junction); no edge appears in two corridors.
    """
    adj = {int(k): [int(v) for v in vs] for k, vs in adjacency.items()}
    is_junction = {n: len(vs) != 2 for n, vs in adj.items()}
    seen = set()          # undirected edges already claimed

    def edge(a, b):
        return (a, b) if a < b else (b, a)

    corridors = []
    for start in sorted(n for n in adj if is_junction[n]):
        for first in adj[start]:
            if edge(start, first) in seen:
                continue
            walk = [start, first]
            seen.add(edge(start, first))
            while not is_junction[walk[-1]]:
                prev, here = walk[-2], walk[-1]
                nxt = [n for n in adj[here] if n != prev]
                if not nxt or edge(here, nxt[0]) in seen:
                    break
                seen.add(edge(here, nxt[0]))
                walk.append(nxt[0])
            corridors.append(walk)

    # rings of degree-2 dots touch no junction, so the sweep above misses
    # them entirely; each becomes one corridor that returns to its start
    for start in sorted(adj):
        if is_junction[start]:
            continue
        if any(edge(start, n) not in seen for n in adj[start]):
            walk = [start]
            while True:
                here = walk[-1]
                nxt = [n for n in adj[here] if edge(here, n) not in seen]
                if not nxt:
                    break
                seen.add(edge(here, nxt[0]))
                walk.append(nxt[0])
            corridors.append(walk)
    return corridors


def _heading(xy, a, b):
    return math.atan2(xy[b][1] - xy[a][1], xy[b][0] - xy[a][0])


def _turn(u, v):
    """How far a road bends when leaving along u and arriving along v.

    Both headings point OUTWARD from the shared junction, so continuing
    straight makes them opposite; the turn is the departure from that.
    """
    d = abs(u - v) % (2 * math.pi)
    return abs(math.pi - min(d, 2 * math.pi - d))


def roads(graph, straight_deg=STRAIGHT_ENOUGH_DEG):
    """Corridors chained through junctions into whole roads.

    A corridor stops at every junction, which cuts a single street into a
    dozen stubs -- too fine to define a road frame over. So at each
    junction the corridor ends are paired up by how nearly they continue
    each other, straightest pair first, and each chain becomes one road.

    Returns [[node ids along the road], ...] as ordered walks. Unlike
    corridors these are a road INVENTORY: a piece may lie along several,
    and each gets its own frame.
    """
    corridors = decompose(graph["adjacency"])
    xy = _metres(graph["points"])
    # heading leaving each corridor at each of its two ends
    out = [(_heading(xy, c[1], c[0]), _heading(xy, c[-2], c[-1]))
           for c in corridors]

    at = {}
    for i, c in enumerate(corridors):
        at.setdefault(c[0], []).append((i, 0))
        at.setdefault(c[-1], []).append((i, 1))

    parent = list(range(len(corridors)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    limit = math.radians(straight_deg)
    for ends in at.values():
        cand = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                (ia, ea), (ib, eb) = ends[a], ends[b]
                if ia == ib:
                    continue
                t = _turn(out[ia][ea], out[ib][eb])
                if t < limit:
                    cand.append((t, ia, ib))
        used = set()
        for _, ia, ib in sorted(cand):
            # one continuation per corridor end: a crossroads pairs into
            # two roads passing through, not four
            if ia in used or ib in used:
                continue
            used.add(ia)
            used.add(ib)
            ra, rb = find(ia), find(ib)
            if ra != rb:
                parent[ra] = rb

    groups = {}
    for i in range(len(corridors)):
        groups.setdefault(find(i), []).append(i)
    return [_chain([corridors[i] for i in g]) for g in groups.values()]


def _chain(parts):
    """Stitch corridors sharing end dots into one ordered walk."""
    parts = [list(p) for p in parts]
    walk = parts.pop()
    while parts:
        for k, p in enumerate(parts):
            if p[0] == walk[-1]:
                walk += p[1:]
            elif p[-1] == walk[-1]:
                walk += p[::-1][1:]
            elif p[-1] == walk[0]:
                walk = p[:-1] + walk
            elif p[0] == walk[0]:
                walk = p[::-1][:-1] + walk
            else:
                continue
            parts.pop(k)
            break
        else:
            break        # a branching or looping group; keep what is ordered
    return walk


def summarise(graph):
    """(corridors, [{nodes, length_m}, ...]) for a fetched-graph dict."""
    corridors = decompose(graph["adjacency"])
    xy = _metres(graph["points"])
    stats = [{"nodes": len(c), "length_m": _length(xy, c)} for c in corridors]
    return corridors, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default=FETCHED_GRAPH)
    ap.add_argument("--out", help="write corridors as JSON")
    args = ap.parse_args()

    graph = json.load(open(args.graph))
    corridors, stats = summarise(graph)
    lengths = sorted(s["length_m"] for s in stats)
    covered = len({n for c in corridors for n in c})

    print(f"{len(graph['points'])} dots -> {len(corridors)} corridors "
          f"({covered} dots covered)")
    print(f"length: min {lengths[0]:.0f} m  median {lengths[len(lengths)//2]:.0f} m  "
          f"max {lengths[-1]:.0f} m  total {sum(lengths):.0f} m")
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 200), (200, 1e9)):
        n = sum(1 for l in lengths if lo <= l < hi)
        print(f"  {lo:>3.0f}-{hi if hi < 1e9 else '+':>4} m: {n:>4}")

    if args.out:
        json.dump([{"nodes": c, **s} for c, s in zip(corridors, stats)],
                  open(args.out, "w"))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
