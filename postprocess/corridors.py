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

EARTH_R = 6371000.0


def _metres(points):
    """lat/lon -> local metres, equirectangular about the first point."""
    lat0 = math.radians(points[0][0])
    k = math.cos(lat0)
    return [(math.radians(lon) * k * EARTH_R, math.radians(lat) * EARTH_R)
            for lat, lon in points]


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
