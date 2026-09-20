"""Combine separate runs over the same area into one scene.

A GPU session only covers so much ground, so a large place is built a
stretch at a time. Nothing about that needs reconciling: a node stores its
pano's real lat/lon and DA3's own camera pose, and the scene's centre only
ever sets where metres are measured from. So merging is concatenation --
the runs keep their own DA3 frames, and placement fits each piece to GPS
independently, which is what puts them in the same world.

Runs that overlap share panoramas. The first run to claim one keeps it, so
its points are never counted twice.

    python -m postprocess.merge_scenes --dir run_a run_b --out joined
"""
import argparse
import os
import shutil

import scene as scene_mod


def merge(directories, out_dir, log=print):
    """Write one scene combining several, with every node's ply beside it."""
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    scenes = [(d, scene_mod.Scene.load(d)) for d in directories]
    out = scene_mod.Scene(center=scenes[0][1].center)

    seen = {}                       # pano key -> index in the merged scene
    for d, sc in scenes:
        keep = {}                   # its index -> the merged one
        for i, node in enumerate(sc.nodes):
            if node.pano.key in seen:
                keep[i] = seen[node.pano.key]
                continue
            j = len(out.nodes)
            if node.ply:
                name = f"node_{j}.ply"
                shutil.copy(os.path.join(d, node.ply), os.path.join(out_dir, name))
                node = scene_mod.Node(pano=node.pano, ply=name,
                                      position=node.position, rotation=node.rotation)
            out.nodes.append(node)
            seen[node.pano.key] = keep[i] = j

        for e in sc.edges:
            if keep[e.a] != keep[e.b]:
                out.edges.append(scene_mod.Edge(a=keep[e.a], b=keep[e.b],
                                                keep_a=e.keep_a, keep_b=e.keep_b))
        for k, vs in sc.adjacency.items():
            out.adjacency.setdefault(str(keep[int(k)]), [])
            out.adjacency[str(keep[int(k)])] += [keep[v] for v in vs if v in keep]

        log(f"{os.path.basename(d)}: {len(sc.nodes)} node(s), {len(sc.edges)} edge(s)")

    out.adjacency = {k: sorted(set(v)) for k, v in out.adjacency.items()}
    out.save(out_dir)
    shared = sum(len(sc.nodes) for _, sc in scenes) - len(out.nodes)
    log(f"-> {len(out.nodes)} node(s), {len(out.edges)} edge(s), "
        f"{len(out.pieces())} piece(s)"
        + (f", {shared} shared panorama(s) kept once" if shared else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", nargs="+", required=True, help="scene directories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge([os.path.expanduser(d) for d in args.dir], args.out)


if __name__ == "__main__":
    main()
