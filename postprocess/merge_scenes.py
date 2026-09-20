"""Combine separate runs over the same area into one scene.

A GPU session only covers so much ground, so a large place is built a
stretch at a time. Nothing about that needs reconciling: a node stores its
pano's real lat/lon and DA3's own camera pose, and the scene's centre only
ever sets where metres are measured from. So merging is concatenation --
the runs keep their own DA3 frames, and placement fits each piece to GPS
independently, which is what puts them in the same world.

Runs that overlap share panoramas, and each run still keeps its own node
for one: its camera position is in ITS OWN DA3 frame, so letting a shared
pano join the two runs' edges would weld two unrelated frames into one
rigid body -- measured on two real runs, that took the GPS residual from
5.15 m and 1.83 m to 18.12 m. Only the points are shared: the later run's
node carries no ply, so the geometry is not laid down twice.

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

    seen, shared = set(), 0
    for d, sc in scenes:
        keep = {}                   # its index -> the merged one
        for i, node in enumerate(sc.nodes):
            j = len(out.nodes)
            ply = None
            if node.ply and node.pano.key not in seen:
                ply = f"node_{j}.ply"
                shutil.copy(os.path.join(d, node.ply), os.path.join(out_dir, ply))
            elif node.ply:
                shared += 1
            out.nodes.append(scene_mod.Node(pano=node.pano, ply=ply,
                                            position=node.position,
                                            rotation=node.rotation))
            seen.add(node.pano.key)
            keep[i] = j

        for e in sc.edges:
            out.edges.append(scene_mod.Edge(a=keep[e.a], b=keep[e.b],
                                            keep_a=e.keep_a, keep_b=e.keep_b))
        for k, vs in sc.adjacency.items():
            out.adjacency.setdefault(str(keep[int(k)]), [])
            out.adjacency[str(keep[int(k)])] += [keep[v] for v in vs if v in keep]

        log(f"{os.path.basename(d)}: {len(sc.nodes)} node(s), {len(sc.edges)} edge(s)")

    out.adjacency = {k: sorted(set(v)) for k, v in out.adjacency.items()}
    out.save(out_dir)
    log(f"-> {len(out.nodes)} node(s), {len(out.edges)} edge(s), "
        f"{len(out.pieces())} piece(s)"
        + (f"; {shared} pano(s) seen by more than one run, points kept once"
           if shared else ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", nargs="+", required=True, help="scene directories")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge([os.path.expanduser(d) for d in args.dir], args.out)


if __name__ == "__main__":
    main()
