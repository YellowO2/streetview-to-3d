"""Build a scene's Google base and write it as a scene folder: one piece per
Google pano (its walls and other surfaces) and one piece for the shared
ground. Plain grey -- the base is shape only.

usage: python -m streetview_to_3d.google_base SCENE_DIR OUT_DIR
"""
import copy
import json
import os
import sys
import time

import numpy as np

from streetview_to_3d.google_base import build, gather
from streetview_to_3d.postprocess.ply_io import write_ply

GREY = 0.5


def write(base, scene, out):
    os.makedirs(out, exist_ok=True)
    tmpl = next(n for n in scene["nodes"] if n["pano"]["source"] == "google")
    nodes = []
    for k, (p, x) in enumerate(zip(base.panos, base.points)):
        name = f"google_{k}.ply"
        write_ply(os.path.join(out, name), x, np.full((len(x), 3), GREY))
        n = copy.deepcopy(tmpl)
        n["pano"].update(id=p.id, lat=p.lat, lon=p.lon, heading=p.heading, elevation=p.elevation, date=p.date)
        n.update(ply=name, position=p.pos.tolist(), rotation=p.R.tolist(), transform=np.eye(4).tolist())
        nodes.append(n)
    write_ply(os.path.join(out, "ground.ply"), base.ground, np.full((len(base.ground), 3), GREY))
    g = copy.deepcopy(nodes[0])
    g.update(ply="ground.ply")
    nodes.append(g)
    json.dump({"center": scene["center"], "nodes": nodes, "adjacency": {}, "edges": []},
              open(os.path.join(out, "scene.json"), "w"))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1:3]
    scene = json.load(open(os.path.join(src, "scene.json")))
    t0 = time.time()
    panos = gather(scene)
    print(f"fetched {len(panos)} Google panos ({sum(p.in_scene for p in panos)} in the scene)  "
          f"[{time.time() - t0:.1f}s]")
    write(build(panos), scene, out)
    print(f"wrote {out}  [total {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
