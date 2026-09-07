"""The one artefact the whole pipeline exists to produce.

Solving is expensive and the answer is small: one 4x4 per piece taking
its stored .ply coordinates straight to final world metres. Save that and
everything downstream -- render this chunk, render a different chunk,
render the lot -- is a matrix multiply against clouds already on disk,
with nothing re-solved.

The matrix composes all three stages:

    GPS FIT     DA3 units to metres, and into the shared world frame
    ROAD        the turn and slide that put the piece on the road
    SEATING     the height and tilt that put it on the road surface

Each stage is also written out separately. The composed matrix is what
you apply; the parts are how you tell which stage put a piece somewhere
odd, and they are why the file records drift and residuals rather than
just the answer.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np

VERSION = 1
FILENAME = "piece_transforms.json"


def gps_transform(fit):
    """The GPS fit as a 4x4: raw piece coordinates -> world metres.

    load_pieces applies this while loading, so a transform solved on top
    of it only maps already-fitted coordinates. Composing the two is what
    makes the saved matrix stand alone -- otherwise reading it back means
    knowing to redo the GPS fit first, in exactly the same way.
    """
    R, s, t = fit["R"], float(fit["scale"]), fit["t"]
    T = np.eye(4)
    T[0, 0], T[0, 2] = s * R[0, 0], s * R[0, 1]
    T[2, 0], T[2, 2] = s * R[1, 0], s * R[1, 1]
    T[1, 1] = s
    T[0, 3], T[2, 3] = t[0], t[1]
    return T


def save(directory, fits, transforms, meta=None, filename=FILENAME):
    """Write one composed 4x4 per piece, plus how it was arrived at."""
    out = {
        "version": VERSION,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dir": os.path.abspath(directory),
        "pieces": {},
    }
    if meta:
        out.update(meta)
    for i, extra in transforms.items():
        T = np.asarray(extra["matrix"], float)
        gps = gps_transform(fits[i])
        out["pieces"][str(i)] = {
            "matrix": (T @ gps).tolist(),          # raw ply -> final world
            "gps_matrix": gps.tolist(),
            "post_gps_matrix": T.tolist(),
            "nodes": int(fits[i]["n"]),
            "gps_residual_m": (None if fits[i].get("resid") is None
                               else float(fits[i]["resid"])),
            **{k: v for k, v in extra.items() if k != "matrix"},
        }
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load(directory, filename=FILENAME):
    """{piece_id: 4x4} taking each piece's stored .ply to world metres."""
    with open(os.path.join(directory, filename)) as f:
        doc = json.load(f)
    if doc.get("version") != VERSION:
        raise ValueError(f"{filename} is version {doc.get('version')}, expected {VERSION}")
    return {int(k): np.asarray(v["matrix"], float) for k, v in doc["pieces"].items()}


def report(directory, filename=FILENAME):
    """The full record, for looking at rather than applying."""
    with open(os.path.join(directory, filename)) as f:
        return json.load(f)
