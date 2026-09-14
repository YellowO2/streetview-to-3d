"""The area a reconstruction covers, recorded beside its output.

Every position in postprocess is metres east/north of this centre, and
piece_transforms.json is saved relative to it, so it is written once by the
run that produced the pieces and never recomputed afterwards. The centre is
the coordinate the user searched for, which is what defined the area.
"""
import json
import os

FILENAME = "area.json"


def save(directory, lat, lon):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, FILENAME)
    with open(path, "w") as f:
        json.dump({"center": {"lat": float(lat), "lon": float(lon)}}, f, indent=2)
    return path


def load(directory):
    """(lat, lon) of the area's centre."""
    path = os.path.join(directory, FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing -- it records the centre every position here "
            "is measured from, so the pieces cannot be placed without it.")
    with open(path) as f:
        centre = json.load(f)["center"]
    return centre["lat"], centre["lon"]
