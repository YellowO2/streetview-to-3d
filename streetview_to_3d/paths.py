"""Where data lives: one root, so everything the app writes is in one place.

    <DATA_DIR>/panos/       downloaded panoramas, a cache any run can reuse
    <DATA_DIR>/runs/<id>/   one reconstruction: scene.json and its clouds

The root is STREETVIEW_TO_3D_DATA if set, else ./data under the working
directory -- not beside this file, which is inside site-packages once the
package is installed.
"""
import os

DATA_DIR = os.path.abspath(os.environ.get("STREETVIEW_TO_3D_DATA", "data"))
PANOS_DIR = os.path.join(DATA_DIR, "panos")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

os.makedirs(PANOS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
