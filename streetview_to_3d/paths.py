"""Where data lives: one root, so everything the app writes is in one place.

    <DATA_DIR>/panos/       downloaded panoramas, a cache any run can reuse
    <DATA_DIR>/runs/<id>/   one run's output

The root is STREETVIEW_TO_3D_DATA if set, else ./data under the working
directory -- not beside this file, which is inside site-packages once the
package is installed.
"""
import os
import shutil
import time
import uuid

DATA_DIR = os.path.abspath(os.environ.get("STREETVIEW_TO_3D_DATA", "data"))
PANOS_DIR = os.path.join(DATA_DIR, "panos")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

# A Space's disk is only emptied when it restarts, so anything under
# DATA_DIR left untouched this long is removed when a new run starts.
KEEP_S = 24 * 3600

os.makedirs(PANOS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def new_run_dir():
    """A fresh, empty folder for one run, after clearing out old data."""
    remove_older_than(time.time() - KEEP_S)
    path = os.path.join(RUNS_DIR, uuid.uuid4().hex)
    os.makedirs(path)
    return path


def remove_older_than(cutoff):
    """Delete every entry one level inside DATA_DIR's folders (a run, a
    cached pano, an upload) last modified before `cutoff`."""
    for folder in os.scandir(DATA_DIR):
        if not folder.is_dir():
            continue
        for entry in os.scandir(folder.path):
            try:
                if entry.stat().st_mtime >= cutoff:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry.path, ignore_errors=True)
                else:
                    os.remove(entry.path)
            except FileNotFoundError:
                pass  # another request removed it first
