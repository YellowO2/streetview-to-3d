"""Where data lives: STREETVIEW_TO_3D_DATA, else ./data in the working directory.
panos/ caches downloaded panoramas; runs/<id>/ holds one run's output."""
import os
import shutil
import time
import uuid

DATA_DIR = os.path.abspath(os.environ.get("STREETVIEW_TO_3D_DATA", "data"))
PANOS_DIR = os.path.join(DATA_DIR, "panos")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

# A Space's disk only empties on restart: anything untouched this long goes when a run starts.
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
    """Delete every entry one level inside DATA_DIR's folders last modified before `cutoff`."""
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
