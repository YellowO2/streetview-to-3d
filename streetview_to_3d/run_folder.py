"""Make a fresh folder for one run's scene, and say where it is."""
from streetview_to_3d.common.paths import new_run_dir


def run_folder(log=print):
    """A fresh, empty folder under the data folder's runs/ (old runs are cleared first)."""
    path = new_run_dir()
    log(f"run dir: {path}")
    return path
