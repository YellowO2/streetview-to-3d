"""The one way anything gets a GPU (`run`, the only @spaces.GPU function), and the DA3 model.

On a Space, DA3 is built at startup so ZeroGPU only moves it per call (built inside a call it
reloaded every time). `spaces` must be imported before CUDA initialises: the package imports this first.
"""
import os
import sys
import types

try:
    import spaces
    # installed locally too (requirements.txt), so only a Space counts
    ON_SPACES = bool(os.getenv("SPACE_ID"))
except ImportError:
    ON_SPACES = False


def _run(task, *args, seconds, **kwargs):
    """task(*args, **kwargs), with a GPU attached for at most `seconds`.
    task must be a module-level function: ZeroGPU pickles it."""
    return task(*args, **kwargs)


def _duration(task, *args, seconds, **kwargs):
    """spaces.GPU calls this with run's own arguments to size the window."""
    return seconds


run = spaces.GPU(duration=_duration)(_run) if ON_SPACES else _run

_da3_configs = {}
_da3s = {}


def get_da3_config(repo=None):
    from streetview_to_3d.config import DA3_MODEL_REPO, load_da3_config
    repo = repo or DA3_MODEL_REPO
    if repo not in _da3_configs:
        _da3_configs[repo] = load_da3_config(repo)
    return _da3_configs[repo]


def get_da3(repo=None):
    """A DA3 model, by repo (default config.DA3_MODEL_REPO): the default is
    built at startup on a Space, anything else on first use."""
    from streetview_to_3d.config import DA3_MODEL_REPO
    repo = repo or DA3_MODEL_REPO
    if repo not in _da3s:
        from panoramic_da3 import DA3Model
        _da3s[repo] = DA3Model(get_da3_config(repo).da3_model)
    return _da3s[repo]


if ON_SPACES:
    # depth_anything_3 imports pycolmap (unused), whose CUDA init segfaults outside a GPU call
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    get_da3()
    from streetview_to_3d.services.segment import get_segmenter
    get_segmenter()  # the maskers, built at startup like DA3
