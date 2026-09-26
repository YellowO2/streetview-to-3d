"""The one way anything here gets a GPU, and the one DA3 model.

ZeroGPU attaches a GPU only inside a function decorated with @spaces.GPU.
There is exactly one, `run`, like DA3's own official Space: a task is a
plain function handed to it, and the window it asks for is the task's own
estimate. The street reconstruction and the splat both go through it.

On a Space, DA3 is built at startup and placed on cuda, as HF's ZeroGPU docs
ask: ZeroGPU then moves it onto the GPU for each call, quickly. Built inside
a call instead, it loaded from disk every time (17-28 s of the window),
because each call runs in a fresh worker and nothing it loads survives.

`spaces` must be imported before anything initialises CUDA -- streetlevel's
Look Around reprojection does on import -- or it refuses to load. The
package's __init__ imports this module first for that reason.
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
    # depth_anything_3 imports pycolmap for an export format we never use,
    # and pycolmap's native init calls CUDA directly -- outside a GPU call,
    # past ZeroGPU's emulation, it segfaults. A placeholder satisfies the
    # import without ever loading it. (Same fix the old 3DGS app used.)
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    get_da3()
    # solo mode's model: downloaded now, built inside the call that uses it
    from streetview_to_3d.config import DA3_SOLO_MODEL_REPO
    get_da3_config(DA3_SOLO_MODEL_REPO)
    from streetview_to_3d.services.segment import get_segmenter
    get_segmenter()  # the car/person masker, small, same treatment as DA3
