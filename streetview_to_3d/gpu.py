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

_da3_config = None
_da3 = None


def get_da3_config():
    global _da3_config
    if _da3_config is None:
        from streetview_to_3d.config import load_da3_config
        _da3_config = load_da3_config()
    return _da3_config


def get_da3():
    """The DA3 model: built at startup on a Space, on first use elsewhere."""
    global _da3
    if _da3 is None:
        from panoramic_da3 import DA3Model
        _da3 = DA3Model(get_da3_config().da3_model)
    return _da3


if ON_SPACES:
    # depth_anything_3 imports pycolmap for an export format we never use,
    # and pycolmap's native init calls CUDA directly -- outside a GPU call,
    # past ZeroGPU's emulation, it segfaults. A placeholder satisfies the
    # import without ever loading it. (Same fix the old 3DGS app used.)
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    get_da3()
    from streetview_to_3d.services.segment import get_segmenter
    get_segmenter()  # the car/person masker, small, same treatment as DA3
