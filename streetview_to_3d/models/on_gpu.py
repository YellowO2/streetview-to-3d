"""Run a task with a GPU attached, on a Space through ZeroGPU.

The one way anything gets a GPU (`on_gpu`, through the only @spaces.GPU function).

On a Space, DA3 and the maskers are built at startup so ZeroGPU only moves them per call
(built inside a call they reloaded every time). `spaces` must be imported before CUDA
initialises: the package imports this first.
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
    return task(*args, **kwargs)


def _duration(task, *args, seconds, **kwargs):
    """spaces.GPU calls this with _run's own arguments to size the window."""
    return seconds


_on_gpu = spaces.GPU(duration=_duration)(_run) if ON_SPACES else _run


def on_gpu(task, *args, seconds, **kwargs):
    """task(*args, **kwargs), with a GPU attached for at most `seconds`.
    task must be a module-level function: ZeroGPU pickles it."""
    return _on_gpu(task, *args, seconds=seconds, **kwargs)


if ON_SPACES:
    # depth_anything_3 imports pycolmap (unused), whose CUDA init segfaults outside a GPU call
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))
    from streetview_to_3d.models.da3 import da3
    from streetview_to_3d.models.segment import segment
    da3()
    segment([])  # the maskers, built at startup like DA3
