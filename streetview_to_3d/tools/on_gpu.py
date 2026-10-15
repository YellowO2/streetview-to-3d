"""Run a task with a GPU attached, on a Space through ZeroGPU.

The one way anything gets a GPU (`on_gpu`, through the only @spaces.GPU function).

`spaces` must be imported before CUDA initialises: the package imports this first.
"""
import os

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
