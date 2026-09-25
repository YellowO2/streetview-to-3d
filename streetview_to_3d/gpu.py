"""The one way anything here gets a GPU, and the one DA3 model.

ZeroGPU attaches a GPU only inside a function decorated with @spaces.GPU.
There is exactly one, `run`, like DA3's own official Space: a task is a
plain function handed to it, and the window it asks for is the task's own
estimate. The street reconstruction and the splat both go through it, so
they share one loaded DA3 rather than each keeping their own.

`spaces` must be imported before anything initialises CUDA -- streetlevel's
Look Around reprojection does on import -- or it refuses to load. The
package's __init__ imports this module first for that reason.
"""
import os

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
    """The DA3 model, built on first use INSIDE a GPU call and reused after.

    Not at import: building it before a GPU is attached segfaults on
    pycolmap's raw CUDA calls, which bypass spaces' PyTorch-only .cuda()
    emulation. Re-attached to CUDA on every call, not just the first, in
    case it drifted back to CPU in between -- DA3's own Space
    (model_inference.py's _MODEL_CACHE) does the same check every time.
    """
    global _da3
    if _da3 is None:
        from panoramic_da3 import DA3Model
        _da3 = DA3Model(get_da3_config().da3_model)
    elif next(_da3.model.parameters()).device.type != "cuda":
        _da3.model = _da3.model.to(device="cuda")
    return _da3
