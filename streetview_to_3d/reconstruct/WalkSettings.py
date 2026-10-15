"""Per-run settings for the walk on the GPU."""
from dataclasses import dataclass


@dataclass(frozen=True)
class WalkSettings:
    """Per-run settings for the walk; None keeps each default. Picklable for ZeroGPU.

    gpu_seconds: the window (None: reconstruct/tools/gpu_seconds.estimate_gpu_seconds). model: a DA3 repo (models.DepthModel.DA3_MODEL_REPO).
    masker: a segmenter repo (models.Segmenter.MODEL_ID); hfov, mask_classes, conf_floor: see
    reconstruct.PanoDepth.
    """
    step_degrees: int | None = None
    conf_lower_percentile: float | None = None
    gpu_seconds: float | None = None
    model: str | None = None
    hfov: float | None = None
    masker: str | None = None
    mask_classes: list[str] | None = None
    conf_floor: float | None = None
