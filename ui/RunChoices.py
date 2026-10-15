"""What was chosen for one build, hidden overrides included."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunChoices:
    """keep_pct of each view's pixels; gpu_seconds (0: sized from effort); view_hfov,
    da3_model, masker, mask_classes (comma-separated), conf_floor (0 or blank: defaults);
    fill the ground; effort: how hard to try; share: publish to the gallery."""
    keep_pct: float
    gpu_seconds: float
    view_hfov: float
    da3_model: str
    masker: str
    mask_classes: str
    fill: bool
    conf_floor: float
    effort: str
    share: bool
