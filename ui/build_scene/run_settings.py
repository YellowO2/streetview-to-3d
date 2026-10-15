"""Turn the choices for a build into the engine's Settings."""
from streetview_to_3d.build.Settings import Settings
from streetview_to_3d.build.postprocess.Settings import Settings as PostprocessSettings
from streetview_to_3d.build.reconstruct.WalkSettings import WalkSettings
from streetview_to_3d.build.reconstruct.tools.gpu_seconds import DEFAULT_EFFORT, estimate_gpu_seconds


def run_settings(corridor, choices):
    """The Settings for building corridor with choices (RunChoices): 0 or blank overrides
    leave the engine's defaults; the GPU window is sized from the effort unless set."""
    walk = WalkSettings(
        conf_lower_percentile=100 - choices.keep_pct,
        gpu_seconds=choices.gpu_seconds or estimate_gpu_seconds(len(corridor.points), choices.effort or DEFAULT_EFFORT),
        hfov=choices.view_hfov or None,
        model=(choices.da3_model or "").strip() or None, masker=(choices.masker or "").strip() or None,
        mask_classes=[c.strip() for c in (choices.mask_classes or "").split(",") if c.strip()] or None,
        conf_floor=choices.conf_floor or None)
    return Settings(walk=walk, post=PostprocessSettings(fill=bool(choices.fill)))
