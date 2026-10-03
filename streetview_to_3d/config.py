from dataclasses import dataclass

from huggingface_hub import snapshot_download

# Metres per DA3 unit when a scene can't fit its own (postprocess/place.piece_scale);
# per-scene fits range about 1.17-1.62.
DA3_UNITS_TO_METRES = 1.3

# Nested 1.1: better per-pano clouds than the original Nested; the non-nested model's
# pieces each came out a different scale.
DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


@dataclass
class DA3Config:
    """All panoramic_da3.run_da3 needs from a config: the model's local path."""
    da3_model: str = ""


def load_da3_config(repo: str = DA3_MODEL_REPO) -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id=repo))
