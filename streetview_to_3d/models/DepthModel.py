"""DA3, the depth model, loaded once per repo, with the constants it runs by.

DA3, the depth model: which one, where it is (DA3Config) and our decisions about its
results: thresholds and view settings. reconstruct/PanoDepth.py is the only caller of
panoramic_da3.panoramic_da3.
"""
from huggingface_hub import snapshot_download

from streetview_to_3d.models.DA3Config import DA3Config

# Metres per DA3 unit when a scene can't fit its own (postprocess/place.piece_scale);
# per-scene fits range about 1.17-1.62.
DA3_UNITS_TO_METRES = 1.3

# Nested 1.1: better per-pano clouds than the original Nested; the non-nested model's
# pieces each came out a different scale.
DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"

KEEP_RATE_THRESHOLD = 0.6

# Yaw step between DA3 views: 30 gives 12 views, so a view count means the same everywhere.
VIEW_STEP_DEGREES = 30

# How much of each view's least confident pixels DA3 drops; the UI's hidden "Keep %" overrides it.
CONF_LOWER_PERCENTILE = 25.0   # keep top 75%

# Width of each view in degrees: wider kept more points but looked worse.
VIEW_HFOV = 90.0

# The lowest DA3 confidence a pixel needs to become a point, on top of CONF_LOWER_PERCENTILE.
CONF_FLOOR = 1.05

# Leave cars, people and poles out of every point cloud (see models.Segmenter).
MASK_MOVERS = True

_loaded = {}


class DepthModel:
    """One DA3 repo's model (default DA3_MODEL_REPO), built once and shared: the default at
    startup on a Space (tools/on_gpu.py), anything else on first use. Only the repo is
    pickled, so it travels into a GPU call for free."""

    def __init__(self, repo=None):
        self.repo = repo or DA3_MODEL_REPO

    def loaded(self):
        """(DA3Config, panoramic_da3's DA3Model)."""
        if self.repo not in _loaded:
            from panoramic_da3 import DA3Model
            config = DA3Config(da3_model=snapshot_download(repo_id=self.repo))
            _loaded[self.repo] = config, DA3Model(config.da3_model)
        return _loaded[self.repo]
