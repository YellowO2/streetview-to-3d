from dataclasses import dataclass

from huggingface_hub import snapshot_download

# How many metres one DA3 unit is, when a scene cannot measure its own.
# Placement fits the scale per scene from its linked pieces' GPS (see
# postprocess/gps_fit/load_pieces.scene_scale); this is only the fallback,
# for a scene of lone panoramas or a fit out of range.
#
# It is not one constant: per-scene fits have come out 1.17 (NTU), 1.62
# (another NTU street), 1.22 (Stockholm), 1.25 (Gotland) and 1.31 -- one
# per place, middle 1.25, mean 1.31. The older 1.46 predates the fix for
# two pieces sharing a place, which glued unrelated frames into one piece
# and fitted nonsense scales.
DA3_UNITS_TO_METRES = 1.3

# The original Nested, deprecated by DA3 but the best of the three tried
# on Stockholm (same pairs, same panos):
#   DA3NESTED-GIANT-LARGE-1.1  more views kept per pano on its own, but no
#                              pair test kept views on both panos (0 of 34,
#                              vs 20 of 89 here), so nothing linked.
#   DA3-GIANT-1.1              links like this one (7 of 30), but relative
#                              only: each piece came out its own scale
#                              (Apple about half of Google), so no one
#                              DA3_UNITS_TO_METRES fits.
DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"
# Solo mode (reconstruct.solo) never links, so it takes the 1.1 that keeps
# more views per pano on its own.
DA3_SOLO_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config(repo: str = DA3_MODEL_REPO) -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id=repo))
