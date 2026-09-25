from dataclasses import dataclass

from huggingface_hub import snapshot_download

# How many metres one DA3 unit is, when a scene cannot measure its own.
# Placement fits the scale per scene from its linked pieces' GPS (see
# postprocess/gps_fit/load_pieces.scene_scale); this is only the fallback,
# for a scene of lone panoramas or a fit out of range.
#
# Two measurements disagree, and that is not yet explained. Over NTU's 117
# chunks the answer held at 1.334-1.346 across
# every break threshold from 12 m down to 0.75 m. Three later runs down one
# street each want 1.46-1.54 instead, and they agree with each other far
# better than with 1.335: at 1.46 one of the three fits its own GPS to
# 0.18 m and asks for no further scaling at all.
DA3_UNITS_TO_METRES = 1.46
#
# Every number above was measured on the original DA3NESTED-GIANT-LARGE,
# which DA3 has since deprecated for a training bug. Later runs on it fit
# 1.17-1.26 (Stockholm, and gen_10node at NTU), so it was never one
# constant -- hence the per-scene fit.

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


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config() -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id=DA3_MODEL_REPO))
