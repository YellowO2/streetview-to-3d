from dataclasses import dataclass

from huggingface_hub import snapshot_download

# How many metres one DA3 unit is, when a scene cannot measure its own.
# Placement fits the scale per piece from its GPS (see
# postprocess/place.piece_scale); this is only the fallback,
# for a scene of lone panoramas or a fit out of range.
#
# It is not one constant: per-scene fits have come out 1.17 (NTU), 1.62
# (another NTU street), 1.22 (Stockholm), 1.25 (Gotland) and 1.31 -- one
# per place, middle 1.25, mean 1.31. The older 1.46 predates the fix for
# two pieces sharing a place, which glued unrelated frames into one piece
# and fitted nonsense scales.
DA3_UNITS_TO_METRES = 1.3

# Nested 1.1: better than the original Nested on a pano on its own. It
# links too where a street is open enough (Singapore: 2 of 3 edges); in
# Stockholm's narrow alleys nothing linked (0 of 34 pair tests), and then
# every place keeps its own best pano, placed by its GPS and heading.
#   DA3NESTED-GIANT-LARGE      the original; linked Stockholm (20 of 89),
#                              its per-pano clouds are worse.
#   DA3-GIANT-1.1              links, but relative only: each piece came
#                              out its own scale, so no one
#                              DA3_UNITS_TO_METRES fits.
DA3_MODEL_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config(repo: str = DA3_MODEL_REPO) -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id=repo))
