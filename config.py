from dataclasses import dataclass

from huggingface_hub import snapshot_download

# How many metres one DA3 unit is. DA3 is internally consistent -- the same
# unit means the same length in every reconstruction it produces -- so this
# is one property of the model, not something to re-derive per run.
#
# Two measurements disagree, and that is not yet explained. Over NTU's 117
# chunks (tools/measure_da3_scale.py) the answer held at 1.334-1.346 across
# every break threshold from 12 m down to 0.75 m. Three later runs down one
# street each want 1.46-1.54 instead, and they agree with each other far
# better than with 1.335: at 1.46 one of the three fits its own GPS to
# 0.18 m and asks for no further scaling at all.
DA3_UNITS_TO_METRES = 1.46


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config() -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id="depth-anything/da3nested-giant-large"))
