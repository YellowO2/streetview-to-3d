from dataclasses import dataclass

from huggingface_hub import snapshot_download

# How many metres one DA3 unit is. DA3 is internally consistent -- the same
# unit means the same length in every reconstruction it produces -- so this
# is one property of the model, not something to re-derive per run.
#
# Measured over NTU's 117 chunks: cut each chunk wherever its own nodes stop
# matching their GPS, fit every resulting piece of 4+ nodes on its own, and
# take the middle. The answer holds at 1.334-1.346 across break thresholds
# from 12 m down to 0.75 m. At 0.75 m the mean and the median agree to four
# decimals, meaning nothing skews what is left. See the README.
DA3_UNITS_TO_METRES = 1.335


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config() -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id="depth-anything/da3nested-giant-large"))
