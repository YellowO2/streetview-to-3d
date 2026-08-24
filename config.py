from dataclasses import dataclass

from huggingface_hub import snapshot_download


@dataclass
class DA3Config:
    """The only thing panoramic_da3.run_da3 needs from a config: a
    `.da3_model` attribute (model path/repo id). This app has no SHARP/GS
    pipeline, so there's nothing else to configure here."""
    da3_model: str = ""


def load_da3_config() -> DA3Config:
    return DA3Config(da3_model=snapshot_download(repo_id="depth-anything/da3nested-giant-large"))
