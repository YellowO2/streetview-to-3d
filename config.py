from huggingface_hub import hf_hub_download, snapshot_download
from panoramic_to_3dgs import PipelineConfig

PIPELINE_CONFIG_PATH = "config.yaml"


def load_pipeline_config() -> PipelineConfig:
    config = PipelineConfig.from_yaml(PIPELINE_CONFIG_PATH)
    config.sharp_model = hf_hub_download(
        repo_id="apple/Sharp",
        filename="sharp_2572gikvuh.pt",
    )
    config.da3_model = snapshot_download(
        repo_id="depth-anything/da3nested-giant-large",
    )
    return config
