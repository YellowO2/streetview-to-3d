from panoramic_to_3dgs import PipelineConfig

PIPELINE_CONFIG_PATH = "config.yaml"


def load_pipeline_config() -> PipelineConfig:
    return PipelineConfig.from_yaml(PIPELINE_CONFIG_PATH)
