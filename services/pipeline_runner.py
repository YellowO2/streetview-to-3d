"""
Runs the 3DGS pipeline in an isolated subprocess so all GPU memory is
released unconditionally on process exit — no dependence on Python GC or
PyTorch's caching allocator.
"""
import json
import os


def run_pipeline(
    output_dir: str,
    panorama_paths: list,
    target_pano_id: str,
    support_pano_ids: list,
    metadata: dict | None,
    scale_mode: str = "da3_2dgrid_global",
):
    from panoramic_to_3dgs import Pipeline
    from config import load_pipeline_config

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "job_metadata.json"), "w") as f:
        json.dump(
            {
                "target_pano_id": target_pano_id,
                "support_pano_ids": support_pano_ids,
                "metadata": metadata,
            },
            f,
            indent=2,
        )

    config = load_pipeline_config()
    config.scale_mode = scale_mode
    pipeline = Pipeline(config)
    pipeline.run(
        panorama_paths=panorama_paths,
        output_dir=output_dir,
        target_pano_id=0,
    )
