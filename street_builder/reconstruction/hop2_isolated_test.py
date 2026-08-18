"""One-off diagnostic: feed exactly the two panos from a specific hop
(WbqrDtjmbGNqHBZC3U2ZxA -> tBKrnz8UAtdMmhkmgePciw, the "hop 2" edge from a
real pathfind run) into DA3 completely on its own -- no chaining, no other
panos in the batch -- so the resulting point cloud can be inspected
directly to see whether DA3's relative placement of just these two panos
is visibly wrong on its own. Delete this file + its button once the
question's answered.
"""
import os
import uuid

from paths import SPLATS_DIR
from services.pipeline_runner import run_pointcloud_gpu
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES

TARGET_ID = "WbqrDtjmbGNqHBZC3U2ZxA"
SUPPORT_ID = "tBKrnz8UAtdMmhkmgePciw"


def run_hop2_isolated_test() -> str:
    """Downloads the two hardcoded panos, runs one DA3 pointcloud call
    (target + one support, nothing else), returns the .ply path."""
    target_path = run_async(download_pano_by_id(TARGET_ID, zoom=DA3_ONLY_ZOOM))
    support_path = run_async(download_pano_by_id(SUPPORT_ID, zoom=DA3_ONLY_ZOOM))
    if not target_path or not support_path:
        raise RuntimeError(f"Download failed: target={target_path}, support={support_path}")

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    ply = run_pointcloud_gpu(
        target_depth_path=target_path,
        output_dir=output_dir,
        support_paths=[support_path],
        step_degrees=BEST4_STEP_DEGREES,
    )
    if not ply:
        raise RuntimeError("No views survived the DA3 filter for this pair.")
    return ply
