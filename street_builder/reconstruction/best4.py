"""Solo-scores a candidate pool (chain nodes + nearby Apple panos) through
DA3 and reconstructs using only the top-scoring candidates.

Only measures each candidate's own internal coherence, not whether it'll
correlate with the others once combined -- see greedy.py for the
pairwise-tested alternative this session found more reliable.
"""
import tempfile

from services.pipeline_runner import run_pointcloud_gpu, score_candidates_gpu
from street_builder.reconstruction.common import Candidate, gather_candidate_pool, labeled_alias

BEST4_FINAL_COUNT = 4

# Yaw step for DA3's view slicing. 30 (12 slices) is the tested middle
# ground between DA3's default 20 (18 slices) and the too-coarse 45 (8
# slices, caused 2/4 winners to go from partial acceptance to fully
# rejected). Used for BOTH scoring and final reconstruction so the two
# stay consistent (a candidate's keep-rate depends on slice count).
BEST4_STEP_DEGREES = 30


def score_and_rank(pool: list[Candidate], step_degrees: int = 20) -> list[Candidate]:
    """Solo-score every candidate and return them sorted best-first."""
    scores = score_candidates_gpu([c.path for c in pool], step_degrees=step_degrees)
    ranked = [c for c, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
    print(f"Candidate scores (label, keep-count/{360 // step_degrees + (360 % step_degrees > 0)}): {list(zip((c.label for c in pool), scores))}")
    return ranked


def reconstruct_chain_best4(nodes: list[dict], output_dir: str, step_degrees: int = BEST4_STEP_DEGREES) -> str:
    pool = gather_candidate_pool(nodes)
    if len(pool) < 2:
        raise ValueError("Need at least 2 candidate panos (chain nodes + Apple support) to score.")

    ranked = score_and_rank(pool, step_degrees=step_degrees)
    winners = ranked[:BEST4_FINAL_COUNT]
    if len(winners) < 2:
        raise ValueError("Not enough candidates survived scoring for multi-view reconstruction.")

    print(f"Reconstructing with top {len(winners)} (step={step_degrees}): {[c.label for c in winners]}")
    with tempfile.TemporaryDirectory() as alias_dir:
        winner_paths = [labeled_alias(c, alias_dir) for c in winners]
        ply_path = run_pointcloud_gpu(
            target_depth_path=winner_paths[0],
            output_dir=output_dir,
            support_paths=winner_paths[1:],
            step_degrees=step_degrees,
        )
    if not ply_path:
        raise RuntimeError("Pipeline finished but no point cloud was produced.")
    return ply_path
