import asyncio
import json
import os

from aiohttp import ClientSession
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from streetlevel import streetview

from config import load_pipeline_config
from services.job_store import create_job, get_job, update_job
from services.download_street_panorama import ensure_pano_downloaded

router = APIRouter()

# Depth-only support panoramas added to each job (target's nearest Street View
# neighbors). More supports give DA3 stronger translation baselines for better
# depth/pose at the cost of VRAM (18 DA3 slices per pano).
MAX_SUPPORT_PANOS = 3


class GenerateRequest(BaseModel):
    pano_id: str
    metadata: dict | None = None


@router.post("/generate_3dgs")
async def generate_3dgs(req: GenerateRequest, request: Request):
    job = create_job(req.pano_id)
    session = request.app.state.session
    asyncio.create_task(
        _run_pipeline_task(job.job_id, req.pano_id, req.metadata, session)
    )
    return {"job_id": job.job_id}


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job.status,
        "ply_files": job.ply_files,
        "error": job.error,
        "target_pano_id": job.target_pano_id,
        "support_pano_ids": job.support_pano_ids,
    }


async def _select_support_panos(target_pano_id: str, lat: float, lon: float, session: ClientSession) -> list[str]:
    """Fetch the target's Street View neighbors by lat/lon, dedupe, cap at MAX_SUPPORT_PANOS."""
    pano = await streetview.find_panorama_async(lat, lon, session=session)
    if not pano:
        return []
    candidates = pano.links or pano.neighbors
    support_ids: list[str] = []
    for item in candidates:
        neighbor = item.pano if hasattr(item, "pano") else item
        if neighbor and neighbor.id != target_pano_id and neighbor.id not in support_ids:
            support_ids.append(neighbor.id)
        if len(support_ids) >= MAX_SUPPORT_PANOS:
            break
    return support_ids


async def _run_pipeline_task(
    job_id: str,
    target_pano_id: str,
    metadata: dict | None,
    session: ClientSession,
):
    update_job(job_id, status="running")
    try:
        lat = metadata.get("lat") if metadata else None
        lon = metadata.get("lon") if metadata else None
        support_ids = await _select_support_panos(target_pano_id, lat, lon, session) if lat and lon else []
        print(f"Job {job_id}: target={target_pano_id} supports={support_ids}")

        target_path = await ensure_pano_downloaded(target_pano_id, session)
        support_paths: list[str] = []
        for sid in support_ids:
            try:
                support_paths.append(await ensure_pano_downloaded(sid, session))
            except Exception as e:
                print(f"Support pano {sid} failed to download, skipping: {e}")

        update_job(job_id, support_pano_ids=[
            sid for sid, p in zip(support_ids, support_paths) if p
        ])

        await asyncio.to_thread(
            _pipeline_sync,
            job_id,
            target_pano_id,
            target_path,
            support_paths,
            metadata,
        )
    except Exception as e:
        update_job(job_id, status="error", error=str(e))
        print(f"Pipeline job {job_id} failed: {e}")


_gpu_lock = None


def _get_gpu_lock():
    global _gpu_lock
    import threading
    if _gpu_lock is None:
        _gpu_lock = threading.Lock()
    return _gpu_lock


def _pipeline_sync(
    job_id: str,
    target_pano_id: str,
    target_path: str,
    support_paths: list[str],
    metadata: dict | None,
):
    """Blocking pipeline execution — runs in a thread pool."""
    import torch
    from panoramic_to_3dgs import Pipeline

    job = get_job(job_id)
    output_dir = job.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "job_metadata.json"), "w") as f:
        json.dump(
            {
                "target_pano_id": target_pano_id,
                "support_pano_ids": job.support_pano_ids,
                "metadata": metadata,
            },
            f,
            indent=2,
        )

    # Target is index 0; supports follow. Only the target gets SHARP'd.
    panorama_paths = [target_path, *support_paths]

    with _get_gpu_lock():
        config = load_pipeline_config()
        pipeline = Pipeline(config)
        try:
            pipeline.run(
                panorama_paths=panorama_paths,
                output_dir=output_dir,
                generate_pano_ids=[0],
            )
        finally:
            del pipeline
            torch.cuda.empty_cache()

    ply_files = [f"/splats/{job_id}/final_output.ply"]
    update_job(job_id, status="done", ply_files=ply_files)
    print(f"Job {job_id} complete: {ply_files}")
