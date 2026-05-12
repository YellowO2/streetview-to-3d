import asyncio
import json
import os

from aiohttp import ClientSession
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from config import load_pipeline_config
from services.job_store import Job, create_job, get_job, update_job
from services.street_view import ensure_pano_downloaded

router = APIRouter()


class GenerateRequest(BaseModel):
    pano_ids: List[str]
    metadata: list = []


@router.post("/generate_3dgs")
async def generate_3dgs(req: GenerateRequest):
    job = create_job(req.pano_ids)
    asyncio.create_task(_run_pipeline_task(job.job_id, req.pano_ids, req.metadata))
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
    }


async def _run_pipeline_task(job_id: str, pano_ids: list[str], metadata: list[dict]):
    update_job(job_id, status="running")
    try:
        await asyncio.to_thread(_pipeline_sync, job_id, pano_ids, metadata)
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


def _pipeline_sync(job_id: str, pano_ids: list[str], metadata: list[dict]):
    """Blocking pipeline execution — runs in a thread pool."""
    import torch
    from panoramic_to_3dgs import Pipeline

    job = get_job(job_id)
    output_dir = job.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata alongside the output for traceability
    with open(os.path.join(output_dir, "job_metadata.json"), "w") as f:
        json.dump({"pano_ids": pano_ids, "metadata": metadata}, f, indent=2)

    # Download any panos not already cached
    panorama_paths = []
    for pano_id in pano_ids:
        img_path = f"images/pano_{pano_id}.jpg"
        if not os.path.exists(img_path):
            print(f"Downloading missing panorama {pano_id}...")
            asyncio.run(_download_missing(pano_id, img_path))
        panorama_paths.append(img_path)

    # Serialize GPU jobs and free all memory after each run
    with _get_gpu_lock():
        config = load_pipeline_config()
        pipeline = Pipeline(config)
        try:
            pipeline.run(panorama_paths=panorama_paths, output_dir=output_dir)
        finally:
            del pipeline
            torch.cuda.empty_cache()

    # Collect output PLY URLs (served via static /splats mount)
    ply_files = [
        f"/splats/{job_id}/{f}"
        for f in sorted(os.listdir(output_dir))
        if f.endswith(".ply")
    ]
    update_job(job_id, status="done", ply_files=ply_files)
    print(f"Job {job_id} complete: {ply_files}")


async def _download_missing(pano_id: str, img_path: str):
    async with ClientSession() as session:
        await ensure_pano_downloaded(pano_id, session)
