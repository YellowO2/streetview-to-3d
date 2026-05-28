import asyncio
import json
import os

from aiohttp import ClientSession
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from streetlevel import streetview

from services.job_store import create_job, get_job, update_job
from services.download_street_panorama import ensure_pano_downloaded

router = APIRouter()

# Depth-only support panoramas added to each job (target's nearest Street View
# neighbors). More supports give DA3 stronger translation baselines for better
# depth/pose at the cost of VRAM (18 DA3 slices per pano).
MAX_SUPPORT_PANOS = 2


class GenerateRequest(BaseModel):
    pano_id: str
    metadata: dict | None = None
    scale_mode: str = "da3_2dgrid_global"


@router.post("/generate_3dgs")
async def generate_3dgs(req: GenerateRequest, request: Request):
    job = create_job(req.pano_id)
    session = request.app.state.session
    asyncio.create_task(
        _run_pipeline_task(job.job_id, req.pano_id, req.metadata, req.scale_mode, session)
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


async def _select_support_panos(target_pano_id: str, lat: float, lon: float, session: ClientSession):
    """Fetch the target's Street View neighbors by lat/lon, return pano objects (cap at MAX_SUPPORT_PANOS)."""
    pano = await streetview.find_panorama_async(lat, lon, session=session)
    if not pano:
        return []
    candidates = pano.links or pano.neighbors
    support_panos = []
    seen_ids: set[str] = set()
    for item in candidates:
        neighbor = item.pano if hasattr(item, "pano") else item
        if neighbor and neighbor.id != target_pano_id and neighbor.id not in seen_ids:
            seen_ids.add(neighbor.id)
            support_panos.append(neighbor)
        if len(support_panos) >= MAX_SUPPORT_PANOS:
            break
    return support_panos


async def _download_pano_object(pano, session: ClientSession) -> str:
    """Download a support pano. Neighbor stubs lack image_sizes, so re-fetch by lat/lon first."""
    from services.download_street_panorama import download_panorama_image
    img_path = f"images/pano_{pano.id}.jpg"
    if not os.path.exists(img_path):
        os.makedirs("images", exist_ok=True)
        full_pano = await streetview.find_panorama_async(pano.lat, pano.lon, session=session)
        if not full_pano:
            raise ValueError(f"Could not fetch support pano {pano.id} at ({pano.lat}, {pano.lon})")
        await download_panorama_image(full_pano, img_path)
    return img_path


async def _run_pipeline_task(
    job_id: str,
    target_pano_id: str,
    metadata: dict | None,
    scale_mode: str,
    session: ClientSession,
):
    update_job(job_id, status="running")
    try:
        lat = metadata.get("lat") if metadata else None
        lon = metadata.get("lon") if metadata else None
        support_panos = await _select_support_panos(target_pano_id, lat, lon, session) if lat and lon else []
        support_ids = [p.id for p in support_panos]
        print(f"Job {job_id}: target={target_pano_id} supports={support_ids}")

        target_path = await ensure_pano_downloaded(target_pano_id, session)
        support_paths: list[str] = []
        for pano in support_panos:
            try:
                support_paths.append(await _download_pano_object(pano, session))
            except Exception as e:
                print(f"Support pano {pano.id} failed to download, skipping: {e}")

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
            scale_mode,
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
    scale_mode: str = "da3_2dgrid_global",
):
    """Runs the pipeline in an isolated subprocess — GPU memory is freed on process exit."""
    import subprocess
    import sys

    job = get_job(job_id)
    panorama_paths = [target_path, *support_paths]

    args_json = json.dumps({
        "output_dir": job.output_dir,
        "panorama_paths": panorama_paths,
        "target_pano_id": target_pano_id,
        "support_pano_ids": job.support_pano_ids,
        "metadata": metadata,
        "scale_mode": scale_mode,
    })

    with _get_gpu_lock():
        result = subprocess.run(
            [sys.executable, "-c",
             "import json, sys; from services.pipeline_runner import run_pipeline; "
             "run_pipeline(**json.loads(sys.argv[1]))",
             args_json],
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:
        error_msg = result.stderr.strip().splitlines()
        raise RuntimeError(error_msg[-1] if error_msg else f"Pipeline failed (exit {result.returncode})")

    ply_files = [f"/splats/{job_id}/final_output.ply"]
    update_job(job_id, status="done", ply_files=ply_files)
    print(f"Job {job_id} complete: {ply_files}")
