import json
import os
import shutil
import uuid
import zipfile

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from streetlevel import streetview
from typing import List

from services.download_street_panorama import download_panorama_image

router = APIRouter()


@router.get("/debug/tile")
async def debug_tile(panoid: str = "LoA_9MjJTb7FXLEUlzyrpw", zoom: int = 3):
    url = (
        f"https://streetviewpixels-pa.googleapis.com/v1/tile"
        f"?cb_client=maps_sv.tactile&panoid={panoid}&x=0&y=0&zoom={zoom}"
    )
    async with ClientSession() as s:
        async with s.get(url) as resp:
            data = await resp.read()
    return {
        "url": url,
        "status": resp.status,
        "content_type": resp.headers.get("Content-Type"),
        "bytes_len": len(data),
        "first_bytes_hex": data[:32].hex(),
        "first_bytes_text": data[:200].decode("utf-8", errors="replace"),
    }


@router.get("/metadata")
async def get_metadata(
    request: Request,
    lat: float = None,
    lon: float = None,
    pano_id: str = None,
):
    print(f"Received metadata request - lat: {lat}, lon: {lon}, pano_id: {pano_id}")
    if not pano_id and (lat is None or lon is None):
        return {"error": "Must provide either pano_id OR lat and lon"}

    session = request.app.state.session
    try:
        if pano_id:
            pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        else:
            pano = await streetview.find_panorama_async(lat, lon, session=session)

        if not pano:
            return {"error": "Panorama not found"}

        links = []
        if pano.links:
            for link in pano.links:
                if link.pano:
                    links.append({
                        "id": link.pano.id,
                        "lat": link.pano.lat,
                        "lon": link.pano.lon,
                        "direction": link.direction,
                    })

        return {
            "id": pano.id,
            "lat": pano.lat,
            "lon": pano.lon,
            "date": str(pano.date) if pano.date else None,
            "links": links,
        }
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return {"error": str(e)}


@router.get("/panorama/{lat},{lon}")
async def panorama(lat: float, lon: float, request: Request):
    print(f"Received request for panorama at lat: {lat}, lon: {lon}")
    image_path = f"images/panorama_{lat}_{lon}.jpg"
    if os.path.exists(image_path):
        return {"image_path": "/" + image_path}

    session = request.app.state.session
    pano = await streetview.find_panorama_async(lat, lon, session=session)
    if not pano:
        return {"error": "Panorama not found"}

    print(f"Found panorama: {pano}")
    print(f"Available zoom levels: {pano.image_sizes}")

    try:
        await download_panorama_image(pano, image_path)
    except RuntimeError as e:
        return {"error": str(e)}

    return {"image_path": "/" + image_path}


class DownloadPanoRequest(BaseModel):
    pano_id: str
    download_depth: bool = False


@router.post("/download_pano")
async def download_pano(req: DownloadPanoRequest, request: Request):
    try:
        pano_id = req.pano_id
        image_path = f"images/pano_{pano_id}.jpg"
        depth_path = f"images/pano_{pano_id}_depth.npy"

        session = request.app.state.session
        pano = await streetview.find_panorama_by_id_async(
            pano_id, session=session, download_depth=req.download_depth
        )

        if not pano:
            return {"error": "not found"}

        os.makedirs("images", exist_ok=True)

        if not os.path.exists(image_path):
            await download_panorama_image(pano, image_path)

        if req.download_depth and pano.depth and not os.path.exists(depth_path):
            np.save(depth_path, pano.depth.data)

        return {
            "id": pano.id,
            "lat": pano.lat,
            "lon": pano.lon,
            "date": str(pano.date) if pano.date else None,
            "heading": pano.heading,
            "pitch": pano.pitch,
            "has_depth": bool(pano.depth),
        }
    except Exception as e:
        print(f"Error in download_pano: {e}")
        return {"error": str(e)}


class BatchDownloadRequest(BaseModel):
    pano_ids: List[str]
    download_depth: bool = False
    metadata: list = []


def _cleanup_temp_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@router.post("/batch_download")
async def batch_download(req: BatchDownloadRequest, background_tasks: BackgroundTasks):
    out_dir = f"outputs_{uuid.uuid4().hex}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("images", exist_ok=True)

    downloaded_files = []
    for pano_id in req.pano_ids:
        image_path = f"images/pano_{pano_id}.jpg"
        depth_path = f"images/pano_{pano_id}_depth.npy"
        if os.path.exists(image_path):
            downloaded_files.append(image_path)
        if req.download_depth and os.path.exists(depth_path):
            downloaded_files.append(depth_path)

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"spatial_order": req.pano_ids, "nodes": req.metadata}, f, indent=2)
    downloaded_files.append(metadata_path)

    zip_path = os.path.join(out_dir, "panoramas.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in downloaded_files:
            zf.write(fpath, os.path.basename(fpath))

    background_tasks.add_task(_cleanup_temp_dir, out_dir)
    return FileResponse(zip_path, media_type="application/x-zip-compressed", filename="panoramas.zip")
