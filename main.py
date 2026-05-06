import asyncio
import itertools
import math
import os
import uuid
import zipfile
import shutil
import json
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from fastapi.staticfiles import StaticFiles
from streetlevel import streetview
from streetlevel.dataclasses import Tile
from streetlevel.streetview import streetview as _sv_module
from streetlevel.streetview.util import is_third_party_panoid
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientSession
from PIL import UnidentifiedImageError

# Patch: cbk0.google.com is deprecated and returns 403. Use the current endpoint.
_NEW_TILE_URL = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={0:}&x={1:}&y={2:}&zoom={3:}"
_THIRD_PARTY_TILE_URL = "https://lh3.ggpht.com/p/{0:}=x{1:}-y{2:}-z{3:}"


def _patched_generate_tile_list(pano, zoom):
    img_size = pano.image_sizes[zoom]
    cols = math.ceil(img_size.x / pano.tile_size.x)
    rows = math.ceil(img_size.y / pano.tile_size.y)

    # Try using the official tile URL for both first
    url = _NEW_TILE_URL

    return [
        Tile(x, y, url.format(pano.id, x, y, zoom))
        for x, y in itertools.product(range(cols), range(rows))
    ]


_sv_module._generate_tile_list = _patched_generate_tile_list

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    app.state.session = ClientSession()


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.session.close()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/debug/tile")
async def debug_tile(panoid: str = "LoA_9MjJTb7FXLEUlzyrpw", zoom: int = 3):
    url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x=0&y=0&zoom={zoom}"
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


@app.get("/metadata")
async def get_metadata(lat: float = None, lon: float = None, pano_id: str = None):
    print(f"Received metadata request - lat: {lat}, lon: {lon}, pano_id: {pano_id}")
    if not pano_id and (lat is None or lon is None):
        return {"error": "Must provide either pano_id OR lat and lon"}

    try:
        if pano_id:
            pano = await streetview.find_panorama_by_id_async(
                pano_id, session=app.state.session
            )
        else:
            pano = await streetview.find_panorama_async(
                lat, lon, session=app.state.session
            )

        if not pano:
            return {"error": "Panorama not found"}

        links = []
        if pano.links:
            for link in pano.links:
                if link.pano:
                    links.append(
                        {
                            "id": link.pano.id,
                            "lat": link.pano.lat,
                            "lon": link.pano.lon,
                            "direction": link.direction,
                        }
                    )

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


@app.get("/panorama/{lat},{lon}")
async def panorama(lat: float, lon: float):

    print(f"Received request for panorama at lat: {lat}, lon: {lon}")
    image_path = f"images/panorama_{lat}_{lon}.jpg"
    try:
        with open(image_path, "rb") as f:
            return {"image_path": "/" + image_path}
    except FileNotFoundError:
        pass

    pano = await streetview.find_panorama_async(lat, lon, session=app.state.session)
    if not pano:
        return {"error": "Panorama not found"}
    print(f"Found panorama: {pano}")
    print(f"Available zoom levels: {pano.image_sizes}")

    for attempt in range(4):
        try:
            # Fresh session per attempt — avoids reusing a session Google has fingerprinted.
            # zoom=2 fetches ~8 tiles instead of ~32 at zoom=3, much less likely to trigger rate limits.
            async with ClientSession() as dl_session:
                await streetview.download_panorama_async(
                    pano, image_path, session=dl_session, zoom=5
                )
            break
        except (UnidentifiedImageError, Exception) as e:
            if attempt == 3:
                return {"error": f"Failed to download panorama after retries: {e}"}
            wait = 3**attempt  # 1s, 3s, 9s
            print(
                f"Tile fetch failed (attempt {attempt + 1}), retrying in {wait}s: {e}"
            )
            await asyncio.sleep(wait)

    return {"image_path": "/" + image_path}


class BatchDownloadRequest(BaseModel):
    pano_ids: List[str]
    download_depth: bool = False
    metadata: list = []


class DownloadPanoRequest(BaseModel):
    pano_id: str
    download_depth: bool = False


def cleanup_temp_dir(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


@app.post("/download_pano")
async def download_pano(req: DownloadPanoRequest):
    try:
        pano_id = req.pano_id
        image_path = f"images/pano_{pano_id}.jpg"
        depth_path = f"images/pano_{pano_id}_depth.npy"

        pano = await streetview.find_panorama_by_id_async(
            pano_id, session=app.state.session, download_depth=req.download_depth
        )

        if not pano:
            return {"error": "not found"}

        os.makedirs("images", exist_ok=True)

        if not os.path.exists(image_path):
            async with ClientSession() as dl_session:
                await streetview.download_panorama_async(
                    pano, image_path, session=dl_session, zoom=5
                )

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


@app.post("/batch_download")
async def batch_download(req: BatchDownloadRequest, background_tasks: BackgroundTasks):
    out_dir = f"outputs_{uuid.uuid4().hex}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("images", exist_ok=True)

    zip_path = os.path.join(out_dir, "panoramas.zip")

    downloaded_files = []

    for pano_id in req.pano_ids:
        # Pull from local cache
        image_path = f"images/pano_{pano_id}.jpg"
        depth_path = f"images/pano_{pano_id}_depth.npy"

        if os.path.exists(image_path):
            downloaded_files.append(image_path)

        if req.download_depth and os.path.exists(depth_path):
            downloaded_files.append(depth_path)

    # Dump metadata to json
    metadata_path = os.path.join(out_dir, "metadata.json")

    output_metadata = {"spatial_order": req.pano_ids, "nodes": req.metadata}

    with open(metadata_path, "w") as f:
        json.dump(output_metadata, f, indent=2)
    downloaded_files.append(metadata_path)

    # create zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in downloaded_files:
            zf.write(fpath, os.path.basename(fpath))

    # Clean up output directory after returning the file
    background_tasks.add_task(cleanup_temp_dir, out_dir)

    return FileResponse(
        zip_path, media_type="application/x-zip-compressed", filename="panoramas.zip"
    )


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/splats", StaticFiles(directory="splats"), name="splats")
