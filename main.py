import asyncio
import itertools
import math
from fastapi import FastAPI
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
    url = _THIRD_PARTY_TILE_URL if is_third_party_panoid(pano.id) else _NEW_TILE_URL
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

TILE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.google.com/maps/",
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@app.on_event("startup")
async def startup_event():
    app.state.session = ClientSession(headers=TILE_HEADERS)


@app.on_event("shutdown")
async def shutdown_event():
    await app.state.session.close()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/debug/tile")
async def debug_tile(panoid: str = "LoA_9MjJTb7FXLEUlzyrpw", zoom: int = 3):
    url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoid}&x=0&y=0&zoom={zoom}"
    async with ClientSession(headers=TILE_HEADERS) as s:
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
            async with ClientSession(headers=TILE_HEADERS) as dl_session:
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


app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/splats", StaticFiles(directory="splats"), name="splats")
