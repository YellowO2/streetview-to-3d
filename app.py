"""
Gradio interface for Street View to 3DGS.

Run locally:  python gradio_app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

import asyncio
import html as html_lib
import io
import math
import os
import re
import shutil
import time
import uuid

import aiohttp
import gradio as gr
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    if ON_SPACES:
        GPU = spaces.GPU(duration=108)
        GPU_EDIT = spaces.GPU(duration=72)
    else:
        GPU = lambda fn: fn
        GPU_EDIT = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn  # no-op outside HF Spaces
    GPU_EDIT = lambda fn: fn
    ON_SPACES = False

from streetlevel import streetview
from streetlevel.lookaround import lookaround as apple_lookaround
from streetlevel.lookaround.auth import Authenticator as AppleAuthenticator
from streetlevel.lookaround.reproject import to_equirectangular as apple_to_equirectangular
from streetlevel.geo import wgs84_to_tile_coord
from services.download_street_panorama import download_panorama_image
from prompts.presets import PRESET_NAMES, build_prompt, get_preset

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps/",
    "Accept-Language": "en-US,en;q=0.9",
}

# Auto-pick up to N nearest neighbors as DA3 depth-support panos.
MAX_SUPPORT_PANOS = 2

# Apple Look Around face zoom (0=full res/slow, 7=lowest). 3 is a fast, decent-quality default.
APPLE_ZOOM = 3
APPLE_CANDIDATE_COUNT = 5

_apple_auth = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
SPLATS_DIR = os.path.join(PROJECT_ROOT, "splats")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLATS_DIR, exist_ok=True)


# ── async helpers ──────────────────────────────────────────────────────────────


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _format_date(d):
    if d is None:
        return "unknown date"
    s = f"{d.year:04d}-{d.month:02d}"
    if getattr(d, "day", None):
        s += f"-{d.day:02d}"
    return s


def _pano_to_meta(pano):
    """Shared metadata shape for a resolved StreetViewPanorama, however it was found."""
    neighbors = []
    for item in pano.links or pano.neighbors:
        n = item.pano if hasattr(item, "pano") else item
        if n and n.lat is not None:
            neighbors.append({"id": n.id, "lat": n.lat, "lon": n.lon})

    dates = [{"id": pano.id, "label": _format_date(pano.date)}]
    for h in pano.historical or []:
        dates.append({"id": h.id, "label": _format_date(h.date)})

    return {
        "id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "date": _format_date(pano.date),
        "neighbors": neighbors,
        "dates": dates,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
    }


async def _fetch_pano(lat, lon):
    """Fetch the newest pano at a location, with neighbor + historical-date stubs."""
    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_async(lat, lon, session=session)
        if not pano:
            return None
        return _pano_to_meta(pano)


async def _fetch_pano_by_id(pano_id):
    """Fetch pano metadata for a specific panorama ID (e.g. a historical capture)."""
    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        return _pano_to_meta(pano)


async def _download(lat, lon):
    """Download a pano by lat/lon, return absolute path."""
    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_async(lat, lon, session=session)
        if not pano:
            return None
        img_path = os.path.join(IMAGES_DIR, f"pano_{pano.id}.jpg")
        if not os.path.exists(img_path):
            await download_panorama_image(pano, img_path)
        return img_path


async def _download_by_id(pano_id):
    """Download a pano by its exact ID, return absolute path."""
    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        img_path = os.path.join(IMAGES_DIR, f"pano_{pano.id}.jpg")
        if not os.path.exists(img_path):
            await download_panorama_image(pano, img_path)
        return img_path


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _apple_nearby_panos(lat, lon):
    """All Look Around panos on the target tile + its 8 neighbors, keyed by ID."""
    tx, ty = wgs84_to_tile_coord(lat, lon, 17)
    seen = {}
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            tile = apple_lookaround.get_coverage_tile(tx + dx, ty + dy)
            for p in tile.panos:
                seen[p.id] = p
    return seen


def _apple_candidates(lat, lon, k=APPLE_CANDIDATE_COUNT):
    """Nearest k Look Around panos to (lat, lon), sorted by distance."""
    panos = _apple_nearby_panos(lat, lon)
    return sorted(panos.values(), key=lambda p: _haversine_m(lat, lon, p.lat, p.lon))[:k]


def _apple_pano_to_meta(pano):
    """Shared metadata shape for a resolved LookaroundPanorama."""
    return {
        "id": pano.id,
        "build_id": pano.build_id,
        "lat": pano.lat,
        "lon": pano.lon,
        "date": _format_date(pano.date),
        "neighbors": [],
        "dates": [],
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
    }


def _get_apple_auth():
    global _apple_auth
    if _apple_auth is None:
        _apple_auth = AppleAuthenticator()
    return _apple_auth


def _download_lookaround(pano) -> str:
    """Fetch all 6 faces of a Look Around pano and stitch to equirectangular, cached by ID."""
    img_path = os.path.join(IMAGES_DIR, f"lookaround_{pano.id}.jpg")
    if os.path.exists(img_path):
        return img_path

    auth = _get_apple_auth()
    faces = [
        Image.open(io.BytesIO(apple_lookaround.get_panorama_face(pano, face_idx, APPLE_ZOOM, auth)))
        for face_idx in range(6)
    ]
    equi = apple_to_equirectangular(faces, pano.camera_metadata)
    equi.save(img_path, quality=92)
    return img_path


def _correct_slope(image_path: str, heading: float, pitch: float, roll: float, multiplier: float = 1.0) -> str:
    """De-tilt a panorama by its own heading/pitch/roll (Street View's
    upright-correction metadata), so its ground plane looks level before DA3
    depth/pose inference. Experimental — validating whether this improves
    DA3's view-consistency filtering on sloped streets. `multiplier` scales
    pitch/roll, to test whether a stronger-than-reported correction helps
    further. Returns a new file path; original untouched."""
    import cv2
    from components.ViewExtractor.Equirec2Perspec import rotate_equirectangular

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    corrected = rotate_equirectangular(img, heading=heading, roll=roll * multiplier, pitch=pitch * multiplier)
    out_path = image_path.rsplit(".", 1)[0] + "_leveled.jpg"
    cv2.imwrite(out_path, corrected)
    return out_path


# ── input parsing ──────────────────────────────────────────────────────────────


def _extract_lat_lon(raw: str):
    raw = raw.strip()
    m = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError("Use a Google Maps URL with /@lat,lon or paste lat,lon directly.")


# ── HTML / iframe builders ─────────────────────────────────────────────────────


def _file_url(abs_path: str) -> str:
    """Build Gradio's file-serving URL. /gradio_api/file= is the route in Gradio 5+."""
    return f"/gradio_api/file={abs_path}"


def _iframe(srcdoc: str, aspect: str = "16/9") -> str:
    escaped = html_lib.escape(srcdoc, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" sandbox="allow-scripts allow-same-origin" '
        f'style="width:100%;aspect-ratio:{aspect};border:none;border-radius:8px;background:#000">'
        "</iframe>"
    )


_MAP_PLACEHOLDER = _iframe(
    "<html><body style='margin:0;background:#1e1e2e;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Load a location to see it on the map</body></html>",
    aspect="16/9",
)
_PANO_PLACEHOLDER = _iframe(
    "<html><body style='margin:0;background:#111;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Panorama viewer</body></html>"
)
_SPLAT_PLACEHOLDER = _iframe(
    "<html><body style='margin:0;background:#111;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Generate a 3DGS scene to view it here</body></html>"
)


def _build_map(lat: float, lon: float) -> str:
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{margin:0;height:100%;width:100%}}</style>
</head><body><div id="map"></div>
<script>
var m = L.map('map').setView([{lat},{lon}], 18);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{maxZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);
L.circleMarker([{lat},{lon}],{{radius:9,color:'crimson',fillColor:'crimson',fillOpacity:0.9,weight:2}}).addTo(m);
</script></body></html>"""
    return _iframe(doc)


def _build_pano_viewer(img_url: str) -> str:
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;background:#000;overflow:hidden;cursor:grab}}body:active{{cursor:grabbing}}canvas{{display:block}}
#hint{{position:fixed;bottom:8px;right:8px;color:rgba(255,255,255,.4);font:11px sans-serif;pointer-events:none}}</style>
<script type="importmap">
{{"imports":{{"three":"https://unpkg.com/three@0.178.0/build/three.module.js"}}}}
</script></head><body><div id="hint">drag to look around</div>
<script type="module">
import * as THREE from 'three';
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const geo = new THREE.SphereGeometry(100, 64, 32); geo.scale(-1,1,1);
const mat = new THREE.MeshBasicMaterial();
scene.add(new THREE.Mesh(geo, mat));
new THREE.TextureLoader().load('{img_url}', t => {{ t.colorSpace=THREE.SRGBColorSpace; mat.map=t; mat.needsUpdate=true; }});
renderer.outputColorSpace = THREE.SRGBColorSpace;

let lon = 0, lat = 0, dragging = false, lx = 0, ly = 0;
renderer.domElement.addEventListener('pointerdown', e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }});
addEventListener('pointerup', () => dragging = false);
addEventListener('pointermove', e => {{
    if (!dragging) return;
    lon -= (e.clientX - lx) * 0.2; lat += (e.clientY - ly) * 0.2;
    lat = Math.max(-85, Math.min(85, lat));
    lx = e.clientX; ly = e.clientY;
}});
renderer.domElement.addEventListener('wheel', e => {{
    e.preventDefault();
    camera.fov = Math.max(20, Math.min(100, camera.fov + e.deltaY * 0.05));
    camera.updateProjectionMatrix();
}}, {{passive:false}});

addEventListener('resize', () => {{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}});
const tick = () => {{
    const phi = THREE.MathUtils.degToRad(90 - lat);
    const theta = THREE.MathUtils.degToRad(lon);
    camera.lookAt(
        100 * Math.sin(phi) * Math.cos(theta),
        100 * Math.cos(phi),
        100 * Math.sin(phi) * Math.sin(theta),
    );
    renderer.render(scene, camera);
}};
renderer.setAnimationLoop(tick);
document.addEventListener('visibilitychange', () => {{
    renderer.setAnimationLoop(document.hidden ? null : tick);
}});
</script></body></html>"""
    return _iframe(doc)


def _splat_viewer_with_download(ply_url: str) -> str:
    """Splat iframe + an inline download link below it. The link rides inside
    the same HTML payload as the viewer, so it survives backgrounded-tab
    WebSocket throttling that would otherwise drop separate component updates."""
    download_link = (
        f'<a href="{ply_url}" download '
        f'style="display:inline-block;margin-top:8px;padding:10px 16px;'
        f'background:#5b47d1;color:#fff;text-decoration:none;border-radius:8px;'
        f'font:600 14px sans-serif;">⬇ Download 3DGS (.ply)</a>'
    )
    return f'<div>{_build_splat_iframe(ply_url)}{download_link}</div>'


def _pointcloud_download_view(ply_url: str) -> str:
    """Plain download link for the raw DA3 point cloud — no live viewer, since
    the splat viewer's SplatMesh renderer expects 3DGS splat data, not a
    colored point cloud."""
    return (
        '<div style="display:flex;align-items:center;justify-content:center;'
        'aspect-ratio:16/9;background:#111;border-radius:8px">'
        f'<a href="{ply_url}" download '
        'style="display:inline-block;padding:10px 16px;'
        'background:#5b47d1;color:#fff;text-decoration:none;border-radius:8px;'
        'font:600 14px sans-serif;">⬇ Download DA3 point cloud (.ply)</a>'
        "</div>"
    )


def _build_splat_iframe(ply_url: str) -> str:
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;background:#000;overflow:hidden;font:14px sans-serif;color:#bbb}}canvas{{display:block}}
#hint{{position:fixed;bottom:8px;right:8px;color:rgba(255,255,255,.4);font:11px sans-serif;pointer-events:none}}
#loading{{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
  background:#000;transition:opacity .4s;pointer-events:none;padding:1em}}
#loading.gone{{opacity:0}}
.dot{{display:inline-block;animation:blink 1.4s infinite both}}
.dot:nth-child(2){{animation-delay:.2s}}.dot:nth-child(3){{animation-delay:.4s}}
@keyframes blink{{0%,80%,100%{{opacity:0}}40%{{opacity:1}}}}</style>
<script type="importmap">
{{"imports":{{
    "three":"https://unpkg.com/three@0.178.0/build/three.module.js",
    "three/addons/":"https://unpkg.com/three@0.178.0/examples/jsm/",
    "@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"
}}}}
</script></head><body>
<div id="loading">Loading 3DGS scene<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span><br><small style="color:#666">(a few hundred MB — ~30s)</small></div>
<div id="hint">drag to move</div>
<script type="module">
import * as THREE from 'three';
import {{ SplatMesh, SparkControls }} from '@sparkjsdev/spark';
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const controls = new SparkControls({{canvas: renderer.domElement}});
const splat = new SplatMesh({{url: '{ply_url}'}});
splat.quaternion.set(1, 0, 0, 0);  // flip 180° around X — splats come out upside-down otherwise
scene.add(splat);
const hideLoading = () => {{
    const el = document.getElementById('loading');
    if (el) {{ el.classList.add('gone'); setTimeout(() => el.remove(), 500); }}
}};
if (splat.initialized && typeof splat.initialized.then === 'function') {{
    splat.initialized.then(hideLoading).catch(hideLoading);
}} else {{
    // Fallback: hide once the splat has any visible content (numSplats > 0).
    const check = () => {{
        if (splat.numSplats && splat.numSplats > 0) hideLoading();
        else setTimeout(check, 500);
    }};
    check();
    setTimeout(hideLoading, 90000);  // hard cap
}}
addEventListener('resize', () => {{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}});
const tick = () => {{ controls.update(camera); renderer.render(scene, camera); }};
renderer.setAnimationLoop(tick);
document.addEventListener('visibilitychange', () => {{
    renderer.setAnimationLoop(document.hidden ? null : tick);
}});
</script></body></html>"""
    return _iframe(doc)


# ── GPU pipeline ───────────────────────────────────────────────────────────────


_pipeline = None
_flux_editor = None
if ON_SPACES:
    from editors.flux_editor import FluxEditor
    _flux_editor = FluxEditor(offload=False)


@GPU_EDIT
def _run_editor_gpu(image_path, prompt, mode, output_path):
    global _flux_editor
    if _flux_editor is None:
        from editors.flux_editor import FluxEditor
        _flux_editor = FluxEditor(offload=True)
    _flux_editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    return output_path


@GPU
def _run_pipeline_gpu(target_appearance_path, output_dir, scale_mode, gs_backend, support_paths=None, target_depth_path=None):
    global _pipeline
    if _pipeline is None:
        from panoramic_to_3dgs import Pipeline
        from config import load_pipeline_config
        _pipeline = Pipeline(load_pipeline_config())

    os.makedirs(output_dir, exist_ok=True)
    _pipeline.config.scale_mode = scale_mode
    _pipeline.config.gs_backend = gs_backend
    _pipeline.run(
        target_appearance_path=target_appearance_path,
        output_dir=output_dir,
        target_depth_path=target_depth_path,
        support_paths=support_paths,
    )

    ply = os.path.join(output_dir, "final_output.ply")
    return ply if os.path.exists(ply) else None


@GPU
def _run_pointcloud_gpu(target_depth_path, output_dir, support_paths=None):
    global _pipeline
    if _pipeline is None:
        from panoramic_to_3dgs import Pipeline
        from config import load_pipeline_config
        _pipeline = Pipeline(load_pipeline_config())

    os.makedirs(output_dir, exist_ok=True)
    _pipeline.run_da3_pointcloud(
        target_depth_path=target_depth_path,
        output_dir=output_dir,
        support_paths=support_paths,
    )

    ply = os.path.join(output_dir, "da3_pointcloud.ply")
    return ply if os.path.exists(ply) else None


# ── handlers ───────────────────────────────────────────────────────────────────


def handle_load(url_input):
    try:
        lat, lon = _extract_lat_lon(url_input)
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        meta = _run_async(_fetch_pano(lat, lon))
    except Exception as e:
        raise gr.Error(str(e))
    if not meta:
        raise gr.Error("Panorama not found at that location.")

    try:
        img_path = _run_async(_download(meta["lat"], meta["lon"]))
    except Exception as e:
        raise gr.Error(f"Download failed: {e}")

    state = {
        "source": "streetview",
        "image_path": img_path,
        "original_image_path": img_path,
        **meta,
    }
    pano_choices = [(f"Google · {d['label']}", f"google:{d['id']}") for d in meta["dates"]]
    try:
        for p in _apple_candidates(meta["lat"], meta["lon"]):
            dist_m = _haversine_m(meta["lat"], meta["lon"], p.lat, p.lon)
            pano_choices.append((f"Apple · {dist_m:.0f}m · {_format_date(p.date)}", f"apple:{p.id}"))
    except Exception:
        pass  # Look Around coverage lookup is best-effort; Google picker still works without it.

    return (
        _build_map(meta["lat"], meta["lon"]),
        _build_pano_viewer(_file_url(img_path)),
        state,
        gr.update(choices=pano_choices, value=f"google:{meta['id']}", visible=len(pano_choices) > 1),
    )


def handle_select_pano(pano_state, selected_value):
    if not pano_state or pano_state.get("source") not in ("streetview", "lookaround"):
        raise gr.Error("Load a Street View location first.")
    current_value = ("google" if pano_state["source"] == "streetview" else "apple") + ":" + str(pano_state.get("id"))
    if not selected_value or selected_value == current_value:
        return gr.update(), gr.update(), pano_state

    source, pano_id = selected_value.split(":", 1)

    if source == "google":
        try:
            meta = _run_async(_fetch_pano_by_id(pano_id))
        except Exception as e:
            raise gr.Error(f"Failed to load that date: {e}")
        if not meta:
            raise gr.Error("Panorama not found for that date.")
        try:
            img_path = _run_async(_download_by_id(meta["id"]))
        except Exception as e:
            raise gr.Error(f"Download failed: {e}")
        state = {"source": "streetview", "image_path": img_path, "original_image_path": img_path, **meta}
    else:
        try:
            pano = _apple_nearby_panos(pano_state["lat"], pano_state["lon"]).get(int(pano_id))
        except Exception as e:
            raise gr.Error(f"Failed to look up that Apple panorama: {e}")
        if not pano:
            raise gr.Error("Apple panorama not found near that location.")
        try:
            img_path = _download_lookaround(pano)
        except Exception as e:
            raise gr.Error(f"Download failed: {e}")
        meta = _apple_pano_to_meta(pano)
        state = {"source": "lookaround", "image_path": img_path, "original_image_path": img_path, **meta}

    return (
        _build_map(meta["lat"], meta["lon"]),
        _build_pano_viewer(_file_url(img_path)),
        state,
    )


def handle_edit(pano_state, prompt, preset_name, progress=gr.Progress()):
    if not pano_state or not pano_state.get("image_path"):
        raise gr.Error("Load or upload a panorama first.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter an edit prompt.")

    preset = get_preset(preset_name)
    mode = preset["mode"] if preset else "general"
    geom_preserving = preset["geom_preserving"] if preset else False

    src = pano_state["image_path"]
    out_path = os.path.join(IMAGES_DIR, f"edit_{uuid.uuid4().hex}.png")

    progress(0, desc="Running edit (~40s)...")
    try:
        _run_editor_gpu(src, prompt.strip(), mode, out_path)
    except Exception as e:
        raise gr.Error(f"Edit failed: {e}")

    new_state = {**pano_state, "image_path": out_path}
    if not geom_preserving:
        new_state["original_image_path"] = out_path

    return (
        _build_pano_viewer(_file_url(out_path)),
        new_state,
    )


def handle_upload(file_path):
    if not file_path:
        raise gr.Error("No file selected.")
    ext = os.path.splitext(file_path)[1] or ".jpg"
    dest = os.path.join(IMAGES_DIR, f"upload_{uuid.uuid4().hex}{ext}")
    shutil.copy(file_path, dest)
    state = {"source": "upload", "image_path": dest, "original_image_path": dest}
    return (_build_pano_viewer(_file_url(dest)), state)


def handle_generate(pano_state, scale_mode, output_mode, use_support_panos, correct_slope, slope_multiplier, progress=gr.Progress(track_tqdm=True)):
    if not pano_state or not pano_state.get("image_path"):
        raise gr.Error("Load or upload a panorama first.")

    yield _SPLAT_PLACEHOLDER

    target_path = pano_state["image_path"]
    target_depth_path = pano_state.get("original_image_path", target_path)

    if (
        output_mode == "DA3 Point Cloud"
        and correct_slope
        and pano_state.get("heading") is not None
        and pano_state.get("pitch") is not None
        and pano_state.get("roll") is not None
    ):
        try:
            target_depth_path = _correct_slope(
                target_depth_path,
                pano_state["heading"],
                pano_state["pitch"],
                pano_state["roll"],
                multiplier=slope_multiplier,
            )
        except Exception as e:
            raise gr.Error(f"Slope correction failed: {e}")

    support_paths = []

    neighbors = (
        pano_state.get("neighbors", [])[:MAX_SUPPORT_PANOS]
        if use_support_panos and pano_state.get("source") == "streetview"
        else []
    )
    for i, n in enumerate(neighbors):
        try:
            progress(0, desc=f"Downloading support pano {i+1}/{len(neighbors)}...")
            p = _run_async(_download(n["lat"], n["lon"]))
            if p:
                support_paths.append(p)
        except Exception:
            pass

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    t_start = time.time()

    if output_mode == "DA3 Point Cloud":
        try:
            ply_path = _run_pointcloud_gpu(target_depth_path, output_dir, support_paths=support_paths)
        except Exception as e:
            raise gr.Error(f"Pipeline failed: {e}")

        if not ply_path or not os.path.exists(ply_path):
            raise gr.Error("Pipeline finished but no point cloud produced.")

        elapsed = time.time() - t_start
        progress(1.0, desc=f"Done! {1 + len(support_paths)} panos, {elapsed:.0f}s")
        yield _pointcloud_download_view(_file_url(ply_path))
        return

    try:
        ply_path = _run_pipeline_gpu(
            target_path,
            output_dir,
            scale_mode,
            "sharp",
            support_paths=support_paths,
            target_depth_path=target_depth_path if target_depth_path != target_path else None,
        )
    except Exception as e:
        raise gr.Error(f"Pipeline failed: {e}")

    if not ply_path or not os.path.exists(ply_path):
        raise gr.Error("Pipeline finished but no PLY produced.")

    elapsed = time.time() - t_start
    progress(1.0, desc=f"Done! {1 + len(support_paths)} panos, {elapsed:.0f}s")
    yield _splat_viewer_with_download(_file_url(ply_path))



# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Street View to 3DGS", css=".no-pad { padding-left: 0 !important; padding-right: 0 !important; }") as demo:
    gr.Markdown(
        "# Street View to 3DGS\n"
        "Convert a Google Street View location into a 3D Gaussian Splat scene. "
        "[[GitHub](https://github.com/YellowO2/streetview-to-3dgs)]\n\n"
        "**1.** Paste a Google Maps URL → **2.** Optionally edit the panorama → **3.** Generate 3DGS"
    )

    pano_state = gr.State(None)

    # ── Step 1 ─────────────────────────────────────────────────────────────────
    gr.Markdown("## Step 1. Load panorama")
    with gr.Row(equal_height=True):
        url_input = gr.Textbox(
            placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
            show_label=False,
            container=False,
            scale=5,
        )
        load_btn = gr.Button("Load", variant="primary", scale=1, min_width=80)
        upload_pano = gr.UploadButton(
            "Upload panorama (beta)",
            file_types=[".jpg", ".jpeg", ".png"],
            scale=1,
            min_width=80,
        )

    pano_dropdown = gr.Dropdown(
        label="Source pano",
        info="Other captures of this spot — Google Street View dates and nearby Apple Look Around panos.",
        choices=[],
        visible=False,
    )

    with gr.Row(equal_height=True):
        map_view = gr.HTML(_MAP_PLACEHOLDER, elem_classes="no-pad")
        pano_view = gr.HTML(_PANO_PLACEHOLDER, elem_classes="no-pad")

    pano_download = gr.DownloadButton(
        label="⬇  Download current panorama",
        visible=False,
        size="sm",
    )

    gr.Markdown("Edit panorama (optional) — ~0.7 min")
    with gr.Row(equal_height=True):
        edit_prompt = gr.Textbox(
            placeholder="Pick a preset or type your own (e.g. add snow)",
            show_label=False,
            container=False,
            scale=4,
            lines=2,
        )
        edit_preset = gr.Dropdown(
            choices=PRESET_NAMES,
            value="(none)",
            show_label=False,
            container=False,
            scale=1,
        )
        edit_btn = gr.Button("Edit", scale=1, min_width=80)

    def _apply_preset(preset_name):
        prompt = build_prompt(preset_name)
        return gr.update(value=prompt) if prompt else gr.update(value="")

    edit_preset.change(
        fn=_apply_preset,
        inputs=[edit_preset],
        outputs=[edit_prompt],
    )

    # ── Step 2 ─────────────────────────────────────────────────────────────────
    gr.Markdown("## Step 2. Generate 3DGS ~1.8 min")
    with gr.Row(equal_height=True):
        scale_mode = gr.Dropdown(
            choices=["da3_y_ground", "da3_2dgrid_global"],
            value="da3_y_ground",
            label="Scale mode",
            info="How depth is aligned to the scene.",
            scale=2,
        )
        output_mode = gr.Dropdown(
            choices=["3D Gaussian Splat", "DA3 Point Cloud"],
            value="3D Gaussian Splat",
            label="Output type",
            info="Generate a 3DGS or a point cloud.",
            scale=2,
        )
        use_support_panos = gr.Checkbox(
            value=True,
            label="Use supporting panoramas",
            info="Nearby panos as DA3 depth/pose context.",
            scale=1,
        )
        correct_slope = gr.Checkbox(
            value=False,
            label="Correct for slope (experimental)",
            info="De-tilt the target pano using its pitch/roll before DA3.",
            scale=1,
            visible=False,
        )
        slope_multiplier = gr.Number(
            value=1.0,
            label="Slope correction ×",
            info="Scales the pitch/roll correction. Try >1 if 1x isn't enough.",
            scale=1,
            visible=False,
        )
        generate_btn = gr.Button("Generate", variant="primary", scale=1, min_width=160)

    splat_view = gr.HTML(_SPLAT_PLACEHOLDER)

    def _refresh_pano_download(state):
        path = (state or {}).get("image_path") if state else None
        if path and os.path.exists(path):
            return gr.update(visible=True, value=path)
        return gr.update(visible=False, value=None)

    pano_state.change(
        fn=_refresh_pano_download,
        inputs=[pano_state],
        outputs=[pano_download],
    )

    load_btn.click(
        fn=handle_load,
        inputs=[url_input],
        outputs=[map_view, pano_view, pano_state, pano_dropdown],
    )

    pano_dropdown.change(
        fn=handle_select_pano,
        inputs=[pano_state, pano_dropdown],
        outputs=[map_view, pano_view, pano_state],
    )

    upload_pano.upload(
        fn=handle_upload,
        inputs=[upload_pano],
        outputs=[pano_view, pano_state],
    ).then(
        fn=lambda: gr.update(choices=[], value=None, visible=False),
        outputs=[pano_dropdown],
    )

    edit_btn.click(
        fn=handle_edit,
        inputs=[pano_state, edit_prompt, edit_preset],
        outputs=[pano_view, pano_state],
        show_progress="minimal",
        show_progress_on=[pano_view],
    )

    output_mode.change(
        fn=lambda mode: gr.update(visible=mode == "3D Gaussian Splat"),
        inputs=[output_mode],
        outputs=[scale_mode],
    )
    output_mode.change(
        fn=lambda mode: (
            gr.update(visible=mode == "DA3 Point Cloud"),
            gr.update(visible=mode == "DA3 Point Cloud"),
        ),
        inputs=[output_mode],
        outputs=[correct_slope, slope_multiplier],
    )

    generate_btn.click(
        fn=handle_generate,
        inputs=[pano_state, scale_mode, output_mode, use_support_panos, correct_slope, slope_multiplier],
        outputs=[splat_view],
        show_progress="minimal",
        show_progress_on=[splat_view],
    )


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[IMAGES_DIR, SPLATS_DIR],
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(),
    )
