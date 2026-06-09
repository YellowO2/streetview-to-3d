"""
Gradio interface for Street View to 3DGS.

Run locally:  python gradio_app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

import asyncio
import html as html_lib
import os
import re
import time
import uuid

import aiohttp
import gradio as gr

try:
    import spaces

    # spaces is also installed locally via requirements.txt, so gate on SPACE_ID
    # which HF Spaces always sets but local machines don't have.
    ON_SPACES = bool(os.getenv("SPACE_ID"))
    if ON_SPACES:
        GPU = spaces.GPU(duration=180)
        GPU_EDIT = spaces.GPU(duration=120)
    else:
        GPU = lambda fn: fn
        GPU_EDIT = lambda fn: fn
except ImportError:
    GPU = lambda fn: fn  # no-op outside HF Spaces
    GPU_EDIT = lambda fn: fn
    ON_SPACES = False

from streetlevel import streetview
from services.download_street_panorama import download_panorama_image
from prompts.lighting import PRESET_NAMES, build_prompt

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


async def _fetch_pano(lat, lon):
    """Fetch pano metadata + neighbor stubs."""
    async with aiohttp.ClientSession(headers=_BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_async(lat, lon, session=session)
        if not pano:
            return None
        neighbors = []
        for item in pano.links or pano.neighbors:
            n = item.pano if hasattr(item, "pano") else item
            if n and n.lat is not None:
                neighbors.append({"id": n.id, "lat": n.lat, "lon": n.lon})
        return {"id": pano.id, "lat": pano.lat, "lon": pano.lon, "neighbors": neighbors}


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
new THREE.TextureLoader().load('{img_url}', t => {{ mat.map=t; mat.needsUpdate=true; }});

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


@GPU
def _run_pipeline_gpu(panorama_paths, output_dir, scale_mode, gs_backend, depth_panorama_paths=None):
    from panoramic_to_3dgs import Pipeline
    from config import load_pipeline_config

    os.makedirs(output_dir, exist_ok=True)
    config = load_pipeline_config()
    config.scale_mode = scale_mode
    config.gs_backend = gs_backend
    Pipeline(config).run(
        panorama_paths=panorama_paths,
        output_dir=output_dir,
        target_pano_id=0,
        depth_panorama_paths=depth_panorama_paths,
    )

    ply = os.path.join(output_dir, "final_output.ply")
    return ply if os.path.exists(ply) else None


_editor = None
_flux_editor = None
if ON_SPACES:
    # Eager-load at module level: ZeroGPU's boot-time CUDA mode supports the bulk
    # .to("cuda") for huge multi-shard pipelines. Lazy loading inside @spaces.GPU
    # crashes with NVML errors. Boot is slow (~5min cold), but each edit is fast
    # (~15s) and burns minimal quota.
    from components.ImageCleaner.ImageCleaner import ImageCleaner
    _editor = ImageCleaner(offload=False)

    # Pre-download FLUX.2-klein weights in the background so the container starts
    # immediately. By the time a user triggers a FLUX edit, weights will be cached.
    import threading
    from huggingface_hub import snapshot_download

    def _prefetch_flux():
        snapshot_download("black-forest-labs/FLUX.2-klein-9B")

    threading.Thread(target=_prefetch_flux, daemon=True).start()


@GPU_EDIT
def _run_editor_gpu(image_path, prompt, mode, output_path, edit_model="Qwen-Image-Edit-2511"):
    global _editor, _flux_editor
    if edit_model == "FLUX.2-klein":
        if _flux_editor is None:
            from editors.flux_editor import FluxEditor
            _flux_editor = FluxEditor(offload=False)
        _flux_editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    else:
        if _editor is None:
            from components.ImageCleaner.ImageCleaner import ImageCleaner
            _editor = ImageCleaner(offload=True)
        _editor.edit(image_path, prompt, mode=mode, output_path=output_path)
    return output_path


# ── handlers ───────────────────────────────────────────────────────────────────


def handle_load(url_input):
    try:
        lat, lon = _extract_lat_lon(url_input)
    except ValueError as e:
        return str(e), _MAP_PLACEHOLDER, _PANO_PLACEHOLDER, None

    try:
        meta = _run_async(_fetch_pano(lat, lon))
    except Exception as e:
        return f"Error: {e}", _MAP_PLACEHOLDER, _PANO_PLACEHOLDER, None
    if not meta:
        return (
            "Panorama not found at that location.",
            _MAP_PLACEHOLDER,
            _PANO_PLACEHOLDER,
            None,
        )

    try:
        img_path = _run_async(_download(meta["lat"], meta["lon"]))
    except Exception as e:
        return (
            f"Download failed: {e}",
            _build_map(meta["lat"], meta["lon"]),
            _PANO_PLACEHOLDER,
            None,
        )

    state = {
        "source": "streetview",
        "image_path": img_path,
        "original_image_path": img_path,
        **meta,
    }
    status = f"Loaded pano {meta['id']} at ({meta['lat']:.5f}, {meta['lon']:.5f}). Click Generate 3DGS to continue."
    return (
        status,
        _build_map(meta["lat"], meta["lon"]),
        _build_pano_viewer(_file_url(img_path)),
        state,
    )


def handle_upload(file_path):
    """User uploaded their own equirectangular panorama."""
    if not file_path:
        return "No file uploaded.", _MAP_PLACEHOLDER, _PANO_PLACEHOLDER, None
    abs_path = os.path.abspath(file_path)
    state = {
        "source": "upload",
        "image_path": abs_path,
        "original_image_path": abs_path,
        "neighbors": [],
    }
    return (
        f"Uploaded panorama loaded. Click Generate 3DGS to continue.",
        _MAP_PLACEHOLDER,  # no map for uploads
        _build_pano_viewer(_file_url(abs_path)),
        state,
    )


EDIT_MODES = {
    "General edit": "general",
    "Remove people & vehicles": "remove_objects",
}


def handle_edit(pano_state, prompt, mode_label, preset_name, edit_model, progress=gr.Progress()):
    if not pano_state or not pano_state.get("image_path"):
        return (
            "Load a Street View location or upload a panorama first.",
            _PANO_PLACEHOLDER,
            pano_state,
        )
    if not prompt or not prompt.strip():
        return ("Enter an edit prompt.", gr.update(), pano_state)

    mode = EDIT_MODES.get(mode_label, "general")
    src = pano_state["image_path"]
    out_path = os.path.join(IMAGES_DIR, f"edit_{uuid.uuid4().hex}.png")

    progress(0, desc="Loading editor + running edit (~30s)...")
    t0 = time.time()
    try:
        _run_editor_gpu(src, prompt.strip(), mode, out_path, edit_model=edit_model)
    except Exception as e:
        return (f"Edit failed: {e}", gr.update(), pano_state)

    # Geometry-preserving edits (lighting presets) keep original_image_path
    # pinned so depth runs against the un-relit pano. Object removal or
    # freeform edits assume geometry may have changed, so original moves too.
    geom_preserving = (
        preset_name and preset_name != "(none)" and mode == "general"
    )
    new_state = {**pano_state, "image_path": out_path}
    if not geom_preserving:
        new_state["original_image_path"] = out_path

    elapsed = time.time() - t0
    note = " (depth will use original pano)" if geom_preserving else ""
    return (
        f"Edit applied ({elapsed:.0f}s){note}. Click Generate 3DGS to splat the edited pano.",
        _build_pano_viewer(_file_url(out_path)),
        new_state,
    )


def handle_generate(pano_state, scale_mode, gs_backend, progress=gr.Progress(track_tqdm=True)):
    if not pano_state or not pano_state.get("image_path"):
        yield ("Load a Street View location or upload a panorama first.", _SPLAT_PLACEHOLDER)
        return

    # Clear any previous splat from the viewer immediately so the old scene
    # doesn't linger during the next 2-min run.
    yield ("Starting...", _SPLAT_PLACEHOLDER)

    target_path = pano_state["image_path"]
    target_depth_path = pano_state.get("original_image_path", target_path)
    support_paths = []

    # Only Street View sources have neighbor metadata to fetch depth-support panos from.
    neighbors = (
        pano_state.get("neighbors", [])[:MAX_SUPPORT_PANOS]
        if pano_state.get("source") == "streetview"
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
    panorama_paths = [target_path, *support_paths]
    # Support panos are always originals (never edited), so depth list just
    # swaps the target slot. Pass None when nothing differs to keep the
    # pipeline path identical to pre-split behavior.
    depth_paths = (
        [target_depth_path, *support_paths]
        if target_depth_path != target_path
        else None
    )

    t_start = time.time()
    try:
        ply_path = _run_pipeline_gpu(
            panorama_paths, output_dir, scale_mode, gs_backend, depth_panorama_paths=depth_paths
        )
    except Exception as e:
        yield (f"Pipeline failed: {e}", _SPLAT_PLACEHOLDER)
        return

    if not ply_path or not os.path.exists(ply_path):
        yield ("Pipeline finished but no PLY produced.", _SPLAT_PLACEHOLDER)
        return

    elapsed = time.time() - t_start
    progress(1.0, desc="Done!")
    yield (
        f"3DGS ready ({len(panorama_paths)} panos, {elapsed:.0f}s). Loading viewer (~30s for large scenes)...",
        _splat_viewer_with_download(_file_url(ply_path)),
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Street View to 3DGS") as demo:
    gr.Markdown(
        "# Street View to 3DGS\n"
        "Paste a Google Maps URL or `lat,lon`, load the panorama, then generate a 3D Gaussian Splat scene."
    )

    pano_state = gr.State(None)

    with gr.Tabs():
        with gr.Tab("Street View"):
            with gr.Row(equal_height=True):
                url_input = gr.Textbox(
                    placeholder="Google Maps URL or lat,lon (e.g. 1.3237, 103.7555)",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                load_btn = gr.Button("Load", variant="primary", scale=1, min_width=80)
        with gr.Tab("Upload Panorama"):
            upload_input = gr.File(
                label="Upload an equirectangular panorama (.jpg / .png). This is likely to return suboptimal results due to certain optimisations not made.",
                file_types=["image"],
                type="filepath",
            )

    status = gr.Textbox(
        label="Status",
        value="Paste a Google Maps URL or upload a panorama to start.",
        interactive=False,
    )

    with gr.Row():
        map_view = gr.HTML(_MAP_PLACEHOLDER)
        pano_view = gr.HTML(_PANO_PLACEHOLDER)

    pano_download = gr.DownloadButton(
        label="⬇  Download current panorama",
        visible=False,
        size="sm",
    )

    gr.Markdown("---")
    gr.Markdown("### Edit panorama (optional)")
    with gr.Row(equal_height=True):
        lighting_preset = gr.Dropdown(
            choices=PRESET_NAMES,
            value="(none)",
            label="Time of day preset",
            scale=2,
        )
        edit_mode = gr.Dropdown(
            choices=list(EDIT_MODES.keys()),
            value="General edit",
            label="Edit mode",
            scale=2,
        )
        edit_model_selector = gr.Dropdown(
            choices=["Qwen-Image-Edit-2511", "FLUX.2-klein"],
            value="Qwen-Image-Edit-2511",
            label="Edit model",
            scale=2,
        )
    with gr.Row(equal_height=True):
        edit_prompt = gr.Textbox(
            placeholder="Pick a preset above to auto-fill, or type your own (e.g. remove the cars, add snow)",
            show_label=False,
            container=False,
            scale=4,
            lines=2,
        )
        edit_btn = gr.Button("Edit image", scale=1, min_width=120)

    def _apply_preset(preset_name):
        prompt = build_prompt(preset_name)
        return gr.update(value=prompt) if prompt else gr.update()

    lighting_preset.change(
        fn=_apply_preset,
        inputs=[lighting_preset],
        outputs=[edit_prompt],
    )

    gr.Markdown("---")
    with gr.Row(equal_height=True):
        gs_backend = gr.Dropdown(
            choices=["sharp", "da3"],
            value="sharp",
            label="GS backend",
            scale=2,
        )
        scale_mode = gr.Dropdown(
            choices=["da3_y_ground", "da3_2dgrid_global"],
            value="da3_y_ground",
            label="Scale mode",
            scale=2,
        )
        generate_btn = gr.Button(
            "Generate 3DGS", variant="primary", scale=1, min_width=160
        )

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
        outputs=[status, map_view, pano_view, pano_state],
    )

    upload_input.upload(
        fn=handle_upload,
        inputs=[upload_input],
        outputs=[status, map_view, pano_view, pano_state],
    )

    edit_btn.click(
        fn=handle_edit,
        inputs=[pano_state, edit_prompt, edit_mode, lighting_preset, edit_model_selector],
        outputs=[status, pano_view, pano_state],
        show_progress="minimal",
        show_progress_on=[pano_view],
    )

    generate_btn.click(
        fn=handle_generate,
        inputs=[pano_state, scale_mode, gs_backend],
        outputs=[status, splat_view],
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
