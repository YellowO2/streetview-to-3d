"""HTML/iframe builders for app.py's single-pano flow: map, panorama viewer,
splat viewer, point-cloud download link. `iframe()` is the canonical sandboxed
iframe wrapper -- street_builder/map_ui.py imports it from here too, rather
than keeping its own copy.
"""
import html as html_lib


def iframe(srcdoc: str, aspect: str = "16/9") -> str:
    escaped = html_lib.escape(srcdoc, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" sandbox="allow-scripts allow-same-origin" '
        f'style="width:100%;aspect-ratio:{aspect};border:none;border-radius:8px;background:#000">'
        "</iframe>"
    )


def file_url(abs_path: str) -> str:
    """Build Gradio's file-serving URL. /gradio_api/file= is the route in Gradio 5+."""
    return f"/gradio_api/file={abs_path}"


MAP_PLACEHOLDER = iframe(
    "<html><body style='margin:0;background:#1e1e2e;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Load a location to see it on the map</body></html>",
    aspect="16/9",
)
PANO_PLACEHOLDER = iframe(
    "<html><body style='margin:0;background:#111;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Panorama viewer</body></html>"
)
SPLAT_PLACEHOLDER = iframe(
    "<html><body style='margin:0;background:#111;color:#777;font:14px sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh'>"
    "Generate a 3DGS scene to view it here</body></html>"
)


def build_map(lat: float, lon: float) -> str:
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{margin:0;height:100%;width:100%}}</style>
</head><body><div id="map"></div>
<script>
var m = L.map('map').setView([{lat},{lon}], 18);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
    {{maxZoom:22,maxNativeZoom:19,attribution:'© OpenStreetMap'}}).addTo(m);
L.circleMarker([{lat},{lon}],{{radius:9,color:'crimson',fillColor:'crimson',fillOpacity:0.9,weight:2}}).addTo(m);
</script></body></html>"""
    return iframe(doc)


def build_pano_viewer(img_url: str) -> str:
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
    return iframe(doc)


def splat_viewer_with_download(ply_url: str) -> str:
    """Splat iframe + an inline download link below it. The link rides inside
    the same HTML payload as the viewer, so it survives backgrounded-tab
    WebSocket throttling that would otherwise drop separate component updates."""
    download_link = (
        f'<a href="{ply_url}" download '
        f'style="display:inline-block;margin-top:8px;padding:10px 16px;'
        f'background:#5b47d1;color:#fff;text-decoration:none;border-radius:8px;'
        f'font:600 14px sans-serif;">⬇ Download 3DGS (.ply)</a>'
    )
    return f'<div>{build_splat_iframe(ply_url)}{download_link}</div>'


def build_pointcloud_viewer(ply_url: str | None = None) -> str:
    """Live viewer for a raw DA3 point cloud (XYZ + per-vertex color), via
    three.js's PLYLoader + THREE.Points — distinct from build_splat_iframe,
    which expects 3D Gaussian Splat data (a different format) via SplatMesh
    and can't render a plain point cloud at all.

    Point size and camera distance are both derived from the loaded geometry's
    bounding sphere, since a point cloud's absolute scale isn't known ahead of
    time (unlike the fixed-radius pano sphere in build_pano_viewer). This is a
    rough heuristic, not tuned against a real render — may need adjusting.

    Also accepts drag-and-drop: dropping a local .ply file onto the viewer
    replaces whatever's currently shown, parsed client-side (PLYLoader.parse),
    no upload/round-trip to the server. ply_url is optional — with none given,
    the viewer renders empty and ready for a drop (used as the Street Builder
    tab's initial state, so you can preview an already-downloaded .ply without
    needing a GPU run first)."""
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{margin:0;background:#000;overflow:hidden;cursor:grab;font:14px sans-serif;color:#bbb}}body:active{{cursor:grabbing}}canvas{{display:block}}
#hint{{position:fixed;bottom:8px;right:8px;color:rgba(255,255,255,.4);font:11px sans-serif;pointer-events:none}}
#loading{{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;
  background:#000;transition:opacity .4s;pointer-events:none;padding:1em}}
#loading.gone{{opacity:0}}
.dot{{display:inline-block;animation:blink 1.4s infinite both}}
.dot:nth-child(2){{animation-delay:.2s}}.dot:nth-child(3){{animation-delay:.4s}}
@keyframes blink{{0%,80%,100%{{opacity:0}}40%{{opacity:1}}}}
#dropzone{{position:fixed;inset:0;display:none;align-items:center;justify-content:center;
  background:rgba(66,133,244,.15);border:3px dashed #4285f4;pointer-events:none;
  font-size:18px;color:#fff;z-index:10}}
#dropzone.active{{display:flex}}</style>
<script type="importmap">
{{"imports":{{
    "three":"https://unpkg.com/three@0.178.0/build/three.module.js",
    "three/addons/":"https://unpkg.com/three@0.178.0/examples/jsm/"
}}}}
</script></head><body>
<div id="loading">{"Loading point cloud<span class=\"dot\">.</span><span class=\"dot\">.</span><span class=\"dot\">.</span>" if ply_url else "Drop a .ply file here to preview it"}</div>
<div id="hint">drag to orbit · scroll to zoom · drop a .ply to preview it</div>
<div id="dropzone">Drop .ply to preview</div>
<script type="module">
import * as THREE from 'three';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.01, 10000);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setPixelRatio(devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const loader = new PLYLoader();
let currentPoints = null;

function showGeometry(geometry) {{
    // DA3's raw point cloud comes out in a Y-down/Z-forward (computer-vision)
    // convention; three.js is Y-up. Bake the flip into the geometry itself
    // (not a rotation on the Points mesh) so the bounding-sphere camera
    // framing below -- computed from this geometry -- stays correct. Same
    // root issue as splat.quaternion.set(1,0,0,0) in build_splat_iframe,
    // just applied before framing instead of after, since that viewer has no
    // dynamic bounding-sphere framing to keep in sync.
    geometry.rotateX(Math.PI);
    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    const hasColor = !!geometry.getAttribute('color');
    const material = new THREE.PointsMaterial({{
        size: Math.max(sphere.radius * 0.003, 0.01),
        vertexColors: hasColor,
        color: hasColor ? 0xffffff : 0x4285f4,
    }});

    if (currentPoints) {{
        scene.remove(currentPoints);
        currentPoints.geometry.dispose();
        currentPoints.material.dispose();
    }}
    currentPoints = new THREE.Points(geometry, material);
    scene.add(currentPoints);

    controls.target.copy(sphere.center);
    camera.position.copy(sphere.center).add(new THREE.Vector3(0, 0, sphere.radius * 2.2 || 5));
    camera.near = Math.max(sphere.radius * 0.01, 0.01);
    camera.far = sphere.radius * 20 || 10000;
    camera.updateProjectionMatrix();
    controls.update();

    document.getElementById('loading').classList.add('gone');
}}

{f'''loader.load('{ply_url}', showGeometry, undefined, err => {{
    console.error('PLY load failed', err);
    document.getElementById('loading').textContent = 'Failed to load point cloud';
}});''' if ply_url else ''}

const dropzone = document.getElementById('dropzone');
addEventListener('dragover', e => {{ e.preventDefault(); dropzone.classList.add('active'); }});
addEventListener('dragleave', e => {{ if (e.target === document.documentElement) dropzone.classList.remove('active'); }});
addEventListener('drop', e => {{
    e.preventDefault();
    dropzone.classList.remove('active');
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {{
        try {{
            showGeometry(loader.parse(reader.result));
        }} catch (err) {{
            console.error('Failed to parse dropped PLY', err);
        }}
    }};
    reader.readAsArrayBuffer(file);
}});

addEventListener('resize', () => {{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
}});
const tick = () => {{ controls.update(); renderer.render(scene, camera); }};
renderer.setAnimationLoop(tick);
document.addEventListener('visibilitychange', () => {{
    renderer.setAnimationLoop(document.hidden ? null : tick);
}});
</script></body></html>"""
    return iframe(doc)


def pointcloud_viewer_with_download(ply_url: str) -> str:
    """Live point-cloud viewer + an inline download link below it, same
    combined pattern as splat_viewer_with_download."""
    download_link = (
        f'<a href="{ply_url}" download '
        f'style="display:inline-block;margin-top:8px;padding:10px 16px;'
        f'background:#5b47d1;color:#fff;text-decoration:none;border-radius:8px;'
        f'font:600 14px sans-serif;">⬇ Download DA3 point cloud (.ply)</a>'
    )
    return f'<div>{build_pointcloud_viewer(ply_url)}{download_link}</div>'


def filter_sweep_links(levels: list[tuple[str, str | None]]) -> str:
    """Download-link list for the DA3 filter-threshold sweep debug button
    (street_builder/reconstruct.py's reconstruct_chain_filter_sweep): one
    .ply per (dist_thresh, angle_thresh) level, all from a single DA3
    inference call. No live viewer here -- download each and drag it into
    the point-cloud viewer above to compare visually. Per-level kept-view
    counts are only in the server log, not shown here."""
    items = []
    for label, path in levels:
        if path:
            items.append(
                f'<li style="margin:4px 0"><a href="{file_url(path)}" download '
                f'style="color:#8ab4f8">{html_lib.escape(label)}</a></li>'
            )
        else:
            items.append(
                f'<li style="margin:4px 0;color:#888">{html_lib.escape(label)} — no views survived</li>'
            )
    return (
        '<div style="padding:12px;background:#1e1e2e;border-radius:8px">'
        '<p style="color:#aaa;margin:0 0 8px;font:13px sans-serif">'
        "Filter sweep results — download each and drag it into the viewer above to compare:</p>"
        f'<ul style="margin:0;padding-left:20px;font:13px sans-serif;list-style:none">{"".join(items)}</ul>'
        "</div>"
    )


def build_splat_iframe(ply_url: str) -> str:
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
    return iframe(doc)
