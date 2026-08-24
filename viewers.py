"""HTML/iframe builders for street_builder's point-cloud viewer and result
download links. `iframe()` is the canonical sandboxed iframe wrapper --
street_builder/map_ui.py imports it from here too, rather than keeping its
own copy.
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


def build_pointcloud_viewer(ply_url: str | None = None) -> str:
    """Live viewer for a raw DA3 point cloud (XYZ + per-vertex color), via
    three.js's PLYLoader + THREE.Points.

    Point size and camera distance are both derived from the loaded geometry's
    bounding sphere, since a point cloud's absolute scale isn't known ahead of
    time. This is a rough heuristic, not tuned against a real render — may
    need adjusting.

    Also accepts drag-and-drop: dropping a local .ply file onto the viewer
    replaces whatever's currently shown, parsed client-side (PLYLoader.parse),
    no upload/round-trip to the server. ply_url is optional — with none given,
    the viewer renders empty and ready for a drop (used as the Street Builder
    tab's initial state, so you can preview an already-downloaded .ply without
    needing a GPU run first)."""
    loading_msg = (
        '''Loading point cloud<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>'''
        if ply_url else "Drop a .ply file here to preview it"
    )
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
<div id="loading">{loading_msg}</div>
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
    // framing below -- computed from this geometry -- stays correct.
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


def labeled_download_links(items_in: list[tuple[str, str | None]]) -> str:
    """Download-link list for a set of labeled .ply results (multiple
    segments from greedy/pathfind reconstruction, etc). No live viewer
    here -- download each and drag it into the point-cloud viewer above
    to compare visually."""
    items = []
    for label, path in items_in:
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
        "Results — download each and drag it into the viewer above to compare:</p>"
        f'<ul style="margin:0;padding-left:20px;font:13px sans-serif;list-style:none">{"".join(items)}</ul>'
        "</div>"
    )
