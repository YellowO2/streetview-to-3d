"""
Gradio interface for Street Builder: reconstruct a walkable street corridor
into a 3D point cloud from Google Street View / Apple Look Around panoramas.

Run locally:  python app.py
HF Spaces:    set as app.py, add `spaces` to requirements, enable ZeroGPU.
"""

# Must be the first project import. ZeroGPU requires `spaces` (imported by
# services.pipeline_runner) to load before anything CUDA-related does.
# services.lookaround_fetch imports streetlevel's `reproject` module, which
# initializes CUDA at import time to pick its default device -- if that runs
# first, `import spaces` fails with "CUDA has been initialized before
# importing the `spaces` package." Importing pipeline_runner here, before
# the services import below, guarantees the required order regardless of
# what order the names in that later `from services import ...` get resolved in.
from services import pipeline_runner  # noqa: F401

import spaces
import gradio as gr
import pillow_heif

pillow_heif.register_heif_opener()

from paths import IMAGES_DIR, SPLATS_DIR
from street_builder.tab import build_tab as build_street_builder_tab
from street_builder.map_selection.tab import BRIDGE_HEAD_SCRIPT, BRIDGE_CSS


# ── Debug probe: isolate just our DA3Model caching pattern + run_da3 ───────
#
# Ported from potato-bug/da3-baseline-test (an external known-working DA3
# Space) after confirming there that its environment/depth_anything_3
# version differences made cross-repo comparisons unreliable -- the real
# app crashed identically inside that repo once its own code was copied
# in unmodified, so the crash isn't environment-specific. Keeping this
# probe HERE instead means every future isolation test runs in the actual
# target environment, no cross-repo ambiguity.
#
# This reproduces ONLY: panoramic_da3.DA3Model cached in a module-level
# global (rebuilt on first use, re-checked/re-attached to 'cuda' on every
# call after -- see get_da3() below) plus a real run_da3 call (view
# extraction, multi-view consensus, backprojection). Confirmed elsewhere
# to NOT crash twice in a row on its own -- kept here as a reusable
# building block for testing the next layer (e.g. two DIFFERENT task
# types through pipeline_runner's real _gpu_dispatch) directly in this
# app's own environment.
_probe_da3 = None


class _ProbeCfg:
    da3_model = "depth-anything/DA3NESTED-GIANT-LARGE"


def _get_probe_da3():
    global _probe_da3
    from panoramic_da3 import DA3Model
    if _probe_da3 is None:
        _probe_da3 = DA3Model(_ProbeCfg.da3_model)
    elif next(_probe_da3.model.parameters()).device.type != "cuda":
        _probe_da3.model = _probe_da3.model.to(device="cuda")
    return _probe_da3


@spaces.GPU(duration=120)
def our_pattern_probe(image_files):
    import tempfile

    import torch
    from panoramic_da3 import run_da3

    if not image_files or len(image_files) < 2:
        return "Upload at least 2 images first."
    try:
        da3 = _get_probe_da3()
        paths = [f.name if hasattr(f, "name") else f for f in image_files]
        with tempfile.TemporaryDirectory() as views_base:
            filtered_views, da3_result, pts, cols, per_pano_pts, per_pano_cols = run_da3(
                paths[0], paths[1:], _ProbeCfg(), views_base, da3=da3,
            )
        n_pts = 0 if pts is None else len(pts)
        return f"OK -- {len(filtered_views)} view(s) survived, {n_pts} point(s) backprojected"
    except Exception as e:
        import traceback
        return f"FAILED: {e}\n\n{traceback.format_exc()}"
    finally:
        torch.cuda.empty_cache()


@spaces.GPU(duration=120)
def many_calls_probe(image_files, n_calls):
    """Simulates the call VOLUME a real corridor search makes in one GPU
    lease -- "2. Run auto-path" internally calls run_da3 (via
    services/da3_ops.py's test_edge/rate_pano) once per candidate edge
    tested, easily dozens of times, all within ONE @spaces.GPU call,
    before "3. Join segments" starts a SEPARATE lease that's the one
    observed to crash. our_pattern_probe above only ever makes ONE
    run_da3 call per lease and does NOT crash on a second lease -- this
    tests whether it's specifically the repetition/volume in the FIRST
    lease that leaves something broken for the next one, not just
    "calling GPU twice" in general. Click this first, then click
    "Run our pattern probe" above as the second lease."""
    import tempfile
    import time

    import torch
    from panoramic_da3 import run_da3

    if not image_files or len(image_files) < 2:
        return "Upload at least 2 images first."
    n_ok = 0
    try:
        da3 = _get_probe_da3()
        paths = [f.name if hasattr(f, "name") else f for f in image_files]
        t0 = time.monotonic()
        for i in range(int(n_calls)):
            with tempfile.TemporaryDirectory() as views_base:
                run_da3(paths[0], paths[1:], _ProbeCfg(), views_base, da3=da3)
            n_ok += 1
        elapsed = time.monotonic() - t0
        return f"OK -- {n_ok}/{int(n_calls)} run_da3 call(s) completed in {elapsed:.1f}s"
    except Exception as e:
        import traceback
        return f"FAILED after {n_ok} call(s): {e}\n\n{traceback.format_exc()}"
    finally:
        torch.cuda.empty_cache()


def real_pathfind_probe(image_files):
    """Test 3: the REAL pipeline_runner.run_pathfind_reconstruction_gpu
    (-> real _gpu_dispatch -> real services/da3_ops.py -> real
    street_builder/reconstruction/walk_graph.py) called directly with a
    fabricated 4-dot corridor -- two structurally-disconnected 2-dot
    pairs, close enough together (a few meters) that join_segments'
    real bridging search actually attempts real DA3 pairwise tests
    between them, instead of skipping for being too far apart. No
    Gradio UI/state involved, no network downloads -- same real backend
    code the "Street Builder" tab's Run button calls, minus everything
    upstream of it (map picking, candidate downloading).

    Returns segments from the real corridor search -- pass this
    straight into real_join_probe below as the SECOND, separate GPU
    lease, exactly mirroring "2. Run auto-path" then "3. Join
    segments"."""
    if not image_files or len(image_files) < 2:
        return "Upload at least 2 images first.", None
    path_a, path_b = image_files[0].name if hasattr(image_files[0], "name") else image_files[0], \
        image_files[1].name if hasattr(image_files[1], "name") else image_files[1]

    # ~0.00009 deg lat ~= 10m. Pair (0,1) and pair (2,3) are each
    # structurally adjacent to each other but NOT to the other pair --
    # forces 2 disconnected pieces out of phase 1 -- while staying well
    # within join_segments.BRIDGE_MAX_DIST_M (30m) of each other so the
    # bridging search has a real nearby candidate to test.
    points = [(0.0, 0.0), (0.00009, 0.0), (0.00004, 0.00004), (0.00013, 0.00004)]
    adjacency = {0: [1], 1: [0], 2: [3], 3: [2]}
    date_graphs = [{
        "date": "probe-date",
        "dot_candidates": {
            0: [("probe:0", path_a, points[0][0], points[0][1])],
            1: [("probe:1", path_b, points[1][0], points[1][1])],
            2: [("probe:2", path_a, points[2][0], points[2][1])],
            3: [("probe:3", path_b, points[3][0], points[3][1])],
        },
    }]
    try:
        segments = pipeline_runner.run_pathfind_reconstruction_gpu(
            date_graphs, points, adjacency, points[0][0], points[0][1],
        )
        return f"OK -- {len(segments)} segment(s) produced. Now click Test 3b below.", segments
    except Exception as e:
        import traceback
        return f"FAILED: {e}\n\n{traceback.format_exc()}", None


def real_join_probe(segments):
    """Test 3b: the REAL pipeline_runner.join_segments_gpu, fed the
    segments Test 3 above just produced -- the actual second, separate
    GPU lease. This is as close as this probe tab gets to the exact
    real crash sequence without going through the Gradio map-picking UI."""
    if not segments:
        return "Run Test 3 first."
    if len(segments) < 2:
        return f"Only {len(segments)} segment(s) -- nothing to join. Re-run Test 3."
    try:
        pieces = pipeline_runner.join_segments_gpu(segments)
        return f"OK -- {len(pieces) if pieces else 0} joined piece(s) produced"
    except Exception as e:
        import traceback
        return f"FAILED: {e}\n\n{traceback.format_exc()}"


def build_probe_tab() -> gr.Blocks:
    with gr.Blocks() as probe:
        gr.Markdown(
            "Isolated probes: panoramic_da3.DA3Model caching pattern + real "
            "run_da3 calls, nothing else from this app (no street_builder "
            "logic, no multi-task dispatch)."
        )
        files = gr.File(file_count="multiple", label="Images")

        gr.Markdown("**Test 1** -- single call, twice in a row (already confirmed OK):")
        btn = gr.Button("Run our pattern probe")
        out = gr.Textbox(label="Result", lines=8)
        btn.click(fn=our_pattern_probe, inputs=[files], outputs=[out])

        gr.Markdown(
            "**Test 2** -- simulate corridor-search call volume in ONE lease, "
            "then click Test 1's button as a second, separate lease:"
        )
        n_calls = gr.Number(value=15, label="Number of run_da3 calls to make", precision=0)
        many_btn = gr.Button("Simulate corridor search (many calls, one lease)")
        many_out = gr.Textbox(label="Result", lines=8)
        many_btn.click(fn=many_calls_probe, inputs=[files, n_calls], outputs=[many_out])

        gr.Markdown(
            "**Test 3** -- the REAL pathfind + join code (pipeline_runner.py, "
            "services/da3_ops.py, street_builder/reconstruction/*), fed a "
            "fabricated 4-dot corridor instead of real map data. Click 3a, "
            "then 3b -- the actual real-code equivalent of \"2. Run auto-path\" "
            "then \"3. Join segments\" as two separate GPU leases:"
        )
        segments_state = gr.State(None)
        real_run_btn = gr.Button("3a. Real run_pathfind_reconstruction_gpu")
        real_run_out = gr.Textbox(label="Result", lines=8)
        real_run_btn.click(fn=real_pathfind_probe, inputs=[files], outputs=[real_run_out, segments_state])

        real_join_btn = gr.Button("3b. Real join_segments_gpu (separate lease)")
        real_join_out = gr.Textbox(label="Result", lines=8)
        real_join_btn.click(fn=real_join_probe, inputs=[segments_state], outputs=[real_join_out])
    return probe


def build_street_builder_demo_tab() -> gr.Blocks:
    with gr.Blocks() as sb_demo:
        build_street_builder_tab()
    return sb_demo


sb_demo = build_street_builder_demo_tab()
probe_demo = build_probe_tab()
demo = gr.TabbedInterface([sb_demo, probe_demo], ["Street Builder", "Our Pattern Probe"])


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[IMAGES_DIR, SPLATS_DIR],
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Default(),
        css=".no-pad { padding-left: 0 !important; padding-right: 0 !important; } " + BRIDGE_CSS,
        head=BRIDGE_HEAD_SCRIPT,
        # Explicitly off: the startup log showed "with SSR (Node proxy ->
        # Python :7861)" -- an extra Node.js hop HF Spaces enables by
        # default -- right before the Space got stuck permanently on
        # "restarting" despite the Python server itself logging a
        # successful start. Forcing plain client-side rendering removes
        # that layer as a suspect.
        ssr_mode=False,
    )
