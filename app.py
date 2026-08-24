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
from street_builder.map_selection.tab import build_tab as build_street_builder_tab, BRIDGE_HEAD_SCRIPT, BRIDGE_CSS


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


def build_probe_tab() -> gr.Blocks:
    with gr.Blocks() as probe:
        gr.Markdown(
            "Isolated probe: panoramic_da3.DA3Model caching pattern + a real "
            "run_da3 call, nothing else from this app (no street_builder "
            "logic, no multi-task dispatch). Upload 2+ images and click the "
            "button **twice in a row** to test."
        )
        files = gr.File(file_count="multiple", label="Images")
        btn = gr.Button("Run our pattern probe")
        out = gr.Textbox(label="Result", lines=8)
        btn.click(fn=our_pattern_probe, inputs=[files], outputs=[out])
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
