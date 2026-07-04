"""
Experimental / work-in-progress code removed from app.py.

To restore a feature, copy the relevant handler back into app.py and
add the corresponding tab + event binding inside the `with gr.Blocks()` block.
"""

# ── Upload Panorama tab ────────────────────────────────────────────────────────
#
# Handler — paste into app.py alongside the other handle_* functions:
#
# def handle_upload(file_path):
#     """User uploaded their own equirectangular panorama."""
#     if not file_path:
#         return "No file uploaded.", _MAP_PLACEHOLDER, _PANO_PLACEHOLDER, None
#     abs_path = os.path.abspath(file_path)
#     state = {
#         "source": "upload",
#         "image_path": abs_path,
#         "original_image_path": abs_path,
#         "neighbors": [],
#     }
#     return (
#         f"Uploaded panorama loaded. Click Generate 3DGS to continue.",
#         _MAP_PLACEHOLDER,  # no map for uploads
#         _build_pano_viewer(_file_url(abs_path)),
#         state,
#     )
#
# Tab UI — paste inside `with gr.Tabs():` after the Street View tab:
#
#     with gr.Tab("Upload Panorama"):
#         upload_input = gr.File(
#             label="Upload an equirectangular panorama (.jpg / .png). This is likely to return suboptimal results due to certain optimisations not made.",
#             file_types=["image"],
#             type="filepath",
#         )
#
# Event binding — paste after the load_btn.click binding:
#
#     upload_input.upload(
#         fn=handle_upload,
#         inputs=[upload_input],
#         outputs=[status, map_view, pano_view, pano_state],
#     )


# ── Video Test tab ─────────────────────────────────────────────────────────────
#
# GPU pipeline helpers — paste into app.py before the `# ── GPU pipeline ──` section:

import os
import math
import uuid

def _save_gaussian_ply_pruned(gaussians, save_path, ctx_depth, opacity_threshold=0.05):
    """save_gaussian_ply + opacity pruning added to the spatial mask."""
    import torch
    from depth_anything_3.utils.gsply_helpers import export_ply, inverse_sigmoid
    from einops import rearrange

    src_v, out_h, out_w, _ = ctx_depth.shape

    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = inverse_sigmoid(gaussians.opacities)

    mask = torch.zeros_like(ctx_depth, dtype=torch.bool)
    gstrim_h = int(8 / 256 * out_h)
    gstrim_w = int(8 / 256 * out_w)
    mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w, :] = 1
    mask = mask.squeeze(-1)

    logit_thresh = math.log(opacity_threshold / (1.0 - opacity_threshold))
    op_grid = rearrange(gs_opacities[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w)
    if op_grid.dim() == 4:
        op_vals = op_grid[..., 0]
    else:
        op_vals = op_grid
    op_mask = op_vals > logit_thresh

    final_mask = mask & op_mask
    kept = final_mask.sum().item()
    total = mask.sum().item()
    print(f"  Opacity pruning: {kept:,} / {total:,} kept ({100*kept/total:.1f}%)")

    def trim_op(element):
        return rearrange(element[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w)[final_mask]

    export_ply(
        means=trim_op(world_means),
        scales=trim_op(gs_scales),
        rotations=trim_op(world_rotations),
        harmonics=trim_op(world_shs),
        opacities=trim_op(gs_opacities),
        path=__import__('pathlib').Path(save_path),
        save_sh_dc_only=True,
    )


# Decorate with @GPU when moving back into app.py
def _run_video_pipeline_gpu(frame_paths, output_dir, lyra=False):
    import torch
    from components.DepthMapGenerator.DA3Model import DA3Model
    from config import load_pipeline_config

    os.makedirs(output_dir, exist_ok=True)
    cfg = load_pipeline_config()

    da3 = DA3Model(cfg.da3_model)
    model = da3.model

    if lyra:
        from huggingface_hub import hf_hub_download
        print("Downloading Lyra 2 DA3 checkpoint (cached after first run)...")
        lyra_ckpt = hf_hub_download(repo_id="nvidia/Lyra-2.0", filename="checkpoints/recon/model.pt")
        ckpt = torch.load(lyra_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("module") or ckpt.get("model") or ckpt
        if isinstance(state_dict, dict) and any(k.startswith("model.") for k in state_dict):
            converted = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        else:
            converted = state_dict
        missing, unexpected = model.model.load_state_dict(converted, strict=False)
        print(f"  Lyra weights: {len(converted)} params, {len(missing)} missing, {len(unexpected)} unexpected")
        model.eval()

    prediction = model.inference(frame_paths, infer_gs=True, export_format="mini_npz")

    final_path = os.path.join(output_dir, "final_output.ply")
    ctx_depth = torch.from_numpy(prediction.depth).unsqueeze(-1).to(prediction.gaussians.means)
    _save_gaussian_ply_pruned(prediction.gaussians, final_path, ctx_depth, opacity_threshold=0.05)

    return final_path if os.path.exists(final_path) else None


import gradio as gr

def handle_video_generate(video_path, sample_fps, video_model, progress=gr.Progress(track_tqdm=True)):
    import cv2

    IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    SPLATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splats")

    if not video_path:
        yield ("Upload a video first.", None)
        return

    yield ("Extracting frames...", None)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / sample_fps))

    frames_dir = os.path.join(IMAGES_DIR, f"video_{uuid.uuid4().hex}")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        p = os.path.join(frames_dir, f"frame_{len(frame_paths):04d}.jpg")
        cv2.imwrite(p, frame)
        frame_paths.append(p)
        idx += frame_interval
    cap.release()

    if not frame_paths:
        yield ("Could not extract frames from video.", None)
        return

    use_lyra = video_model == "DA3 (Lyra 2 finetuned)"
    model_label = "Lyra 2 finetuned DA3" if use_lyra else "DA3"
    yield (f"Running {model_label} on {len(frame_paths)} frames (this takes ~2 min)...", None)

    output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
    t_start = __import__('time').time()
    try:
        ply_path = _run_video_pipeline_gpu(frame_paths, output_dir, lyra=use_lyra)
    except Exception as e:
        yield (f"Pipeline failed: {e}", None)
        return

    if not ply_path or not os.path.exists(ply_path):
        yield ("Pipeline finished but no PLY produced.", None)
        return

    elapsed = __import__('time').time() - t_start
    progress(1.0, desc="Done!")
    yield (
        f"3DGS ready ({model_label}, {len(frame_paths)} frames, {elapsed:.0f}s). Loading viewer...",
        None,  # replace None with _splat_viewer_with_download(_file_url(ply_path)) in app.py
    )

#
# Tab UI — paste inside `with gr.Tabs():`:
#
#     with gr.Tab("Video"):
#         gr.Markdown("Upload a phone video. Frames are extracted and fed directly to DA3's GS mode.")
#         video_input = gr.File(label="Upload video (.mp4, .mov, ...)", file_types=["video"], type="filepath")
#         with gr.Row(equal_height=True):
#             n_frames_slider = gr.Slider(minimum=0.1, maximum=10, value=1, step=0.1, label="Sampling FPS", scale=3)
#             video_model_selector = gr.Dropdown(
#                 choices=["DA3 (standard)", "DA3 (Lyra 2 finetuned)"],
#                 value="DA3 (standard)",
#                 label="Model",
#                 scale=2,
#             )
#         video_generate_btn = gr.Button("Generate 3DGS from Video", variant="primary")
#
# Event binding — paste before `if __name__ == "__main__":`:
#
#     video_generate_btn.click(
#         fn=handle_video_generate,
#         inputs=[video_input, n_frames_slider, video_model_selector],
#         outputs=[status, splat_view],
#         show_progress="minimal",
#         show_progress_on=[splat_view],
#     )
