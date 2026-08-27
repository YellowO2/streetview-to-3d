"""Gradio wiring for the street-builder pathfind/reconstruction flow: given
an already-picked graph of nodes (see street_builder/map_selection/tab.py's
build_map_section), prepare candidates, run the corridor search, and join
segments into a final point cloud.

Composes the whole tab: mounts map_selection's own map-picking section,
then this module's own prepare/run/join controls underneath it, wired
against the same shared `state` map_selection's handlers already update.
"""
import gzip
import json
import os
import uuid

import gradio as gr
import numpy as np
from huggingface_hub import HfApi

import viewers
from paths import SPLATS_DIR
from street_builder import main as street_main
from street_builder.map_selection.tab import build_map_section, nodes_by_key

# Where the scripted CLI flow pushes its running state and final results --
# Hub-native storage (fast up/download,
# survives Space restarts/redeploys) instead of routing large files through
# Gradio's own file-serving proxy, which is slow for anything this size.
# Needs an HF_TOKEN secret configured on the Space itself (Settings ->
# Variables and secrets) with write access -- a Space has no Hub write
# access by default. HfApi() picks that env var up automatically.
CLI_JOIN_DATASET_REPO = "potato-bug/ntu-reconstruction"


def handle_pathfind_prepare(state, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 1 of 3: gathers every Google + Apple pano
    near the clicked graph's real shape -- branches and loops included,
    since the selection graph (state["selected"] + state["selected_edges"])
    is only ever built from real Street View edges (see
    map_selection/tab.py's handle_bridge_message), not guessed from click
    order -- and downloads the top-date candidate batch. No GPU here; see
    handle_pathfind_run for why this is its own separate step instead of
    one combined button. See street_main.prepare_pathfind."""
    selected = state.get("selected", [])
    selected_edges = state.get("selected_edges", [])
    if len(selected) < 2 or not selected_edges:
        raise gr.Error("Select at least 2 connected nodes tracing the route (start to a goal).")

    by_key = nodes_by_key(state)
    start_node = by_key.get(selected[0])
    if not start_node:
        raise gr.Error("Start node not found.")
    start = (start_node["lat"], start_node["lon"])
    goals = [(by_key[k]["lat"], by_key[k]["lon"]) for k in selected[1:] if k in by_key]
    corridor_edges = [
        ((by_key[a]["lat"], by_key[a]["lon"]), (by_key[b]["lat"], by_key[b]["lon"]))
        for a, b in selected_edges if a in by_key and b in by_key
    ]

    progress(0, desc="Gathering + downloading candidates...")
    try:
        prep = street_main.prepare_pathfind(start, goals, corridor_edges)
    except Exception as e:
        raise gr.Error(f"Prepare failed: {e}")

    progress(1.0, desc="Done!")
    n = len(prep["node_entries"])
    return prep, f"<p>Prepared {n} candidate(s) across {len(prep['top_dates'])} date(s). Ready — press \"Run Auto-path\".</p>"


def handle_pathfind_run(prep, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 2 of 3: runs the real multi-goal best-first
    search over whatever handle_pathfind_prepare already downloaded -- the
    fixed start node stays the search's start; every other selected node is
    a goal, and the search doesn't stop at the first one reached, it keeps
    growing toward whatever's still outstanding.

    Split from the prepare step specifically so this GPU-triggering click
    is its own fresh, minimal-latency interaction -- the ZeroGPU proxy
    token's validity is wall-clock, and a long download sitting ahead of
    the @spaces.GPU call (as one combined button used to do) is exactly
    what can let it go stale before the schedule request is ever sent.

    Only saves each segment's own preview here -- joining them (step 3,
    handle_pathfind_join) is a separate button on purpose: it's its own
    separate GPU call, so keeping it out of this call means re-testing/
    tuning join/bridging doesn't require re-running the expensive
    corridor search each time.
    See street_main.run_prepared_pathfind_segments/save_pathfind_segments."""
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    try:
        segments = street_main.run_prepared_pathfind_segments(prep)
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results = street_main.save_pathfind_segments(segments, output_dir)
        bundle_path = street_main.save_segments_bundle(segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Auto-path failed: {e}")

    note = "" if len(segments) > 1 else "<p>Single segment -- nothing to join.</p>"
    return viewers.labeled_download_links(results) + note, segments, bundle_path


def handle_pathfind_run_and_join(prep, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, combined 2+3: corridor search + join/bridging
    in ONE GPU session (see street_main.run_prepared_pathfind), instead
    of the separate Run then Join buttons -- avoids paying for two
    separate DA3 model loads when you just want the final result end-
    to-end and don't need to re-test join/bridging separately afterward.
    Still saves a segments bundle (same as handle_pathfind_run), so Join
    can be re-run alone later against this same result if needed."""
    if not prep:
        raise gr.Error("Nothing prepared yet -- press \"Prepare\" first.")

    try:
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results, segments, bundle_path = street_main.run_prepared_pathfind(prep, output_dir)
    except Exception as e:
        raise gr.Error(f"Run + Join failed: {e}")

    return viewers.labeled_download_links(results), segments, bundle_path


def handle_pathfind_load_segments(file_path):
    """Loads a previously downloaded segments bundle (see the "Download
    segments" file handle_pathfind_run produces), so Join can run
    immediately without re-running Prepare or the expensive GPU search --
    a different session, or after tweaking join_segments.py. See
    street_main.load_segments_bundle."""
    if not file_path:
        raise gr.Error("Choose a segments file first.")
    try:
        segments = street_main.load_segments_bundle(file_path)
    except Exception as e:
        raise gr.Error(f"Load failed: {e}")
    # Join no longer needs anything from prep (frame_poses in segments
    # already carries each node's lat/lon) -- this placeholder just keeps
    # the "something's ready" gate other handlers check (e.g.
    # handle_pathfind_join's `if not prep or not segments`) true.
    return True, segments, f"<p>Loaded {len(segments)} segment(s) from file. Ready — press \"Join segments\".</p>"


def handle_pathfind_join(prep, segments, progress=gr.Progress(track_tqdm=True)):
    """Experimental button, step 3 of 3: bridges segments together with
    real DA3 tests (see join_segments.join_segments) -- no GPS placement;
    a segment pair known/expected to be adjacent with zero real
    candidates in range is treated as an upstream bug and raises, rather
    than silently falling back to GPS. Its own separate GPU call from
    Run's -- safe to press again after tweaking the join/bridging logic
    without re-running Run. See street_main.save_joined_pathfind."""
    if not prep or not segments:
        raise gr.Error("Nothing to join yet -- press \"Run Auto-path\" first.")
    if len(segments) < 2:
        raise gr.Error("Only one segment -- nothing to join.")

    try:
        output_dir = os.path.join(SPLATS_DIR, uuid.uuid4().hex)
        results = street_main.save_joined_pathfind(segments, output_dir)
    except Exception as e:
        raise gr.Error(f"Join failed: {e}")

    return viewers.labeled_download_links(results)


CLI_CHECKPOINT_PREFIX = "cli_join/current"
# Each chunk's own raw, un-bridged Run output, saved independently of
# whatever bridging does with it -- see _upload_pieces/_download_pieces.
# Without this, a bug in the BRIDGE step (we've hit two) has no fallback:
# the raw output only ever lived in that one Gradio session's gr.State,
# so fixing the bug meant re-running the expensive GPU generation too,
# not just re-running the fixed bridge step against the same data.
CLI_RAW_PREFIX = "cli_raw"

# int16-xyz quantization step for CLI storage -- see _quantize_ply. 1cm:
# finer than Saver's own 3cm voxel downsampling grid, so no perceptible
# extra loss on top of what's already there.
QUANT_SCALE_M = 0.01


def _quantize_ply(path):
    """Rewrites a plain float32-xyz PLY (Saver._write_ply's own output)
    IN PLACE as int16-xyz quantized at QUANT_SCALE_M -- halves position
    storage (2 bytes/axis vs 4) for no perceptible loss: each piece's
    own DA3 frame is centered near (0,0,0) (the first confirmed node
    anchors it there, see walk_graph.test_and_confirm), so 1cm
    resolution comfortably fits int16's +-327m range for any single
    piece. Leaves the file untouched (plain float32) if any point would
    overflow that range, rather than silently corrupting data.

    Deliberately isolated here rather than in Saver or output_to_piece:
    this only ever runs on the CLI flow's own intermediate/checkpoint
    storage (_upload_pieces), never on the interactive UI's regular
    preview/download output, which a person might open directly in an
    external viewer that doesn't expect non-float vertex coordinates.
    _dequantize_ply is the exact inverse, so output_to_piece itself
    never has to know this format exists at all."""
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    lines = header.splitlines()
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    has_color = "red" in header
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.frombuffer(data[header_end:], dtype=np.dtype(fields), count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)

    if n > 0 and np.abs(pts).max() > 32767 * QUANT_SCALE_M:
        return  # out of int16 range at this scale -- leave as plain float32

    q = np.rint(pts / QUANT_SCALE_M).astype(np.int16)
    out_fields = [("x", "<i2"), ("y", "<i2"), ("z", "<i2")]
    if has_color:
        out_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    structured = np.empty(n, dtype=np.dtype(out_fields))
    structured["x"], structured["y"], structured["z"] = q[:, 0], q[:, 1], q[:, 2]
    if has_color:
        structured["red"] = verts["red"]
        structured["green"] = verts["green"]
        structured["blue"] = verts["blue"]

    new_header = [
        "ply", "format binary_little_endian 1.0", f"comment quant_scale_m {QUANT_SCALE_M}",
        f"element vertex {n}", "property short x", "property short y", "property short z",
    ]
    if has_color:
        new_header += ["property uchar red", "property uchar green", "property uchar blue"]
    new_header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(new_header) + "\n").encode("ascii"))
        f.write(structured.tobytes())


def _dequantize_ply(path):
    """Inverse of _quantize_ply, in place -- a no-op if the file was
    never quantized (the out-of-range fallback, or an older upload that
    predates this format). Always leaves a plain float32-xyz PLY behind,
    so output_to_piece only ever has to understand the one format."""
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    lines = header.splitlines()
    if not any(l.startswith("property short x") for l in lines):
        return
    scale = float(next(l for l in lines if l.startswith("comment quant_scale_m")).split()[-1])
    n = int(next(l for l in lines if l.startswith("element vertex")).split()[-1])
    has_color = "red" in header
    fields = [("x", "<i2"), ("y", "<i2"), ("z", "<i2")]
    if has_color:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.frombuffer(data[header_end:], dtype=np.dtype(fields), count=n)
    pts = (np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)) * scale

    out_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        out_fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    structured = np.empty(n, dtype=np.dtype(out_fields))
    structured["x"], structured["y"], structured["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if has_color:
        structured["red"] = verts["red"]
        structured["green"] = verts["green"]
        structured["blue"] = verts["blue"]

    new_header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
                  "property float x", "property float y", "property float z"]
    if has_color:
        new_header += ["property uchar red", "property uchar green", "property uchar blue"]
    new_header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(new_header) + "\n").encode("ascii"))
        f.write(structured.tobytes())


def _download_pieces(prefix):
    """Downloads whatever's saved under `prefix` in the dataset repo (see
    _upload_pieces) and reconstructs bridgeable pieces from it -- ([],
    []) if nothing's there yet. Used for both the running checkpoint
    (CLI_CHECKPOINT_PREFIX) and each chunk's own raw output
    (CLI_RAW_PREFIX/<chunk_id>) -- same file shape (.ply.gz + .json per
    piece) either way, just a different path. Exactly what output_to_piece
    reconstructs from (see its own docstring for why that's enough to
    keep bridging further) -- the .gz is transparently decompressed and
    dequantized right here, so nothing downstream of this function ever
    deals with either (see _upload_pieces' own docstring for why the
    file's compressed/quantized on disk in the first place)."""
    from huggingface_hub import hf_hub_download
    from street_builder.reconstruction.join_segments import output_to_piece

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset")
    except Exception:
        files = []
    ply_files = sorted(f for f in files if f.startswith(prefix + "/") and f.endswith(".ply.gz"))

    pieces, id_sets = [], []
    for ply_rel in ply_files:
        # _save_joined_pieces names these "pathfind_joined{suffix}.ply" /
        # "pathfind_metadata{suffix}.json" -- different prefixes, not
        # just a different extension, so swap the whole basename prefix
        # rather than assuming they share one.
        dirname, basename = os.path.split(ply_rel)
        if not basename.startswith("pathfind_joined") or not basename.endswith(".ply.gz"):
            continue
        suffix = basename[len("pathfind_joined"):-len(".ply.gz")]
        json_rel = os.path.join(dirname, f"pathfind_metadata{suffix}.json")
        if json_rel not in files:
            continue
        ply_gz_path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=ply_rel)
        json_path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=json_rel)
        ply_path = ply_gz_path[:-len(".gz")]
        with gzip.open(ply_gz_path, "rb") as f_in, open(ply_path, "wb") as f_out:
            f_out.writelines(f_in)
        _dequantize_ply(ply_path)
        with open(json_path) as f:
            metadata = json.load(f)
        piece, chunk_ids = output_to_piece(ply_path, metadata)
        pieces.append(piece)
        id_sets.append(chunk_ids or [])
    return pieces, id_sets


def _upload_pieces(prefix, pieces, id_sets, commit_message):
    """Saves pieces (converted to final output shape via pieces_to_output)
    under `prefix` in the dataset repo, replacing whatever was there
    before -- ONE commit (a folder-delete op + one add op per new file),
    not one commit per file. HF rate-limits dataset commits to 256/hour;
    the original per-file delete_file/upload_file loop could burn
    through dozens of commits for a single update (a 17-piece checkpoint
    = 34 files = 34+ separate commits) and hit that limit after only a
    handful of chunks -- confirmed the hard way. Returns the dataset
    URLs.

    Each .ply gets quantized (see _quantize_ply) then gzip-compressed
    before upload (renamed to .ply.gz); _download_pieces reverses both,
    transparently, so nothing else in the pipeline ever deals with
    either. The writer (panoramic_da3.Saver) stores raw binary float32
    xyz + uint8 rgb with zero compression of its own -- a real chunk's
    own raw output routinely runs 50-100+MB, and this is purely a
    storage/transport concern, not a point-cloud format one, so it's
    handled here at the upload boundary instead of in the general-
    purpose writer (which the interactive UI's own preview/download
    output also goes through, unquantized/uncompressed, for a person
    who might open it directly in an external viewer)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from street_builder.reconstruction.join_segments import pieces_to_output

    results = pieces_to_output(pieces, id_sets=id_sets)
    run_id = uuid.uuid4().hex
    output_dir = os.path.join(SPLATS_DIR, run_id)
    saved = street_main._save_joined_pieces(results, output_dir)

    for fname in os.listdir(output_dir):
        if fname.endswith(".ply"):
            path = os.path.join(output_dir, fname)
            _quantize_ply(path)
            with open(path, "rb") as f_in, gzip.open(path + ".gz", "wb") as f_out:
                f_out.writelines(f_in)
            os.remove(path)

    fnames = sorted(os.listdir(output_dir))
    api = HfApi()
    has_existing = any(f.startswith(prefix + "/") for f in api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset"))
    operations = [CommitOperationDelete(path_in_repo=prefix + "/")] if has_existing else []
    operations += [
        CommitOperationAdd(path_in_repo=f"{prefix}/{fname}", path_or_fileobj=os.path.join(output_dir, fname))
        for fname in fnames
    ]
    api.create_commit(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", operations=operations, commit_message=commit_message)
    return [f"https://huggingface.co/datasets/{CLI_JOIN_DATASET_REPO}/blob/main/{prefix}/{fname}" for fname in fnames]


def handle_cli_run_chunk(payload_str, progress=gr.Progress(track_tqdm=True)):
    """Scripted/CLI: Prepare+Run for ONE chunk -- its own single
    @spaces.GPU call, kept SEPARATE from handle_cli_bridge_chunk's own
    GPU call on purpose. An earlier version combined prepare+run+bridge
    into one handler and hit 'Expired ZeroGPU proxy token' on the very
    first chunk -- exactly the documented failure mode this whole app
    otherwise avoids everywhere else (see street_builder/main.py's own
    module docstring): a second @spaces.GPU call inside the same request
    can fire after the first one's proxy token has already gone stale.
    Two separate client calls (two separate button clicks/gradio_client
    predict()s) each get their own fresh token; one handler making two
    GPU calls back-to-back does not.

    payload_str: JSON. When use_global_cover (default True):
    {"chunk_id": ..., "dots": [dot_index, ...], "date": str,
    "protected_positions": [[lat, lon], ...], "use_global_cover": true} --
    dots/date come straight from
    global_dates.split_cover_into_chunks (this chunk was already cut
    FROM the pre-computed whole-NTU date cover, single-date by
    construction -- see street_main.prepare_pathfind_from_cover_chunk).
    When use_global_cover is False (fallback for an area outside
    whatever tests/fetch_ntu_metadata.py already covered):
    {"chunk_id": ..., "start": [lat, lon], "goals": [[lat, lon], ...],
    "edges": [[[lat1, lon1], [lat2, lon2]], ...], "protected_positions":
    [[lat, lon], ...], "use_global_cover": false} -- this chunk
    independently ranks its own top-N dates (street_main.prepare_pathfind).

    protected_positions (optional): this chunk's own real boundary node
    COORDINATES (known from the chunking step) -- kept even if set_cover
    would otherwise drop them as geographically redundant, since that
    location is what cross-chunk bridging needs later. Matched by real
    distance, not node key -- a node key is date-specific (the same real
    spot gets a different pano id per historical date), so this can't be
    a key. See walk_graph.run_pathfind_reconstruction's own docstring.

    Saves this chunk's own raw segments to CLI_RAW_PREFIX/<chunk_id>
    BEFORE returning -- independent of whatever bridging does with them
    later. Without this, a bug in the bridge step (we've hit two) has no
    fallback: the raw output only lives in this call's return value, so
    fixing a bridge bug would mean re-running this expensive GPU step
    again too, not just re-running the fixed bridge against the same
    data. handle_cli_bridge_chunk reads chunk data back from here by
    chunk_id -- it does NOT need this call's return value at all
    (returned mainly as an immediate status/preview)."""
    try:
        payload = json.loads(payload_str)
        chunk_id = payload["chunk_id"]
        protected_positions = {tuple(p) for p in (payload.get("protected_positions") or [])}
        use_global_cover = payload.get("use_global_cover", True)
    except Exception as e:
        raise gr.Error(f"Bad payload: {e}")

    import time
    t0 = time.monotonic()
    try:
        if use_global_cover:
            prep = street_main.prepare_pathfind_from_cover_chunk(payload["dots"], payload["date"])
        else:
            start = tuple(payload["start"])
            goals = [tuple(g) for g in payload["goals"]]
            edges = [(tuple(a), tuple(b)) for a, b in payload["edges"]]
            prep = street_main.prepare_pathfind(start, goals, edges)
        new_segments = street_main.run_prepared_pathfind_segments(prep, protected_positions=protected_positions)
    except Exception as e:
        raise gr.Error(f"Chunk {chunk_id} failed: {e}")
    dt = time.monotonic() - t0

    t1 = time.monotonic()
    _upload_pieces(f"{CLI_RAW_PREFIX}/{chunk_id}", new_segments, [[chunk_id]] * len(new_segments),
                    commit_message=f"cli raw chunk {chunk_id}: {len(new_segments)} segment(s)")
    dt_save = time.monotonic() - t1

    return (f"<p>Chunk {chunk_id}: {len(new_segments)} segment(s) in {dt:.1f}s "
            f"(saved raw output in {dt_save:.1f}s). Ready to bridge.</p>")


def handle_cli_bridge_chunk(chunk_id, adjacent_ids_str, progress=gr.Progress(track_tqdm=True)):
    """Scripted/CLI: bridges a chunk (already run + saved via
    handle_cli_run_chunk) onto whatever's currently checkpointed in
    CLI_JOIN_DATASET_REPO -- its own separate @spaces.GPU call (see
    handle_cli_run_chunk's docstring for why). See services.
    pipeline_runner.bridge_incremental_gpu's own docstring for why this
    does NOT re-verify pairs a previous call already merged.

    Reads the chunk's raw segments back from CLI_RAW_PREFIX (NOT from any
    in-session state) -- so this call is fully independent of
    handle_cli_run_chunk's own return value/session, and safe to retry
    (e.g. after fixing a bridge-step bug) without regenerating anything,
    in a brand new session if needed. Same for the checkpoint itself:
    downloaded at the start of this call and re-uploaded at the end (see
    _download_pieces/_upload_pieces), so it's always directly viewable
    AND resumable, in the same files, at every step.

    chunk_id: the chunk to bridge in (must have been run+saved already).
    adjacent_ids_str: JSON list of existing chunk ids this chunk is
    known-adjacent to (from the caller's own graph-level chunking, e.g.
    map_selection.candidates.split_into_chunks's known_adjacent_chunk_pairs)
    -- "[]" for the very first chunk, nothing to bridge onto yet."""
    try:
        adjacent_ids = json.loads(adjacent_ids_str) if adjacent_ids_str.strip() else []
    except Exception as e:
        raise gr.Error(f"Bad adjacent_ids: {e}")

    import time
    t0 = time.monotonic()
    new_segments, _ = _download_pieces(f"{CLI_RAW_PREFIX}/{chunk_id}")
    if not new_segments:
        raise gr.Error(f"No saved raw output for chunk {chunk_id} -- call cli_run_chunk for it first.")
    existing_pieces, existing_ids = _download_pieces(CLI_CHECKPOINT_PREFIX)
    t_download = time.monotonic() - t0

    from services.pipeline_runner import bridge_incremental_gpu
    t1 = time.monotonic()
    try:
        pieces, ids = bridge_incremental_gpu(existing_pieces, existing_ids, new_segments, chunk_id, adjacent_ids)
    except Exception as e:
        raise gr.Error(f"Bridging chunk {chunk_id} onto existing pieces failed: {e}")
    t_bridge = time.monotonic() - t1

    t2 = time.monotonic()
    urls = _upload_pieces(CLI_CHECKPOINT_PREFIX, pieces, ids, commit_message=f"cli checkpoint: {len(pieces)} piece(s)")
    t_upload = time.monotonic() - t2

    sizes = [len(id_list) for id_list in ids]
    links = "".join(f'<li><a href="{u}">{u}</a></li>' for u in urls)
    return (f"<p>Chunk {chunk_id}: merged {len(existing_pieces)} existing + {len(new_segments)} new -> {len(pieces)} "
            f"piece(s) total (chunks per piece: {sizes}) "
            f"(download {t_download:.1f}s, bridge {t_bridge:.1f}s, upload {t_upload:.1f}s).</p><ul>{links}</ul>")


def handle_cli_reset():
    """Scripted/CLI testing only: deletes the current checkpoint (bridged/
    merged state) from CLI_JOIN_DATASET_REPO, so the next cli_bridge_chunk
    call starts a fresh merge instead of building onto whatever was there
    before. Deliberately does NOT touch CLI_RAW_PREFIX -- each chunk's
    own raw generation output stays around, so a reset only costs a
    re-bridge (cheap-ish), not a full re-generate (the expensive part).
    ONE commit (a folder-delete op), not one per file -- see
    _upload_pieces's docstring for why that matters."""
    from huggingface_hub import CommitOperationDelete
    try:
        api = HfApi()
        if not any(f.startswith(CLI_CHECKPOINT_PREFIX + "/") for f in api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset")):
            return "<p>Nothing to clear -- checkpoint already empty.</p>"
        api.create_commit(
            repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset",
            operations=[CommitOperationDelete(path_in_repo=CLI_CHECKPOINT_PREFIX + "/")],
            commit_message="cli reset",
        )
    except Exception as e:
        raise gr.Error(f"Reset failed: {e}")
    return "<p>Checkpoint cleared.</p>"


def build_tab():
    state, map_view, selection_view = build_map_section()

    with gr.Row(equal_height=True):
        with gr.Column(scale=0, min_width=140):
            # Auto-path across the whole clicked graph (branches/loops
            # included). Split into three steps -- prepare (gather +
            # download, no GPU), run (the actual GPU search), join (fit +
            # merge multiple segments, no GPU) -- so the GPU-triggering
            # click is its own fresh interaction instead of following a
            # long download inside one combined request (see
            # handle_pathfind_run's docstring), and re-testing/tuning the
            # join step doesn't require re-running the expensive GPU search
            # each time (see handle_pathfind_join's docstring).
            pathfind_prepare_btn = gr.Button("1. Prepare auto-path (experimental)")
            pathfind_run_btn = gr.Button("2. Run auto-path")
            pathfind_join_btn = gr.Button("3. Join segments")
            pathfind_run_join_btn = gr.Button("2+3. Run + Join (one GPU call)")

    pathfind_status = gr.HTML()
    pathfind_prep_state = gr.State(None)
    pathfind_segments_state = gr.State(None)

    with gr.Row(equal_height=True):
        # Produced by Run -- everything Join needs (prep + segments),
        # pickled to one file. Download it to skip Prepare/Run entirely
        # next time (a later session, or after tweaking join_segments.py):
        # just re-upload it below and press "Load segments".
        pathfind_segments_file = gr.File(label="Segments file (from Run, for Join later)", interactive=False)
        with gr.Column():
            pathfind_segments_upload = gr.File(label="...or load a previously downloaded segments file", file_types=[".pkl"], type="filepath")
            pathfind_load_btn = gr.Button("Load segments")

    # Drop-ready from page load (not a static placeholder) -- lets you
    # preview an already-downloaded .ply without needing a GPU run first.
    reconstruct_view = gr.HTML(viewers.build_pointcloud_viewer())

    # Scripted/CLI-only controls for staging a large-area reconstruction
    # (e.g. a whole campus) chunk by chunk via gradio_client, incrementally
    # bridging each new chunk onto whatever's already merged instead of
    # one huge corridor in one GPU call. Run and Bridge are separate GPU
    # calls/separate buttons on purpose -- see handle_cli_run_chunk's own
    # docstring for why combining them into one handler causes 'Expired
    # ZeroGPU proxy token'. Not meant for manual clicking (payloads are
    # raw JSON), kept as real UI so it's inspectable/debuggable too.
    with gr.Accordion("Scripted staged testing (CLI, experimental)", open=False):
        cli_chunk_payload = gr.Textbox(
            label='Chunk payload JSON: {"chunk_id": ..., "start": [lat, lon], "goals": [[lat, lon], ...], "edges": [[[lat1, lon1], [lat2, lon2]], ...]}',
            lines=3,
        )
        cli_run_chunk_btn = gr.Button("1. Run chunk (GPU search, saves raw output)")
        cli_bridge_chunk_id = gr.Textbox(label="Chunk id to bridge (must already be run+saved)")
        cli_adjacent_ids = gr.Textbox(label='Existing chunk ids this chunk is known-adjacent to, JSON e.g. ["chunk0"] ("[]" for the first chunk)')
        cli_bridge_chunk_btn = gr.Button("2. Bridge chunk onto checkpoint (GPU bridge)")
        cli_status = gr.HTML()
        cli_reset_btn = gr.Button("Reset checkpoint (start a fresh merge, keeps raw chunks)")

    pathfind_prepare_btn.click(
        fn=handle_pathfind_prepare,
        inputs=[state],
        outputs=[pathfind_prep_state, pathfind_status],
        show_progress="minimal",
        show_progress_on=[pathfind_status],
    )

    pathfind_run_btn.click(
        fn=handle_pathfind_run,
        inputs=[pathfind_prep_state],
        outputs=[reconstruct_view, pathfind_segments_state, pathfind_segments_file],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    pathfind_load_btn.click(
        fn=handle_pathfind_load_segments,
        inputs=[pathfind_segments_upload],
        outputs=[pathfind_prep_state, pathfind_segments_state, pathfind_status],
        show_progress="minimal",
        show_progress_on=[pathfind_status],
    )

    pathfind_join_btn.click(
        fn=handle_pathfind_join,
        inputs=[pathfind_prep_state, pathfind_segments_state],
        outputs=[reconstruct_view],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    pathfind_run_join_btn.click(
        fn=handle_pathfind_run_and_join,
        inputs=[pathfind_prep_state],
        outputs=[reconstruct_view, pathfind_segments_state, pathfind_segments_file],
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    cli_run_chunk_btn.click(
        fn=handle_cli_run_chunk,
        inputs=[cli_chunk_payload],
        outputs=[cli_status],
        api_name="cli_run_chunk",
        show_progress="minimal",
        show_progress_on=[cli_status],
    )

    cli_bridge_chunk_btn.click(
        fn=handle_cli_bridge_chunk,
        inputs=[cli_bridge_chunk_id, cli_adjacent_ids],
        outputs=[cli_status],
        api_name="cli_bridge_chunk",
        show_progress="minimal",
        show_progress_on=[cli_status],
    )

    cli_reset_btn.click(
        fn=handle_cli_reset,
        inputs=[],
        outputs=[cli_status],
        api_name="cli_reset",
    )
