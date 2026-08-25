"""Gradio wiring for the street-builder pathfind/reconstruction flow: given
an already-picked graph of nodes (see street_builder/map_selection/tab.py's
build_map_section), prepare candidates, run the corridor search, and join
segments into a final point cloud.

Composes the whole tab: mounts map_selection's own map-picking section,
then this module's own prepare/run/join controls underneath it, wired
against the same shared `state` map_selection's handlers already update.
"""
import json
import os
import uuid

import gradio as gr
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


def _download_pieces(prefix):
    """Downloads whatever's saved under `prefix` in the dataset repo (see
    _upload_pieces) and reconstructs bridgeable pieces from it -- ([],
    []) if nothing's there yet. Used for both the running checkpoint
    (CLI_CHECKPOINT_PREFIX) and each chunk's own raw output
    (CLI_RAW_PREFIX/<chunk_id>) -- same file shape (.ply + .json per
    piece) either way, just a different path. Directly
    viewable/downloadable at any point, and exactly what output_to_piece
    reconstructs from (see its own docstring for why that's enough to
    keep bridging further)."""
    from huggingface_hub import hf_hub_download
    from street_builder.reconstruction.join_segments import output_to_piece

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset")
    except Exception:
        files = []
    ply_files = sorted(f for f in files if f.startswith(prefix + "/") and f.endswith(".ply"))

    pieces, id_sets = [], []
    for ply_rel in ply_files:
        # _save_joined_pieces names these "pathfind_joined{suffix}.ply" /
        # "pathfind_metadata{suffix}.json" -- different prefixes, not
        # just a different extension, so swap the whole basename prefix
        # rather than assuming they share one.
        dirname, basename = os.path.split(ply_rel)
        if not basename.startswith("pathfind_joined") or not basename.endswith(".ply"):
            continue
        suffix = basename[len("pathfind_joined"):-len(".ply")]
        json_rel = os.path.join(dirname, f"pathfind_metadata{suffix}.json")
        if json_rel not in files:
            continue
        ply_path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=ply_rel)
        json_path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=json_rel)
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
    URLs."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    from street_builder.reconstruction.join_segments import pieces_to_output

    results = pieces_to_output(pieces, id_sets=id_sets)
    run_id = uuid.uuid4().hex
    output_dir = os.path.join(SPLATS_DIR, run_id)
    saved = street_main._save_joined_pieces(results, output_dir)

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

    payload_str: JSON {"chunk_id": ..., "start": [lat, lon],
    "goals": [[lat, lon], ...], "edges": [[[lat1, lon1], [lat2, lon2]], ...],
    "protected_keys": [...]}. protected_keys (optional): this chunk's own
    real boundary node keys (known from the chunking step, e.g.
    map_selection.candidates.split_into_chunks's boundary edges) -- kept
    even if set_cover would otherwise drop them as geographically
    redundant, since their specific identity (not just their location) is
    what cross-chunk bridging needs later. See walk_graph.
    run_pathfind_reconstruction's own docstring.

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
        start = tuple(payload["start"])
        goals = [tuple(g) for g in payload["goals"]]
        edges = [(tuple(a), tuple(b)) for a, b in payload["edges"]]
        protected_keys = set(payload.get("protected_keys") or [])
    except Exception as e:
        raise gr.Error(f"Bad payload: {e}")

    import time
    t0 = time.monotonic()
    try:
        prep = street_main.prepare_pathfind(start, goals, edges)
        new_segments = street_main.run_prepared_pathfind_segments(prep, protected_keys=protected_keys)
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
