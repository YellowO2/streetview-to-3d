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

# Where cli_join pushes its results -- Hub-native storage (fast up/download,
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


def handle_cli_run_chunk(payload_str, accum, progress=gr.Progress(track_tqdm=True)):
    """Scripted/CLI testing only: runs Prepare+Run for ONE explicit chunk
    of a larger corridor, bypassing map_selection's `state` entirely --
    driven by gradio_client, not the map UI. Lets a large area (e.g. a
    whole campus) be walked through this pipeline in small pieces from a
    script: fetch the full real graph once (locally, via
    map_selection.candidates.expand_area -- no GPU needed for that), slice
    its edges into chunks, then call this endpoint once per chunk, each
    tagged with its own chunk_id.

    payload_str: JSON {"chunk_id": int, "start": [lat, lon],
    "goals": [[lat, lon], ...], "edges": [[[lat1, lon1], [lat2, lon2]], ...]}.
    accum: {"segments": [...], "chunk_ids": [...]} accumulated across
    every chunk run so far in this session (gr.State, persists across
    calls on the same gradio_client session) -- feed straight into
    handle_cli_join once enough chunks are in."""
    try:
        payload = json.loads(payload_str)
        chunk_id = payload["chunk_id"]
        start = tuple(payload["start"])
        goals = [tuple(g) for g in payload["goals"]]
        edges = [(tuple(a), tuple(b)) for a, b in payload["edges"]]
    except Exception as e:
        raise gr.Error(f"Bad payload: {e}")

    accum = accum or {"segments": [], "chunk_ids": []}
    try:
        prep = street_main.prepare_pathfind(start, goals, edges)
        segments = street_main.run_prepared_pathfind_segments(prep)
    except Exception as e:
        raise gr.Error(f"Chunk {chunk_id} failed: {e}")

    accum = {
        "segments": accum["segments"] + segments,
        "chunk_ids": accum["chunk_ids"] + [chunk_id] * len(segments),
    }
    n_chunks = len(set(accum["chunk_ids"]))
    return accum, f"<p>Chunk {chunk_id}: {len(segments)} segment(s). Accumulated: {len(accum['segments'])} segment(s) across {n_chunks} chunk(s).</p>"


def handle_cli_join(pairs_str, accum, progress=gr.Progress(track_tqdm=True)):
    """Scripted/CLI testing only: joins every chunk accumulated so far
    (see handle_cli_run_chunk) via street_main.save_joined_pathfind,
    scoping bridging to declared-adjacent chunk pairs only. Safe to call
    repeatedly as more chunks get added -- always re-joins the FULL
    accumulated set, not just the newest chunk (so don't call this after
    every single chunk on a large run -- each call re-bridges pairs that
    already succeeded in a previous call too, since nothing here
    remembers prior merges; checkpoint every several chunks instead).

    Results are pushed straight to CLI_JOIN_DATASET_REPO (a HF dataset,
    not the Space's own local disk) -- large-file up/download through
    huggingface_hub's own transfer goes straight to Hub storage, instead
    of routing through Gradio's file-serving proxy (slow, and the
    Space's local disk doesn't survive a redeploy anyway). Requires an
    HF_TOKEN secret configured on the Space itself (Settings -> Variables
    and secrets) with write access -- HfApi() picks that up from the
    environment automatically, no token handling needed here.

    pairs_str: JSON [[chunk_id_a, chunk_id_b], ...] -- empty/"" means no
    restriction (blind O(n^2) bridging attempt over every segment pair,
    only reasonable for a small accumulated segment count)."""
    if not accum or not accum.get("segments"):
        raise gr.Error("No chunks run yet -- call cli_run_chunk first.")
    try:
        pairs = [tuple(p) for p in json.loads(pairs_str)] if pairs_str.strip() else None
    except Exception as e:
        raise gr.Error(f"Bad known_adjacent_chunk_pairs: {e}")

    run_id = uuid.uuid4().hex
    output_dir = os.path.join(SPLATS_DIR, run_id)
    try:
        results = street_main.save_joined_pathfind(
            accum["segments"], output_dir, chunk_ids=accum["chunk_ids"], known_adjacent_chunk_pairs=pairs,
        )
    except Exception as e:
        raise gr.Error(f"Join failed: {e}")

    api = HfApi()
    uploaded = []
    for fname in sorted(os.listdir(output_dir)):
        path_in_repo = f"cli_join/{run_id}/{fname}"
        try:
            api.upload_file(
                path_or_fileobj=os.path.join(output_dir, fname),
                path_in_repo=path_in_repo,
                repo_id=CLI_JOIN_DATASET_REPO,
                repo_type="dataset",
            )
            uploaded.append(f"https://huggingface.co/datasets/{CLI_JOIN_DATASET_REPO}/blob/main/{path_in_repo}")
        except Exception as e:
            raise gr.Error(f"Join succeeded but upload to {CLI_JOIN_DATASET_REPO} failed: {e}")

    labels = "".join(f"<li>{label}</li>" for label, _ in results)
    links = "".join(f'<li><a href="{u}">{u}</a></li>' for u in uploaded)
    return f"<p>Joined {len(results)} piece(s):</p><ul>{labels}</ul><p>Uploaded to {CLI_JOIN_DATASET_REPO}:</p><ul>{links}</ul>"


def handle_cli_reset():
    """Scripted/CLI testing only: clears the accumulated chunk state, to
    start a fresh staged run without restarting the whole Space."""
    return None, "<p>Cleared.</p>"


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
    # (e.g. a whole campus) chunk by chunk via gradio_client, instead of
    # one huge corridor in one GPU call -- see handle_cli_run_chunk's own
    # docstring. Not meant for manual clicking (payloads are raw JSON),
    # kept as real UI so it's inspectable/debuggable in the browser too.
    with gr.Accordion("Scripted staged testing (CLI, experimental)", open=False):
        cli_chunk_payload = gr.Textbox(
            label='Chunk payload JSON: {"chunk_id": int, "start": [lat, lon], "goals": [[lat, lon], ...], "edges": [[[lat1, lon1], [lat2, lon2]], ...]}',
            lines=3,
        )
        cli_run_chunk_btn = gr.Button("Run chunk (prepare + GPU search)")
        cli_status = gr.HTML()
        cli_accum_state = gr.State(None)

        cli_pairs = gr.Textbox(label="known_adjacent_chunk_pairs JSON, e.g. [[0, 1], [1, 2]] (blank = try all pairs)")
        cli_join_btn = gr.Button("Join accumulated chunks")
        cli_reset_btn = gr.Button("Reset accumulated chunks")

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
        inputs=[cli_chunk_payload, cli_accum_state],
        outputs=[cli_accum_state, cli_status],
        api_name="cli_run_chunk",
        show_progress="minimal",
        show_progress_on=[cli_status],
    )

    cli_join_btn.click(
        fn=handle_cli_join,
        inputs=[cli_pairs, cli_accum_state],
        outputs=[reconstruct_view],
        api_name="cli_join",
        show_progress="minimal",
        show_progress_on=[reconstruct_view],
    )

    cli_reset_btn.click(
        fn=handle_cli_reset,
        inputs=[],
        outputs=[cli_accum_state, cli_status],
        api_name="cli_reset",
    )
