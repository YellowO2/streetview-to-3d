"""Upload a finished scene to the public gallery.

The scene's files go to the public gallery dataset, listed in index.json, and its viewer
link comes back. Nothing is published where TOKEN_ENV is unset (the UI hides its checkbox).

    GALLERY_TOKEN=hf_... python -m street_builder.tools.gallery.publish SCENE_DIR
"""
import datetime
import json
import logging
import os
import sys
import urllib.parse
import urllib.request

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.utils import disable_progress_bars

from streetview_to_3d.common.Scene import FILENAME as SCENE_FILENAME

logger = logging.getLogger(__name__)

REPO = "potato-bug/street-view-to-3d-scenes"   # a public dataset; index.html reads the same one
PAGE = "https://yellowo2.github.io/streetview-to-3d/"   # where index.html and viewer.html are served
TOKEN_ENV = "GALLERY_TOKEN"                    # a write token for REPO; unset: nothing is published
INDEX = "index.json"
SAME = 0.8          # a run with this share of its panos in one listed scene is already there
RETRIES = 3         # index.json is rewritten per upload; retry when another run got in first


def _viewable(run_dir):
    """The files the viewer loads: scene.json and whatever it names."""
    with open(os.path.join(run_dir, SCENE_FILENAME)) as f:
        data = json.load(f)
    names = [SCENE_FILENAME] + [n["ply"] for n in data["nodes"] if n.get("ply")]
    names += [v for v in data.values() if isinstance(v, str) and v.endswith((".ply", ".json"))]
    return data, [n for n in names if os.path.isfile(os.path.join(run_dir, n))]


def _place_name(lat, lon):
    """A name like "Himi, Toyama Prefecture, Japan" from OpenStreetMap's Nominatim; the
    coordinates if it gives nothing."""
    url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 14, "accept-language": "en"})
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "streetview-to-3d gallery"})
        address = json.load(urllib.request.urlopen(request, timeout=10)).get("address", {})
        parts = [next((address[k] for k in keys if k in address), None)
                 for keys in (("city", "town", "village", "hamlet", "suburb", "municipality"),
                              ("state", "province", "county"), ("country",))]
        name = ", ".join(dict.fromkeys(p for p in parts if p))
        if name:
            return name
    except Exception:
        pass
    return f"{lat:.4f}, {lon:.4f}"


def _link(entry, repo=REPO):
    """The viewer's URL for a listed scene."""
    scene = f"https://huggingface.co/datasets/{repo}/resolve/main/scenes/{entry['id']}/scene.json"
    return PAGE + "viewer.html?" + urllib.parse.urlencode({"scene": scene, "name": entry["place"]})


def _already(scenes, panos):
    """The listed scene holding most of these panos, if it holds SAME of them."""
    panos = set(panos)
    best = max(scenes, key=lambda s: len(panos & set(s["panos"])), default=None)
    if best and panos and len(panos & set(best["panos"])) >= SAME * len(panos):
        return best
    return None


def sharing():
    """Whether scenes can be published here: the gallery's write token is set."""
    return bool(os.environ.get(TOKEN_ENV))


def publish(run_dir, repo=REPO, api=None):
    """Upload run_dir's scene as scenes/<id>/ and add it to index.json. Returns its viewer
    link, or the link of the listed scene that already covers these panos."""
    api = api or HfApi(token=os.environ[TOKEN_ENV])
    disable_progress_bars()   # one bar per file floods the Space's log
    data, names = _viewable(run_dir)
    panos = [n["pano"]["id"] for n in data["nodes"] if n.get("ply")]
    lat, lon = data["center"]
    entry = {
        "id": os.path.basename(os.path.normpath(run_dir)),
        "place": _place_name(lat, lon), "lat": lat, "lon": lon,
        "date": datetime.date.today().isoformat(), "panos": panos,
        "bytes": sum(os.path.getsize(os.path.join(run_dir, n)) for n in names),
    }
    for _ in range(RETRIES):
        head = api.dataset_info(repo).sha
        try:
            with open(hf_hub_download(repo, INDEX, repo_type="dataset", revision=head,
                                      token=api.token)) as f:
                scenes = json.load(f)["scenes"]
        except EntryNotFoundError:
            scenes = []
        found = _already(scenes, panos)
        if found:
            logger.info(f"gallery: already there as {found['id']}")
            return _link(found, repo)
        index = json.dumps({"scenes": scenes + [entry]}, indent=1).encode()
        operations = [CommitOperationAdd(f"scenes/{entry['id']}/{n}", os.path.join(run_dir, n))
                      for n in names] + [CommitOperationAdd(INDEX, index)]
        try:
            api.create_commit(repo, operations, repo_type="dataset", parent_commit=head,
                              commit_message=f"Add {entry['place']} ({entry['id']})")
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in (409, 412):
                continue   # another run committed first: read the index again
            raise
        logger.info(f"gallery: published {entry['id']} ({entry['bytes'] / 1e6:.0f} MB)")
        return _link(entry, repo)
    raise RuntimeError("gallery: the index kept changing; not published")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(publish(sys.argv[1]))
