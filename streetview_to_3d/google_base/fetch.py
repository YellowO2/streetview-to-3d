"""The Google panos a base is built from: the scene's own Google nodes plus
their official neighbours nearby, each with its depth map and pose."""
import asyncio
from dataclasses import dataclass, field

import aiohttp
import numpy as np
from streetlevel import streetview

from streetview_to_3d.services.geo import latlon_to_local_m
from streetview_to_3d.services.http_headers import BROWSER_HEADERS
from streetview_to_3d.services.streetview_fetch import fetch_depth_planes

NEAR_M = 15.0        # neighbours this close to any scene node join in
MAX_YEARS = 5        # ...if captured within this many years of the scene
CAM_H = 2.45         # Google's camera height above its ground, from the depth maps


@dataclass(eq=False)
class GooglePano:
    """One Google pano placed in the scene's frame (x east, y down, z north,
    metres from the scene's centre). Its depth map and plane numbers are
    256 x 512, laid out like the photo."""
    id: str
    date: str
    lat: float
    lon: float
    heading: float
    elevation: float
    depth: np.ndarray
    plane_index: np.ndarray
    in_scene: bool
    pos: np.ndarray = field(default=None)
    R: np.ndarray = field(default=None)    # direction in the photo frame = R @ (world direction)

    def place(self, lat0, lon0):
        """Camera at its GPS and elevation, turned by its heading."""
        e, n = latlon_to_local_m(self.lat, self.lon, lat0, lon0)
        self.pos = np.array([e, -(self.elevation + CAM_H), n])
        h = self.heading
        self.R = np.array([[np.cos(h), 0, -np.sin(h)], [0, 1, 0], [np.sin(h), 0, np.cos(h)]])
        return self


def _official(pano_id):
    """Google's own captures -- not user-uploaded photospheres."""
    return len(pano_id) == 22 and not pano_id.startswith("CIHM")


async def neighbours(seeds, lat0, lon0, session):
    """Google's own metadata for the official panos near a set of seed
    panos (dicts with id, lat, lon, date): each seed's neighbours within
    NEAR_M of any seed, captured within MAX_YEARS of the seeds' median
    year, with an elevation. Seeds themselves are left out."""
    if not seeds:
        return []
    seed_ids = {p["id"] for p in seeds}
    seed_en = np.array([latlon_to_local_m(p["lat"], p["lon"], lat0, lon0) for p in seeds])
    year = int(np.median([int(str(p["date"])[:4]) for p in seeds]))
    ids = set()
    for p in seeds:
        meta = await streetview.find_panorama_by_id_async(p["id"], session=session)
        for nb in (meta.neighbors if meta else []):
            en = np.array(latlon_to_local_m(nb.lat, nb.lon, lat0, lon0))
            if _official(nb.id) and nb.id not in seed_ids and np.linalg.norm(seed_en - en, axis=1).min() < NEAR_M:
                ids.add(nb.id)
    out = []
    for i in sorted(ids):
        meta = await streetview.find_panorama_by_id_async(i, session=session)
        if (meta is not None and meta.elevation is not None and meta.date is not None
                and abs(meta.date.year - year) <= MAX_YEARS):
            out.append(meta)
    return out


async def _gather(scene):
    lat0, lon0 = scene["center"][:2]
    seeds = [n["pano"] for n in scene["nodes"] if n["pano"]["source"] == "google"]
    if not seeds:
        return []
    panos = []
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as s:
        metas = [await streetview.find_panorama_by_id_async(p["id"], session=s) for p in seeds]
        metas = [m for m in metas if m is not None and m.elevation is not None and m.date is not None]
        seed_ids = {m.id for m in metas}
        for meta in metas + await neighbours(seeds, lat0, lon0, s):
            got = await fetch_depth_planes(meta.id, session=s)
            if got is None:
                continue
            panos.append(GooglePano(id=meta.id, date=str(meta.date), lat=meta.lat, lon=meta.lon,
                                    heading=meta.heading, elevation=meta.elevation,
                                    depth=got[0], plane_index=got[1], in_scene=meta.id in seed_ids))
    panos.sort(key=lambda p: p.id)
    return [p.place(lat0, lon0) for p in panos]


def gather(scene):
    """GooglePanos for a scene (a loaded scene.json dict), placed in its frame."""
    return asyncio.run(_gather(scene))
