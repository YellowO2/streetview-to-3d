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


async def _gather(scene):
    lat0, lon0 = scene["center"][:2]
    seeds = [n["pano"] for n in scene["nodes"] if n["pano"]["source"] == "google"]
    if not seeds:
        return []
    seed_ids = {p["id"] for p in seeds}
    seed_en = np.array([latlon_to_local_m(p["lat"], p["lon"], lat0, lon0) for p in seeds])
    year = int(np.median([int(str(p["date"])[:4]) for p in seeds]))
    ids = set(seed_ids)
    panos = []
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as s:
        for p in seeds:
            meta = await streetview.find_panorama_by_id_async(p["id"], session=s)
            for nb in (meta.neighbors if meta else []):
                en = np.array(latlon_to_local_m(nb.lat, nb.lon, lat0, lon0))
                if _official(nb.id) and np.linalg.norm(seed_en - en, axis=1).min() < NEAR_M:
                    ids.add(nb.id)
        for i in sorted(ids):
            meta = await streetview.find_panorama_by_id_async(i, session=s)
            if meta is None or meta.elevation is None or meta.date is None:
                continue
            if i not in seed_ids and abs(meta.date.year - year) > MAX_YEARS:
                continue
            got = await fetch_depth_planes(i, session=s)
            if got is None:
                continue
            panos.append(GooglePano(id=i, date=str(meta.date), lat=meta.lat, lon=meta.lon,
                                    heading=meta.heading, elevation=meta.elevation,
                                    depth=got[0], plane_index=got[1], in_scene=i in seed_ids))
    return [p.place(lat0, lon0) for p in panos]


def gather(scene):
    """GooglePanos for a scene (a loaded scene.json dict), placed in its frame."""
    return asyncio.run(_gather(scene))
