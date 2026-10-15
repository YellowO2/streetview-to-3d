"""Which side of the road cars keep to where the scene is."""
import logging

logger = logging.getLogger(__name__)

# countries driving on the left (ISO 3166-1 alpha-2)
LEFT = {"GB", "IE", "IM", "JE", "GG", "MT", "CY", "JP", "SG", "MY", "BN", "HK", "MO", "TH", "ID", "TL",
        "IN", "LK", "BD", "BT", "NP", "PK", "MV", "AU", "NZ", "PG", "FJ", "WS", "TO", "SB", "KI", "TV",
        "NR", "CK", "NU", "ZA", "NA", "BW", "ZW", "ZM", "MW", "MZ", "TZ", "KE", "UG", "LS", "SZ", "MU",
        "SC", "JM", "TT", "BB", "BS", "GY", "SR", "AG", "DM", "GD", "KN", "LC", "VC", "VG", "VI", "KY",
        "BM", "FK", "AI", "MS", "TC", "SH"}


def driving_side(site):
    """"left" or "right": the side cars keep to in the country of site's first pano with an
    elevation (Site), asked of Google; "right" if it cannot say."""
    try:
        return _side(site.panos[0].id) if site.panos else "right"
    except (OSError, ValueError) as e:
        logger.warning(f"terrain: no country for the cars' side ({e!r})")
        return "right"


def _side(pano_id):
    """"left" or "right": the side cars keep to where pano_id was taken."""
    import aiohttp
    from streetlevel import streetview
    from streetview_to_3d.tools.streetview_fetch.browser_session import BROWSER_HEADERS
    from streetview_to_3d.tools.run_async import run_async

    async def country():
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
            pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
            return pano.country_code if pano else None
    return "left" if run_async(country()) in LEFT else "right"
