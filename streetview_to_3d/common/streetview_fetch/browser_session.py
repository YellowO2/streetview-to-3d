"""Open an HTTP session that looks like a browser on Google Maps."""
import aiohttp

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps/",
    "Accept-Language": "en-US,en;q=0.9",
}


def browser_session(**options):
    """An aiohttp.ClientSession sending BROWSER_HEADERS; options are ClientSession's own."""
    return aiohttp.ClientSession(headers=BROWSER_HEADERS, **options)
