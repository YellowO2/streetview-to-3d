"""What to lay round a scene beyond the map's own world."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """google: a google.ply (Google's 3D Tiles points, saved by the viewer's ?google-save) to
    merge the scene into (google/main.py), None for the map's own world; bend: bend DA3's
    points onto Google's shape as they merge (False: leave them as they are)."""
    google: str | None = None
    bend: bool = True
