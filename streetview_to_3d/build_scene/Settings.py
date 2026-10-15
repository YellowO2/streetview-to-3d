"""One run's settings: how hard the GPU walk tries, and what follows it."""
from dataclasses import dataclass, field

from streetview_to_3d.build_scene.postprocess.Settings import Settings as PostprocessSettings
from streetview_to_3d.build_scene.reconstruct.WalkSettings import WalkSettings


@dataclass(frozen=True)
class Settings:
    """walk: the GPU walk's (reconstruct.WalkSettings); post: the postprocess's (postprocess.Settings)."""
    walk: WalkSettings = field(default_factory=WalkSettings)
    post: PostprocessSettings = field(default_factory=PostprocessSettings)
