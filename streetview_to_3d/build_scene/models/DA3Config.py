"""Which DA3 to load: the model's local path."""
from dataclasses import dataclass


@dataclass
class DA3Config:
    """Which DA3 to load: the model's local path."""
    da3_model: str = ""
