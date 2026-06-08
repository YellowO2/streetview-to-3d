_SCENES = {
    "Morning":     "early morning",
    "Midday":      "midday",
    "Golden hour": "golden hour",
    "Dusk":        "dusk just after sunset",
    "Night":       "late night with natural darkness",
    "Overcast":    "an overcast day",
}


def build_prompt(preset_name: str) -> str:
    """Return the full prompt for a preset, or empty string if not found."""
    scene = _SCENES.get(preset_name)
    if not scene:
        return ""
    return f"Relight the same scene to {scene}, keeping all objects and geometry unchanged."


PRESET_NAMES = ["(none)", *_SCENES.keys()]
