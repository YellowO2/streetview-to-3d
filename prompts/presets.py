_PRESETS = {
    "Night": {
        "prompt": "Relight the same scene to nighttime with natural darkness, keeping all objects and geometry unchanged.",
        "mode": "general",
        "geom_preserving": True,
    },
    "Remove people & vehicles": {
        "prompt": "Remove all people and vehicles from the scene, filling the gaps naturally.",
        "mode": "remove_objects",
        "geom_preserving": False,
    },
}

PRESET_NAMES = ["(none)", *_PRESETS.keys()]


def get_preset(name):
    return _PRESETS.get(name)


def build_prompt(name):
    p = _PRESETS.get(name)
    return p["prompt"] if p else ""
