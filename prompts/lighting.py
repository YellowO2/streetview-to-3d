_PROMPTS = {
    "Morning": "Relight the same scene to early morning, keeping all objects and geometry unchanged.",
    "Midday": "Relight the same scene to midday, keeping all objects and geometry unchanged.",
    "Dusk": "Relight the same scene to dusk just after sunset, keeping all objects and geometry unchanged.",
    "Night": "Relight the same scene to late night with natural darkness, keeping all objects and geometry unchanged.",
    "Deep night": "Change the scene to nighttime, keeping all objects and geometry unchanged. All lights should be off.",
}


def build_prompt(preset_name: str) -> str:
    return _PROMPTS.get(preset_name, "")


PRESET_NAMES = ["(none)", *_PROMPTS.keys()]
