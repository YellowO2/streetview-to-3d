_PROMPTS = {
    "Night": "Relight the same scene to nighttime with natural darkness, keeping all objects and geometry unchanged.",
    "Deep night": "Change the scene to nighttime, keeping all objects and geometry unchanged. All lights should be off.",
}


def build_prompt(preset_name: str) -> str:
    return _PROMPTS.get(preset_name, "")


PRESET_NAMES = ["(none)", *_PROMPTS.keys()]
