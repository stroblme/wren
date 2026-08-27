"""Training corpora.

Stage 1 uses a tiny synthetic corpus of natural-language commands paired
with the tool call they should produce; the point is to prove the pipeline
learns the mapping, not to generalize. Real (varied, held-out) data comes
in a later stage.
"""

import random

ROOMS = ["living room", "kitchen", "bathroom", "bedroom", "office"]
OUTSIDE = ["garden", "patio", "garage", "driveway", "balcony"]
WATERED = ["garden", "patio", "balcony"]
AREAS = ROOMS + OUTSIDE

# Each entry maps several phrasings (direct or indirect) of the same intent
# to a single tool call, so the model sees "turn the lights on" and "it's
# too dark in here" both resolve to `lights.on`.
LIGHTS_ON = ["turn on the lights in the {area}", "switch on the {area} lights", "it's too dark in the {area}"]
LIGHTS_OFF = ["turn off the lights in the {area}", "switch off the {area} lights", "it's too bright in the {area}"]
TEMPERATURE = ["what is the temperature in the {area}", "how warm is it in the {area}", "check the temperature in the {area}"]

MEDIA_PLAY = ["play music in the {area}", "put some music on in the {area}", "it's too quiet in the {area}"]
MEDIA_STOP = ["stop the music in the {area}", "turn the music off in the {area}", "turn that off in the {area}"]

HEATING_SET = ["set the heating in the {area} to 21 degrees"]
HEATING_BOOST = ["boost the heating in the {area}", "it's a bit cold in the {area}", "it's getting cold in the {area}"]
HEATING_OFF = ["turn off the heating in the {area}", "it's too warm in the {area}", "it's getting hot in the {area}"]

SHUTTERS_UP = ["open the rollershutters in the {area}", "raise the blinds in the {area}", "let some light into the {area}"]
SHUTTERS_DOWN = ["close the rollershutters in the {area}", "lower the blinds in the {area}", "the sun is too strong in the {area}"]
SHUTTERS_STOP = ["stop the rollershutters in the {area}"]

WATERING_START = ["start watering the {area}", "water the {area}", "the {area} looks dry"]
WATERING_STOP = ["stop watering the {area}", "that's enough water for the {area}"]


def _slug(area: str) -> str:
    return area.replace(" ", "_")


def _add(lines: list[str], phrasings: list[str], tool: str, area: str) -> None:
    slug = _slug(area)
    for phrasing in phrasings:
        lines.append(f"{phrasing.format(area=area)}\n>> {tool.format(slug=slug)}\n")


def home_corpus(copies: int = 40) -> str:
    """Every command/tool-call pair, repeated in `copies` shuffled passes."""
    lines = []
    for area in AREAS:
        _add(lines, LIGHTS_ON, "lights.on('{slug}')", area)
        _add(lines, LIGHTS_OFF, "lights.off('{slug}')", area)
        _add(lines, TEMPERATURE, "sensors.temperature('{slug}')", area)
    for room in ROOMS:
        _add(lines, MEDIA_PLAY, "media.play('{slug}')", room)
        _add(lines, MEDIA_STOP, "media.stop('{slug}')", room)
        _add(lines, HEATING_SET, "heating.set('{slug}', 21)", room)
        _add(lines, HEATING_BOOST, "heating.boost('{slug}')", room)
        _add(lines, HEATING_OFF, "heating.off('{slug}')", room)
        _add(lines, SHUTTERS_UP, "shutters.up('{slug}')", room)
        _add(lines, SHUTTERS_DOWN, "shutters.down('{slug}')", room)
        _add(lines, SHUTTERS_STOP, "shutters.stop('{slug}')", room)
    for area in WATERED:
        _add(lines, WATERING_START, "watering.start('{slug}')", area)
        _add(lines, WATERING_STOP, "watering.stop('{slug}')", area)
    rng = random.Random(0)
    corpus = []
    for _ in range(copies):
        rng.shuffle(lines)
        corpus += lines
    return "".join(corpus)
