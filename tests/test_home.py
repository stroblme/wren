"""Check imports, state rendering, area selection, and shipped-model calls."""

import random
import subprocess
import sys

from wren import Call, Device, House, act, render_state, select_areas
from wren.dataset import _house, _render_state

HOUSE = House(
    time="19:30",
    outdoor_c=6.2,
    dark=True,
    user_area="kitchen",
    areas={
        "living room": [
            Device("light", "lq7rt", "off"),
            Device("media", "mz4kd", "off"),
            Device("heat", "hx2pw", "19C"),
        ],
        "kitchen": [Device("light", "lk3m9", "off"), Device("shutter", "sv8xr", "up")],
        "bedroom": [Device("light", "lb6ne", "on")],
        "garden": [Device("light", "lg5ya", "off"), Device("water", "wt8uc", "off")],
    },
)


def test_import_does_not_load_jax():
    code = "import sys, wren; from wren import act; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_render_state():
    assert render_state(HOUSE, ["living room", "kitchen"]) == (
        "19:30 out 6C dark\n"
        "user kitchen\n"
        "living room: light lq7rt off, media mz4kd off, heat hx2pw 19C\n"
        "kitchen: light lk3m9 off, shutter sv8xr up\n"
    )


def test_dataset_renders_through_render_state():
    """Check that shared state rendering preserves the training format."""
    house = _house(random.Random(0), set())
    assert _render_state(house, list(house["areas"])) == (
        "18:14 out 10C dark\n"
        "user bathroom\n"
        "living room: light l52pi off, media mn3jr on, heat hsj2v 19C\n"
        "bathroom: light lrdpv on\n"
        "office: light li3ta off\n"
        "kitchen: light lv4kz on\n"
    )


def test_select_areas():
    # The named area, padded with others to three lines, in house order.
    named = select_areas(HOUSE, "turn on the lights in the Garden")
    assert named == ["living room", "kitchen", "garden"]
    # Area-less: the user's area and the living room.
    assert select_areas(HOUSE, "turn off the light", limit=2) == [
        "living room",
        "kitchen",
    ]
    # Padding goes first when the prompt outgrows the budget.
    assert select_areas(HOUSE, "water the garden", budget=110) == ["garden"]


def test_act():
    cold_away = House("21:10", 4, True, None, HOUSE.areas)
    cases = [
        (
            HOUSE,
            "turn on the lights in the kitchen",
            [Call("lights.on", {"id": "lk3m9"})],
        ),
        (HOUSE, "turn off the light", [Call("lights.off", {"id": "lk3m9"})]),
        (
            HOUSE,
            "set the heating in the living room to 21 degrees",
            [Call("heating.set", {"id": "hx2pw", "temp": 21})],
        ),
        (
            HOUSE,
            "what is the temperature in the garden",
            [Call("sensors.temperature", {"area": "garden"})],
        ),
        (
            cold_away,
            "coming home in 15m",
            [
                Call("heating.boost", {"id": "hx2pw"}),
                Call("lights.on", {"id": "lq7rt"}, delay_s=900.0),
            ],
        ),
        (HOUSE, "tell me a joke about cats", []),
    ]
    for house, command, want in cases:
        calls, confidence = act(command, house)
        assert calls == want, command
        assert 0 < confidence <= 1


def test_act_maps_any_id_shape():
    """Check that device aliases map generated calls back to caller-provided IDs."""
    house = House(
        "08:00",
        12,
        False,
        "office",
        {
            "office": [Device("light", "light.office_ceiling", "on")],
            "garden": [Device("water", "3f2a9c1e-valve", "off")],
        },
    )
    assert act("switch off the office lights", house)[0] == [
        Call("lights.off", {"id": "light.office_ceiling"})
    ]
    assert act("water the garden", house)[0] == [
        Call("watering.start", {"id": "3f2a9c1e-valve"})
    ]
