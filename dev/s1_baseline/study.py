"""Run the AdamW baseline on the home corpus as a Fluksio flow.

Start the engine with `uv run fluksio serve --gpus 1`. See README.md for
the study protocol and reproduction commands.
"""

import pathlib
import tempfile

import fluksio
from fluksio import Flow, Port, node

# One seen prompt per device domain, one indirect phrasing among them; greedy
# decode must reproduce the exact tool call.
EVAL = [
    ("turn on the lights in the kitchen\n", ">> lights.on('kitchen')"),
    ("it's too dark in the office\n", ">> lights.on('office')"),
    ("boost the heating in the bedroom\n", ">> heating.boost('bedroom')"),
    ("water the garden\n", ">> watering.start('garden')"),
    ("stop the music in the living room\n", ">> media.stop('living_room')"),
    ("close the rollershutters in the office\n", ">> shutters.down('office')"),
]


@node(provides=Port("corpus", "str"))
def corpus(copies: int = 40) -> str:
    import random

    from wren import dataset as d
    from wren.home import AREAS, ROOMS

    # The template corpus s1 ran on, rebuilt from the phrase lists; the
    # state-conditioned generator replaced it in 8764546.
    lines = []

    def add(phrasings: list[str], tool: str, area: str) -> None:
        slug = area.replace(" ", "_")
        for phrasing in phrasings:
            command = phrasing.format(area=area, temp=21)
            lines.append(f"{command}\n>> {tool.format(slug=slug)}\n")

    for area in AREAS:
        add(d.LIGHTS_ON, "lights.on('{slug}')", area)
        add(d.LIGHTS_OFF, "lights.off('{slug}')", area)
        add(d.TEMPERATURE, "sensors.temperature('{slug}')", area)
    for room in ROOMS:
        add(d.MEDIA_PLAY, "media.play('{slug}')", room)
        add(d.MEDIA_STOP, "media.stop('{slug}')", room)
        add(d.HEATING_SET, "heating.set('{slug}', 21)", room)
        add(d.HEATING_BOOST, "heating.boost('{slug}')", room)
        add(d.HEATING_OFF, "heating.off('{slug}')", room)
        add(d.SHUTTERS_UP, "shutters.up('{slug}')", room)
        add(d.SHUTTERS_DOWN, "shutters.down('{slug}')", room)
        add(d.SHUTTERS_STOP, "shutters.stop('{slug}')", room)
    for area in ["garden", "patio", "balcony"]:
        add(d.WATERING_START, "watering.start('{slug}')", area)
        add(d.WATERING_STOP, "watering.stop('{slug}')", area)
    rng = random.Random(0)
    out = []
    for _ in range(copies):
        rng.shuffle(lines)
        out += lines
    return "".join(out)


@node(
    requires=[
        Port("corpus", "str"),
        Port("seed", "int"),
        Port("steps", "int"),
        Port("lr", "float"),
    ],
    provides=[
        Port("loss", "float", stream=True),
        Port("final_loss", "float"),
        Port("accuracy", "float"),
        Port("weights", "artifact"),
    ],
    # preallocate=false: jax otherwise takes 75% of the card at import, so a
    # second engine (a `--local` run beside a running `serve`) collides with
    # this one instead of fitting beside it.
    resources={"gpus": 1, "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}},
)
def baseline(
    corpus: str,
    seed: int,
    steps: int = 600,
    batch_size: int = 32,
    seq_len: int = 128,
    lr: float = 3e-3,
    log_every: int = 25,
) -> dict:
    import os

    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from wren import tokenizer
    from wren.model import Config, Wren, generate, save
    from wren.train import sample_batch, train_step

    # An unsatisfiable `gpus: 1` arrives as CUDA_VISIBLE_DEVICES="" and would
    # train on the CPU without saying so; fail instead.
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        raise RuntimeError("no card declared: `fluksio serve --gpus 1`")

    model = Wren(Config(dim=128, n_layers=8), rngs=nnx.Rngs(seed))
    data = jnp.array(tokenizer.encode(corpus), dtype=jnp.int32)
    optimizer = nnx.Optimizer(model, optax.adamw(lr), wrt=nnx.Param)
    key = jax.random.PRNGKey(seed)
    loss = float("nan")
    for step in range(steps):
        key, sub = jax.random.split(key)
        x, y = sample_batch(data, batch_size, seq_len, sub)
        loss = float(train_step(model, optimizer, x, y))
        if step % log_every == 0 or step == steps - 1:
            fluksio.emit(loss=loss)

    hits = 0
    for prompt, expected in EVAL:
        tokens = tokenizer.encode(prompt)
        out = generate(model, tokens, len(expected.encode()) + 2, temperature=0.05)
        hits += tokenizer.decode(out[len(tokens) :]).startswith(expected)

    path = pathlib.Path(tempfile.mkdtemp()) / "weights.npz"
    save(model, path)
    ref = fluksio.save_artifact(path.read_bytes(), name="weights.npz")
    return {"final_loss": loss, "accuracy": hits / len(EVAL), "weights": ref}


Flow(
    "s1_baseline",
    title="S1 — baseline AdamW",
    nodes=[corpus, baseline],
    inputs=[
        Port("seed", "int", initial=0),
        Port("steps", "int", initial=1000),
        Port("lr", "float", initial=3e-3),
    ],
    outputs=["final_loss", "accuracy"],
)
