# Development and integration

## Checkout setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git,
then run:

```sh
git clone https://github.com/stroblme/wren.git
cd wren
uv sync --locked
uv run wren "turn on the kitchen light"
```

The checkout pins Python 3.14 in `.python-version`; the package supports
Python 3.12 and newer. uv can download the pinned interpreter if needed.
The default `dev` group includes the training extra, Fluksio, pytest, Ruff,
ty and study dependencies. No GPU is required for installation training.

For an inference-only checkout, use `uv sync --locked --no-dev` and run
commands with `uv run --no-dev wren ...`. For CPU training, the default
setup is sufficient. Python `fit` uses JAX's available backend; set
`JAX_PLATFORMS=cpu` before starting Python or the engine to force CPU use.

## Tracked training

From the checkout, start the local engine in its own terminal:

```sh
uv run fluksio serve
```

Keep it running, then use a second terminal in the same directory:

```sh
uv run wren --train home.json
```

Follow the [How-to](guide.md) to create `home.json`. The command trains
through `wren.flow`, records parameters, per-step loss, evaluation scores
and the pack, and downloads `home.bin`. Local engine state is stored in
the ignored `.fluksio/` directory.

To retrain a copy of the demo installation without replacing packaged data:

```sh
cp wren/home.json home.json
uv run wren --train home.json
uv run wren --targets home.json --model home.bin "turn on the kitchen light"
```

`home.json` and `home.bin` are your local outputs; keep them out of a source
change unless you intend to publish a model.

## Checks and documentation

Run the same checks as CI:

```sh
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
uv build
```

The runtime tests compile `runtime/wren.c`, so a C compiler must be on
`PATH`. The tests include small training checks but do not retrain the
shipped demo model.

Preview and build the documentation from the repository root:

```sh
uvx zensical==0.0.72 serve
uvx zensical==0.0.72 build --clean --strict
```

The site is configured in `zensical.toml`, built into `site/` and published
by `.github/workflows/docs.yml`. Edit the Markdown in `docs/`; edit
`README.md` for the repository overview. Ignored working notes are excluded
from the site. To update the architecture drawing, edit
`docs/architecture.d2` and render it with
`d2 docs/architecture.d2 docs/architecture.svg`.

## Research studies

Each study under
[`dev/`](https://github.com/stroblme/wren/tree/main/dev) has a README with its
question, findings and reproduction commands. Study s12 runs with the current
target API; its published measurements used the older demo data at tag
`v1.1.0`. Its constant learning rate also differs from `fit`'s current cosine
decay.

Studies s1–s10 document historical models and data. Their READMEs target
`v1.1.0`, which retains the home API removed in Wren 2.0. Use a separate
checkout for those commands:

```sh
git clone --branch v1.1.0 https://github.com/stroblme/wren.git wren-v1
cd wren-v1
uv sync --locked --group cuda12
uv run --group cuda12 fluksio serve --gpus 1
```

Training those studies requires a compatible NVIDIA GPU and driver. Current
HEAD offers `cuda12` and `cuda13` dependency groups. Include the selected
group in later `uv sync` commands, which otherwise remove those dependencies.
In a second terminal at the study's checkout, follow its README. Use Fluksio to
track experiments; data downloads go into ignored `data/`, and plots into
ignored `dev/*/figures/`.

## Wren in Fluksio

Fluksio's `wren` automation node runs Wren on typed or spoken commands. Its
wired devices define the targets; your installation's model interprets
commands and Fluksio applies calls or sends unhandled work to its assistant.
Register this checkout's training flow on the engine used by the node:

```sh
uv run fluksio sync wren.flow
```

The node's Train button submits its target definitions, examples and tests
to that flow. See the
[Fluksio node reference](https://docs.fluksio.com/reference/node-types/#wren)
for node settings and engine setup.

Wren 2.0's library API uses `list[Target]` and returns `list[Call]`, with
confidence on each call. Code using the old `House` or `Device` API must
migrate; the demo home is now JSON data rather than a built-in device model.
