"""Command line interface for inference and training.

`wren <prompt>` or piped input runs the shipped weights. `wren --train
home.json` runs the Fluksio training flow on an installation's targets and
examples and writes the model's pack beside it, as `home.bin`.
"""

import argparse
import pathlib
import sys
import time
from typing import Any

from wren import tokenizer


def main() -> None:
    try:  # JAX is an optional extra; the NumPy runtime does not need it.
        from wren.model import WEIGHTS, generate_batch, load
        from wren.train import DEFAULTS
    except ImportError:
        sys.exit("the wren CLI needs the train extra: pip install 'wren[train]'")

    parser = argparse.ArgumentParser(prog="wren", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt; read from stdin if omitted")
    parser.add_argument(
        "--train",
        metavar="JSON",
        help="train on an installation's targets and examples; write the pack as .bin",
    )
    parser.add_argument(
        "-n", "--tokens", type=int, default=64, help="bytes to generate"
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    # `--dim`, `--steps`, `--lr`, ... — one flag per knob, from the one place
    # its default lives. Every value is sent, so the run records the whole
    # configuration rather than only what was said out loud.
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default,
            help=f"training: {name} (default: %(default)s)",
        )
    args = parser.parse_args()

    if args.train is not None:
        train_run(args)
        return

    # Argv prompts end with a newline, as in the C runtime.
    prompt = " ".join(args.prompt) + "\n" if args.prompt else sys.stdin.read()
    if not prompt.strip():
        parser.error("empty prompt")
    if not WEIGHTS.exists():
        parser.error(f"no weights at {WEIGHTS}; run `wren --train` first")
    # Persist compiled programs across invocations: the generation loop
    # otherwise recompiles on every CLI start, dwarfing the generation.
    import jax

    jax.config.update(
        "jax_compilation_cache_dir",
        str(pathlib.Path.home() / ".cache" / "wren" / "jax"),
    )
    tokens = tokenizer.encode(prompt)
    [out], _ = generate_batch(
        load(WEIGHTS),
        [tokens],
        args.tokens,
        stop=(10, 10),
        temperature=args.temperature,
    )
    # A blank line ends the answer; the prompt's own newline may open it.
    if tokens[-1:] == out[:1] == [10]:
        out = out[:1]
    sys.stdout.write(tokenizer.decode(out))


def train_run(args: argparse.Namespace) -> None:
    """Run the training flow and write its pack beside the installation's JSON."""
    try:
        from fluksio.sdk import SyncError
        from fluksio.sdk.cli import discover
        from fluksio.sdk.client import ApiError, Client, repo_root, sync

        from wren import flow
        from wren.train import DEFAULTS
    except ImportError:  # fluksio is a dev-group dependency; so is training.
        sys.exit("wren --train needs fluksio: `uv sync --group dev`")

    # Absolute: the worker reads the file from its own working directory.
    installation = pathlib.Path(args.train).resolve()
    params: dict[str, Any] = {"installation": str(installation)}
    params |= {name: getattr(args, name) for name in DEFAULTS}

    try:
        client = Client()
        # Upload what this checkout declares before running it, the way
        # `fluksio run` does: the reason to train is usually the edit before it.
        sync(discover([flow.__name__]), client, repo=repo_root(__file__))
        # no_cache: `--train` is asked for to train, not to be handed the
        # result an identical earlier run already produced.
        handle = client.submit(flow.NAME, params, no_cache=True, cause="cli")
        log(f"{handle.id}  {flow.NAME}  {params}")
        follow(client, handle)
        if handle.status != "ok":
            for failed in handle.failures:
                log(f"{failed.get('node', '')}  {failed.get('error', '')}")
                log(str(failed.get("logs") or ""))
            sys.exit(f"run {handle.id} {handle.status}")
        # Artifacts and metrics are named for the flow that reported them.
        pack = handle.download(f"{flow.NAME}.pack")
    except (SyncError, ApiError) as exc:
        sys.exit(f"fluksio: {exc} (is `uv run fluksio serve` running?)")

    out = installation.with_suffix(".bin")
    out.write_bytes(pack)
    result = handle.result
    log(
        f"{result['n_params']:,.0f} params, final loss {result['final_loss']:.3f}, "
        f"held-out exact match {result['exact_match']:.1%}, coverage "
        f"{result['coverage']:.1%} at threshold {result['threshold']:.2f}; saved {out}"
    )


def follow(client: Any, handle: Any, poll: float = 2.0) -> None:
    """Wait for the run, logging the latest loss after each status poll."""
    while True:
        done = handle.refresh().done
        points = client.metrics(handle.id)
        if points:
            log(f"step {len(points):4d}  loss {points[-1]['value']:.3f}")
        if done:
            return
        time.sleep(poll)


def log(message: str) -> None:
    """Write progress to stderr so stdout contains only model output."""
    print(message, file=sys.stderr)
