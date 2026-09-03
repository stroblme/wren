"""Command line entry point.

`wren <prompt>` (or piped stdin) runs inference against the shipped
weights; `wren --train` retrains them from a corpus and overwrites the
weights file inside the package, which is what the wheel ships.

Training goes through fluksio (wren/flow.py): the parameters it ran with,
the loss curve it reported and the weights it produced are recorded by the
engine (`uv run fluksio serve`) instead of scrolling past in a terminal.
"""

import argparse
import pathlib
import sys
import time
from typing import Any

from wren import tokenizer
from wren.model import WEIGHTS, generate, load
from wren.train import DEFAULTS


def main() -> None:
    parser = argparse.ArgumentParser(prog="wren", description=__doc__)
    parser.add_argument("prompt", nargs="*", help="prompt; read from stdin if omitted")
    parser.add_argument(
        "--train",
        nargs="?",
        const="",
        metavar="CORPUS",
        help="train on CORPUS (default: built-in demo corpus) and save the weights",
    )
    parser.add_argument(
        "-n", "--tokens", type=int, default=64, help="bytes to generate"
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.1)
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

    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read()
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
    model = load(WEIGHTS)
    # Only the tail of an over-long prompt fits the context window.
    tokens = tokenizer.encode(prompt)[-model.cfg.context :]
    out = generate(model, tokens, args.tokens, args.temperature)
    sys.stdout.write(tokenizer.decode(out[len(tokens) :]))


def train_run(args: argparse.Namespace) -> None:
    """Run the training flow on the engine, and keep the weights it made."""
    try:
        from fluksio.sdk import SyncError
        from fluksio.sdk.cli import discover
        from fluksio.sdk.client import ApiError, Client, repo_root, sync

        from wren import flow
    except ImportError:  # fluksio is a dev-group dependency; so is training.
        sys.exit("wren --train needs fluksio: `uv sync --group dev`")

    # Absolute: the worker reads the corpus from its own working directory.
    params: dict[str, Any] = {
        "corpus": str(pathlib.Path(args.train).resolve()) if args.train else ""
    }
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
        weights = handle.download(f"{flow.NAME}.weights")
    except (SyncError, ApiError) as exc:
        sys.exit(f"fluksio: {exc} (is `uv run fluksio serve` running?)")

    WEIGHTS.write_bytes(weights)
    result = handle.result
    log(
        f"{result['n_params']:,.0f} params, final loss "
        f"{result['final_loss']:.3f}; saved {WEIGHTS}"
    )


def follow(client: Any, handle: Any, poll: float = 2.0) -> None:
    """Wait for the run, reporting where its loss has got to as it goes.

    The latest reading per poll rather than every one of them: the node
    reports a loss per step, and the whole curve is on the engine either way.
    The status is read before the numbers, so the last reading is never the
    one that gets missed.
    """
    while True:
        done = handle.refresh().done
        points = client.metrics(handle.id)
        if points:
            log(f"step {len(points):4d}  loss {points[-1]['value']:.3f}")
        if done:
            return
        time.sleep(poll)


def log(message: str) -> None:
    """Progress goes to stderr so stdout stays pipeable model output."""
    print(message, file=sys.stderr)
