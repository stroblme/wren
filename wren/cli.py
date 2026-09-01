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

#: `--steps 2000`, `--lr 3e-3`, ... — `train()`'s parameters, which are the
#: flow's inputs bar the corpus that `--train CORPUS` already carries. Named
#: here rather than read off the flow so `wren <prompt>` needs no fluksio; a
#: test pins the two together.
TRAIN_PARAMS = {
    "steps": int,
    "batch_size": int,
    "seq_len": int,
    "lr": float,
    "log_every": int,
}


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
    for name, kind in TRAIN_PARAMS.items():
        # Left unset, so an unmentioned parameter takes the flow's own value
        # and the run records which of the two the number came from.
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            help=f"training: {name} (default: the flow's)",
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
    params |= {
        name: getattr(args, name)
        for name in TRAIN_PARAMS
        if getattr(args, name) is not None
    }

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
    log(f"final loss {handle.result['final_loss']:.3f}; saved {WEIGHTS}")


def follow(client: Any, handle: Any, poll: float = 2.0) -> None:
    """Print the run's loss readings as the engine records them.

    The status is read before the numbers, so the last batch is never the one
    that gets missed. The step a reading carries is the engine's count of
    them, not the training step -- the node's own `step` lines are in the
    run's logs.
    """
    seen = 0
    while True:
        done = handle.refresh().done
        points = client.metrics(handle.id)
        for point in points[seen:]:
            log(f"loss[{point['step']:3d}]  {point['value']:.3f}")
        seen = len(points)
        if done:
            return
        time.sleep(poll)


def log(message: str) -> None:
    """Progress goes to stderr so stdout stays pipeable model output."""
    print(message, file=sys.stderr)
