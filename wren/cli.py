"""Command line interface for inference and training.

`wren "turn on the kitchen light"`, or the command on stdin, runs `act` with
a pack on an installation's targets and prints each call with its
confidence. By default it runs the shipped demo model on the demo home it
learnt (`wren/home.json`). `wren --train home.json` runs the Fluksio training
flow on an installation's targets, examples and tests and writes the model's
pack beside it, as `home.bin`.
"""

import argparse
import json
import pathlib
import sys
import time
from typing import Any

from wren import runtime
from wren.targets import Target, act, answer


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="wren", description=__doc__)
    parser.add_argument(
        "command", nargs="*", help="command; read from stdin if omitted"
    )
    parser.add_argument(
        "--targets",
        metavar="JSON",
        type=pathlib.Path,
        default=runtime.DEMO,
        help="installation whose targets the command acts on (default: the demo home)",
    )
    parser.add_argument(
        "--model",
        metavar="BIN",
        type=pathlib.Path,
        default=runtime.MODEL,
        help="pack to run (default: the demo model)",
    )
    parser.add_argument(
        "--train",
        metavar="JSON",
        help="train on an installation's targets, examples and tests; write the "
        "pack as .bin (`wren --train JSON --help` lists the training flags)",
    )
    # `--dim`, `--steps`, ... come from `fit`, whose module needs JAX, so
    # only a training command gets them.
    if any(arg.split("=")[0] == "--train" for arg in argv):
        add_knobs(parser)
    args = parser.parse_args(argv)

    if args.train is not None:
        train_run(args)
        return

    command = " ".join(args.command) or sys.stdin.read().strip()
    if not command:
        parser.error("empty command")
    targets = [Target(**t) for t in json.loads(args.targets.read_text())["targets"]]
    calls = act(command, targets, model=runtime.load_pack(args.model))
    for call in calls:
        print(f"{answer([call]).strip()}  {call.confidence:.2f}")
    if not calls:
        print("not handled")


def add_knobs(parser: argparse.ArgumentParser) -> None:
    """Add one `--flag` per training knob, from the one place its default lives."""
    try:
        from wren.train import DEFAULTS
    except ImportError:
        sys.exit("wren --train needs the train extra: pip install 'wren[train]'")
    # Every value is sent, so the run records the whole configuration rather
    # than only what was said out loud.
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default,
            help=f"training: {name} (default: %(default)s)",
        )


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

    installation = pathlib.Path(args.train)
    params: dict[str, Any] = {"installation": installation.read_text()}
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
    result, meta = handle.result, runtime.load_pack(out)["meta"]
    tests = meta.get("test_exact_match")
    log(
        f"{result['n_params']:,.0f} params, final loss {result['final_loss']:.3f}, "
        f"held-out exact match {result['exact_match']:.1%}"
        + ("" if tests is None else f", tests {tests:.1%}")
        + f", coverage {result['coverage']:.1%} at threshold "
        f"{result['threshold']:.2f}; saved {out}"
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
