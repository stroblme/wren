"""Command line entry point.

`wren <prompt>` (or piped stdin) runs inference against the shipped
weights; `wren --train` retrains them from a corpus and overwrites the
weights file inside the package, which is what the wheel ships.
"""

import argparse
import pathlib
import sys

from flax import nnx

from wren import tokenizer
from wren.dataset import home_corpus
from wren.model import WEIGHTS, Config, Wren, generate, load, save
from wren.train import train


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
    args = parser.parse_args()

    if args.train is not None:
        text = pathlib.Path(args.train).read_text() if args.train else home_corpus()
        model = Wren(Config(), rngs=nnx.Rngs(0))
        log(f"wren: {model.n_params():,} params, corpus: {len(text):,} bytes")
        train(model, text)
        save(model, WEIGHTS)
        log(f"saved {WEIGHTS}")
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


def log(message: str) -> None:
    """Progress goes to stderr so stdout stays pipeable model output."""
    print(message, file=sys.stderr)
