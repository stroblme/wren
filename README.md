# Wren

[![CI](https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg)](https://github.com/stroblme/wren/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Wren is a tiny attention-only byte-level model for grounded tool calls on a CPU.
Given the state of a home and a command, it emits the device calls to make. The
packed model is 0.8 MB and runs in ~12 MB of RAM. Wren is part of the
[Fluksio](https://fluksio.com) automation software.

Technology:
- JAX, Flax and Optax: model and training
- NumPy: the inference library
- C: a dependency-free runtime
- Fluksio: experiment tracking
- uv, ruff, ty and pytest: packaging, code quality and testing

## Architecture

Wren is an attention-only byte-level decoder (a "Simple Attention Network"): standard transformer blocks minus the feed-forward MLPs, with the saved parameters spent on more attention layers.
QK-normalization is what keeps the deep attention-only stack trainable, and RoPE encodes relative positions.
Attention-only models are weak at knowledge stored in weights but strong at context-grounded answers — a deliberate fit, since Wren receives its knowledge via RAG/context at runtime.

![Wren architecture](docs/architecture.svg)

Diagram source: [docs/architecture.d2](docs/architecture.d2), rendered with `d2 docs/architecture.d2 docs/architecture.svg`.

## Layout

```
wren/            the package
├── home.py      library API: a House and a command in, Calls out (numpy only)
├── runtime.py   the packed ternary format and its numpy inference
├── model.py     the attention-only decoder and KV-cached generation (JAX)
├── train.py     next-byte training with Muon, and the training defaults
├── dataset.py   the generated home corpus · mobile-actions · Home-Assistant-Requests
├── eval.py      exact match, argument F1 and companion metrics
├── flow.py      the Fluksio flow behind `wren --train`
├── cli.py       the `wren` command
├── wren.bin     the packed model the library runs
└── weights.npz  fp32 weights for the CLI (git LFS)
runtime/         export.py (npz -> wren.bin) · wren.c · the QAT weights wren.bin is made from
dev/             one folder per study, each with its own README
docs/            the diagram above (architecture.d2 / .svg)
tests/           run with `uv run pytest`
```

## Getting started

Clone with [git LFS](https://git-lfs.com) installed (the CLI's fp32 weights live
there), then run `uv sync`.

**Library.** Only numpy is needed; install with
`pip install git+https://github.com/stroblme/wren`.

```python
from wren import Device, House, act

house = House(
    "19:30",
    outdoor_c=6,
    dark=True,
    user_area="kitchen",
    areas={"kitchen": [Device("light", "light.kitchen_ceiling", "off")]},
)
calls, confidence = act("turn on the lights in the kitchen", house)
# [Call(tool='lights.on', args={'id': 'light.kitchen_ceiling'}, delay_s=None)], 0.96
```

Areas, device kinds and states must come from the trained vocabulary
(`wren.AREAS`, `wren.KINDS`, see `wren/home.py`); device ids can be anything.

**Command line.** The model reads a state block followed by the command, and
writes the calls:

```sh
printf '19:30 out 6C dark\nuser kitchen\nkitchen: light lk3m9 off\nturn on the lights in the kitchen\n' | uv run wren -n 40
# >> lights.on(id='lk3m9')
```

**C runtime.** The same packed model, no Python:

```sh
cc -O3 -march=native -o wren-rt runtime/wren.c -lm
./wren-rt wren/wren.bin "turn on the lights in the kitchen"
uv run python runtime/export.py runtime/weights-qat.npz wren/wren.bin  # re-export
```

**Training.** `wren --train` runs on the project-local Fluksio engine, which
records each run's parameters, loss curve and weights, then writes the weights
to `wren/weights.npz`. Each flag defaults to `wren.train.DEFAULTS`:

```sh
uv run fluksio serve                             # the engine, in its own terminal
uv run wren --train                              # built-in corpus
uv run wren --train corpus.txt --steps 2000 --dim 256 --n-layers 8
```

The shipped weights come from study s8 (`dev/s8_state_quant`), which trains longer
on more data than these defaults. Each study in `dev/` documents its question,
findings and how to reproduce it.

## Benchmark

Reference point (2026-09-01) against [needle 2](https://cactuscompute.com/needle), the smallest published tool-calling model, on [google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions): all 961 eval rows, ordered strict exact match (needle's protocol), greedy decoding. Wren rows are fluksio `s7_bottleneck` runs (corpus=mobile, 8000 steps, muon 3e-3, seed 0; runs `1788289190357-de0baeab` and `1788122526556-cb31a992`), trained on the 8,693-row train split plus ~2% synthetic negatives.

| model | params | weights | peak RAM | CPU decode | exact match | arg F1 |
|---|---|---|---|---|---|---|
| Wren, dim 128 × 36 layers | 2.40M | 9.7 MB fp32 | 988 MB¹ | 8.9 B/s¹ | 34.8% | 44.3% |
| Wren, dim 256 × 8 layers | 2.17M | 8.7 MB fp32 | 877 MB¹ | 25.3 B/s¹ | 30.6% | 40.1% |
| Wren, dim 256 × 8, ternary QAT | 2.17M | 0.80 MB packed³ | 12.5 MB³ | 1447 B/s³ | 34.8% | 45.1% |
| needle 2 | 45M | 14 MB (2-bit) | 28 MB | 500+ tok/s² | 63.7% | — |

¹ Single-stream greedy decode, prompt processing included, through the JAX CPU runtime on x86. RAM is dominated by that runtime, not the ~10 MB of weights: the memory/speed gap against needle is mostly engine (dependency-free C++, 2-bit quantization-aware weights), not model size. At matched parameter count, depth is the CPU latency axis — 8 layers decode ~3× faster than 36.

² Raspberry Pi 5, from the needle 2 page.

³ BitNet-style quantization-aware training (`dev/s8_state_quant`, run `1788349172553-e26a05ad`): attention weights at 2 bits, embedding/norms fp32. RAM and speed measured through `runtime/wren.c` (single-threaded C, x86, steady-state decode; ~120 ms per request including process start and model load). Task metrics match or beat the fp32 arm at the same shape and budget.

Protocol differences that favor needle: tool schemas in context with retrieval, grammar-constrained decoding, and a 115B-token pretraining corpus. Wren holds the 7-function surface in its weights (implicit-tool mode) and decodes freely after task-only training.

## Related Research

- [A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363)
