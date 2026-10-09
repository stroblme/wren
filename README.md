# Wren

[![CI](https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg)](https://github.com/stroblme/wren/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Wren is a small attention-only model that reads bytes and generates tool calls
on a CPU. Given a home's state and a command, it generates the device calls
needed to carry out the command. The packed model is 0.8 MB and runs in ~12 MB
of RAM. Wren is part of the [Fluksio](https://fluksio.com) automation software.

Built with:

- JAX, Flax and Optax: model and training
- NumPy: the inference library
- C: a dependency-free runtime
- Fluksio: experiment tracking
- uv, ruff, ty and pytest: packaging, code quality and testing

## Architecture

Wren is an attention-only byte decoder (a "Simple Attention Network"). It omits
the feed-forward MLPs from standard transformer blocks and uses the saved
parameters for more attention layers. QK normalization supports training at
greater depths, and RoPE encodes relative positions.

Attention-only models perform well when answers depend on the prompt, but are
weaker at recalling facts from their weights. This suits Wren, which receives
the relevant state in its context at runtime.

![Wren architecture](docs/architecture.svg)

Diagram source: [docs/architecture.d2](docs/architecture.d2), rendered with `d2 docs/architecture.d2 docs/architecture.svg`.

## Layout

```
wren/            the package
├── home.py      library API: generate Calls from a House and command (NumPy)
├── runtime.py   packed ternary weights and NumPy inference
├── model.py     the attention-only decoder and KV-cached generation (JAX)
├── train.py     next-byte training with Muon and shared defaults
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

**Library.** The library requires only NumPy. Install it with
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

Use the trained vocabulary for areas, device kinds, and states (`wren.AREAS`,
`wren.KINDS`; see `wren/home.py`). Device IDs can use any format.

**Command line.** The model reads a state block followed by the command, and
writes the calls:

```sh
printf '19:30 out 6C dark\nuser kitchen\nkitchen: light lk3m9 off\nturn on the lights in the kitchen\n' | uv run wren -n 40
# >> lights.on(id='lk3m9')
```

**C runtime.** Run the packed model directly in C:

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

The shipped weights come from study s8 (`dev/s8_state_quant`), which uses more
training steps and data than the defaults. Each study in `dev/` documents its
question, findings and how to reproduce it.

## Benchmark

Results as of 2026-09-01, compared with
[needle 2](https://cactuscompute.com/needle), the smallest published
tool-calling model, on
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions).
Evaluation uses all 961 rows, ordered strict exact match (needle's protocol),
and greedy decoding. Wren rows are fluksio `s7_bottleneck` runs (corpus=mobile,
8000 steps, muon 3e-3, seed 0; runs `1788289190357-de0baeab` and
`1788122526556-cb31a992`), trained on the 8,693-row train split plus ~2%
synthetic negatives.

| model | params | weights | peak RAM | CPU decode | exact match | arg F1 |
|---|---|---|---|---|---|---|
| Wren, dim 128 × 36 layers | 2.40M | 9.7 MB fp32 | 988 MB¹ | 8.9 B/s¹ | 34.8% | 44.3% |
| Wren, dim 256 × 8 layers | 2.17M | 8.7 MB fp32 | 877 MB¹ | 25.3 B/s¹ | 30.6% | 40.1% |
| Wren, dim 256 × 8, ternary QAT | 2.17M | 0.80 MB packed³ | 12.5 MB³ | 1447 B/s³ | 34.8% | 45.1% |
| needle 2 | 45M | 14 MB (2-bit) | 28 MB | 500+ tok/s² | 63.7% | — |

¹ Single-stream greedy decode, prompt processing included, through the JAX CPU
runtime on x86. Most RAM is used by the JAX runtime; the weights use ~10 MB.
Needle's dependency-free C++ engine and 2-bit quantization-aware weights account
for much of the memory and speed difference. At matched parameter counts, 8
layers decode ~3× faster than 36.

² Raspberry Pi 5, from the needle 2 page.

³ BitNet-style quantization-aware training (`dev/s8_state_quant`, run
`1788349172553-e26a05ad`): attention weights at 2 bits, embedding/norms fp32.
RAM and speed measured through `runtime/wren.c` (single-threaded C, x86,
steady-state decode; ~120 ms per request including process start and model
load). Task metrics match or exceed the fp32 results at the same model shape and
training budget.

Protocol differences that favor needle: tool schemas in context with retrieval,
grammar-constrained decoding, and a 115B-token pretraining corpus. Wren learns
the seven tool functions in its weights (implicit-tool mode) and uses
unconstrained decoding after training only on the task data.

## Related Research

- [A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363)
