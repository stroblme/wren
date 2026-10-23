<p align="center">
  <img src="docs/wren-logo.svg" width="160" alt="Wren logo">
</p>

<p align="center">
  <a href="https://github.com/stroblme/wren/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/stroblme/wren/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status"></a>
  <a href="https://stroblme.github.io/wren/"><img src="https://img.shields.io/badge/docs-github.io-blue" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

Wren is a small attention-only model that reads bytes and generates tool calls
on a CPU. Given a home's state and a command, it generates the device calls
needed to carry out the command. The packed model is 0.8 MB and runs in ~12 MB
of RAM. Wren is part of the [Fluksio](https://fluksio.com) automation software.

![Wren architecture](docs/architecture.svg)

## Results

Compared with [needle 2](https://cactuscompute.com/needle), the smallest
published tool-calling model, on
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
(all 961 eval rows, ordered strict exact match as in needle's protocol, greedy
decoding). Wren trains on the 8,693-row train split plus ~2% synthetic
negatives for 8000 steps (`dev/s7_bottleneck`, `dev/s8_state_quant`).

| model | params | weights | peak RAM | CPU decode | exact match | arg F1 |
|---|---|---|---|---|---|---|
| Wren, dim 128 × 36 layers | 2.40M | 9.7 MB fp32 | 988 MB¹ | 8.9 B/s¹ | 34.8% | 44.3% |
| Wren, dim 256 × 8 layers | 2.17M | 8.7 MB fp32 | 877 MB¹ | 25.3 B/s¹ | 30.6% | 40.1% |
| Wren, dim 256 × 8, ternary QAT | 2.17M | 0.80 MB packed | 12.5 MB² | 1447 B/s² | 34.8% | 45.1% |
| needle 2 | 45M | 14 MB (2-bit) | 28 MB | 500+ tok/s³ | 63.7% | — |

¹ JAX CPU runtime on x86, prompt processing included; most of the RAM is the
JAX runtime. ² `runtime/wren.c`, single-threaded on x86. ³ Raspberry Pi 5, from
the needle 2 page.

Needle has tool schemas in context, grammar-constrained decoding and a
115B-token pretraining corpus. Wren learns the seven tool functions in its
weights and decodes unconstrained after training only on the task data.

Each study in `dev/` documents its question, findings and how to reproduce it:

| study | question |
|---|---|
| [s1](dev/s1_baseline) | baseline loss and accuracy of the starting architecture |
| [s2](dev/s2_mobile_actions) | score on the public mobile-actions benchmark |
| [s3](dev/s3_optimizers) | AdamW vs. Muon vs. K-FAC at a fixed budget |
| [s4](dev/s4_learnable_freqs) | learnable RoPE frequencies |
| [s5](dev/s5_qknorm) | QK-norm and gradient variance across depth |
| [s6](dev/s6_banded_heads), [s6b](dev/s6b_head_pressure) | per-head RoPE frequency bands |
| [s7](dev/s7_bottleneck) | which errors limit exact match |
| [s8](dev/s8_state_quant), [s8b](dev/s8b_copy_spectrum) | state-in-context data, ternary QAT, the copy circuit |
| [s10](dev/s10_fluksio_tail) | switches, two areas, area-less and unknown commands, recogniser noise |

The packed model (`wren/wren.bin`) comes from s10, the fp32 weights for the
JAX command line (`wren/weights.npz`) from s8.

## Setup

Clone with [git LFS](https://git-lfs.com) installed (the fp32 weights live
there), then run `uv sync`. Training needs a CUDA GPU (`uv sync --group cuda12`).

The library needs only NumPy:
`pip install git+https://github.com/stroblme/wren`.

## Usage

**Library.**

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
# [Call(tool='lights.on', args={'id': 'light.kitchen_ceiling'}, delay_s=None)], 0.97
```

Areas, device kinds and states use the trained vocabulary (`wren.AREAS`,
`wren.KINDS`). Device IDs can use any format.

**Command line.** The model reads a state block followed by the command:

```sh
printf '19:30 out 6C dark\nuser kitchen\nkitchen: light lk3m9 off\nturn on the lights in the kitchen\n' | uv run wren -n 40
# >> lights.on(id='lk3m9')
```

**C runtime.**

```sh
cc -O3 -march=native -o wren-rt runtime/wren.c -lm
./wren-rt wren/wren.bin "turn on the lights in the kitchen"
```

**Training.** Runs are tracked by a local Fluksio engine:

```sh
uv run fluksio serve    # in its own terminal
uv run wren --train     # writes wren/weights.npz
```

The library and C runtime run the packed `wren/wren.bin`, exported from
ternary QAT weights with
`uv run python runtime/export.py runtime/weights-qat.npz wren/wren.bin`.

## Data

Studies download their benchmarks into `data/`:
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
(CC-BY-4.0) and
[acon96/Home-Assistant-Requests](https://huggingface.co/datasets/acon96/Home-Assistant-Requests)
(MIT). The shipped model is trained on the generated home corpus in
`wren/dataset.py`.

## Related research

- [A Controlled Study of Attention-Only Transformers](https://arxiv.org/abs/2607.18363)

## License

MIT
