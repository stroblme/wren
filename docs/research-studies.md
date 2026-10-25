# Research and benchmarks

This page collects Wren's architecture, measured results and research
studies. For installation and use, see the [How-to](guide.md).

## Architecture

Wren is an attention-only byte-level decoder: transformer blocks without the
feed-forward layers, with the parameter budget spent on attention instead.
The prompt supplies target names and current values; the model generates
`set` and `read` calls. Attention projections use ternary weights, packed
for storage; embeddings and normalization scales remain float32. NumPy and
C runtimes load the same pack.

![Wren architecture](architecture.svg)

## Demo installation

The shipped demo home (`wren/home.json`: 20 targets, 50 examples, 18 tests)
was trained by `fit` at its defaults: 256 wide, 4 layers, 4000 steps with
batches of 16.

| Measurement | Value |
|---|---|
| Parameters | 1.12M |
| Pack size | 535 kB |
| Median request latency | 52 ms, one BLAS thread on an idle CPU |
| End-to-end training | 41 minutes on an idle 20-core CPU |
| Saved confidence threshold | About 0.973 |

| Evaluation set | Exact match | Coverage at saved threshold |
|---|---|---|
| 1000 generated samples | 92.7% | 69.8% |
| 18 installation tests | 8 of 18 | 5 of 18 |

Generated evaluation samples use another seed and the same templates and
examples as training; command wording can overlap. Exact match compares
target, value and delay, ignoring call order and confidence. Coverage
counts samples handled exactly right after applying only calls at or above
the threshold, including correct abstentions. Two of the five covered tests
are abstentions. These measurements describe this pack, not expected
accuracy for a new installation.

The threshold sits just above the most confident wrong answer to a test:
"kill all the lights" switches them all on, at 0.96. On generated commands
alone the threshold would be 0.74.

The demo knows only its 20 targets, by their names and descriptions, and
English commands in the generator's style. Words the generator does not
use are where it fails: "kill" is not a word it knows for off, "half an
hour" is not a delay it knows (it opens the awning now, at 0.68), and "good
night" switches the bedroom light off (at 0.51). It also miscopies the number
in about a third of the delays ("in 87 minutes" as 88), always below the
threshold: at the threshold it handles 2 of the 97 delayed commands. Of a
duration ("off for 7 minutes") it often applies the call for now but not the
one that ends it.

The [s12 study](https://github.com/stroblme/wren/tree/main/dev/s12_cpu_fit)
compares the shapes and step budgets behind the training defaults.

## Historical mobile-actions benchmark

Comparison with [needle 2](https://cactuscompute.com/needle) on
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions):
all 961 eval rows, ordered strict exact match as in needle's protocol, greedy
decoding. Wren trains on the 8,693-row train split plus about 2% synthetic
negatives for 8000 steps in
[s7](https://github.com/stroblme/wren/tree/main/dev/s7_bottleneck) and
[s8](https://github.com/stroblme/wren/tree/main/dev/s8_state_quant).
These studies use historical models and APIs rather than the current demo.

| Model | Params | Weights | Peak RAM | CPU decode | Exact match | Arg F1 |
|---|---|---|---|---|---|---|
| Wren, dim 128 × 36 layers | 2.40M | 9.7 MB fp32 | 988 MB¹ | 8.9 B/s¹ | 34.8% | 44.3% |
| Wren, dim 256 × 8 layers | 2.17M | 8.7 MB fp32 | 877 MB¹ | 25.3 B/s¹ | 30.6% | 40.1% |
| Wren, dim 256 × 8, ternary QAT | 2.17M | 0.80 MB packed | 12.5 MB² | 1447 B/s² | 34.8% | 45.1% |
| needle 2 | 45M | 14 MB (2-bit) | 28 MB | 500+ tok/s³ | 63.7% | — |

¹ JAX CPU runtime on x86, prompt processing included; most of the RAM is the
JAX runtime. ² `runtime/wren.c`, single-threaded on x86. ³ Raspberry Pi 5, from
the needle 2 page. These throughput measurements use different hardware and
units.

Needle has tool schemas in context, grammar-constrained decoding and a
115B-token pretraining corpus. Wren learns the seven tool functions in its
weights and decodes unconstrained after training only on the task data.

## Study index

Each study in
[`dev/`](https://github.com/stroblme/wren/tree/main/dev) documents its
question, findings and reproduction commands.

| Study | Question |
|---|---|
| [s1](https://github.com/stroblme/wren/tree/main/dev/s1_baseline) | baseline loss and accuracy of the starting architecture |
| [s2](https://github.com/stroblme/wren/tree/main/dev/s2_mobile_actions) | score on the public mobile-actions benchmark |
| [s3](https://github.com/stroblme/wren/tree/main/dev/s3_optimizers) | AdamW vs. Muon vs. K-FAC at a fixed budget |
| [s4](https://github.com/stroblme/wren/tree/main/dev/s4_learnable_freqs) | learnable RoPE frequencies |
| [s5](https://github.com/stroblme/wren/tree/main/dev/s5_qknorm) | QK-norm and gradient variance across depth |
| [s6](https://github.com/stroblme/wren/tree/main/dev/s6_banded_heads), [s6b](https://github.com/stroblme/wren/tree/main/dev/s6b_head_pressure) | per-head RoPE frequency bands |
| [s7](https://github.com/stroblme/wren/tree/main/dev/s7_bottleneck) | which errors limit exact match |
| [s8](https://github.com/stroblme/wren/tree/main/dev/s8_state_quant), [s8b](https://github.com/stroblme/wren/tree/main/dev/s8b_copy_spectrum) | state-in-context data, ternary QAT, the copy circuit |
| [s10](https://github.com/stroblme/wren/tree/main/dev/s10_fluksio_tail) | switches, two areas, area-less and unknown commands, recogniser noise |
| [s12](https://github.com/stroblme/wren/tree/main/dev/s12_cpu_fit) | steps and shape for training an installation on a CPU |

## Reproduce studies

Study s12 runs with the current target API; its published measurements used
the older demo data at tag `v1.1.0`. Its constant learning rate also differs
from `fit`'s current cosine decay.

Studies s1–s10 document historical models and data. Their READMEs target
`v1.1.0`, which retains the home API removed in Wren 2.0. Use a separate
checkout for those commands:

```sh
git clone --branch v1.1.0 https://github.com/stroblme/wren.git wren-v1
cd wren-v1
uv sync --locked --group cuda12
uv run --group cuda12 fluksio serve --gpus 1
```

Training those studies requires a compatible NVIDIA GPU and driver.
In a second terminal at the study's checkout, follow its README. Use Fluksio
to track experiments; data downloads go into ignored `data/`, and plots
into ignored `dev/*/figures/`.

## Data

Studies download their benchmarks into `data/`:
[google/mobile-actions](https://huggingface.co/datasets/google/mobile-actions)
(CC-BY-4.0) and
[acon96/Home-Assistant-Requests](https://huggingface.co/datasets/acon96/Home-Assistant-Requests)
(MIT). An installation's model trains on commands generated from its own
target list (`wren/dataset.py`).
