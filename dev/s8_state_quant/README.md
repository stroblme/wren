# s8 — State corpus × quantization

## Question

Does state-in-context data force the copying circuit that never formed in s7,
and what does quantization-aware training cost at about 2M parameters?

## Method

Training uses the generated home corpus (`wren.dataset.home`), whose eval houses
use unseen device IDs, so held-out exact match tests copying. Dim 256 × 8
layers, Muon 3e-3, 8000 steps, seed 0. BitNet-style QAT fake-quantizes attention
projections with straight-through gradients. Follow-ups vary the number of
unique samples (`n_train`, 8k default) and budget; a final retrain prunes state
blocks to relevant areas.

| arm | attention weights |
| --- | --- |
| `fp32` | unquantized (control) |
| `int8` | 8-bit, absmax scale |
| `ternary` | {−1, 0, +1}, absmean scale |

## Findings

- State-in-context data alone does not cause copying: with 8k samples (~40
  epochs), fp32 reaches 13.4% exact but copies 0/262 unseen IDs.
- Copying needs both diversity and budget. ID-copy accuracy is 14.7% at 64k/8000
  steps, 0% at 256k/8000, and 99.3% at 256k/16000 (88.6% exact). The abrupt
  change is consistent with an induction-head phase transition
  (arXiv:2209.11895).
- Quantization preserves task performance in these runs. int8 matches fp32 (15.0
  vs 13.4% exact); ternary preserves task metrics despite +0.29 nats LM loss.
  Packed attention weights: 8.66 → 2.37 → 0.80 MB. With pruned blocks, fp32
  reaches 91.0% exact and ternary 89.8%, both 99.7% ID-copy. Each configuration
  uses one seed.
- The pruned fp32 run was `wren/weights.npz` until Wren 2.0. The pruned
  ternary run (`runtime/weights-qat.npz`) packed into `wren/wren.bin`, run by
  the library and `runtime/wren.c` (12.5 MB RSS, 1447 B/s in C). A mobile
  ternary run is the README's QAT row.

## Reproduce

The commands below run at tag `v1.1.0` (and `v1.0.0`).
Training requires one CUDA GPU (`uv sync --group cuda12`):

```sh
uv run python dev/s8_state_quant/study.py   # self-checks
uv run fluksio sweep s8_state_quant --param quant=fp32,int8,ternary
uv run fluksio sweep s8_state_quant --param n_train=64000,256000
uv run fluksio sweep s8_state_quant --param n_train=256000 --param steps=16000 --param quant=fp32,ternary
uv run fluksio sweep s8_state_quant --param corpus=mobile --param quant=ternary
```

The generator now prunes state blocks, so the first two sweeps rerun earlier
points in that format. The engine (`uv run fluksio serve --gpus 1`) records
runs; there is no figure.
