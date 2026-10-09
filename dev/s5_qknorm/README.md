# s5 — QK-norm as a barren-plateau diagnostic

## Question

Does the claim that QK-norm keeps deep attention-only stacks trainable show
up as a gradient-variance-vs-depth curve, the instrument QML uses to diagnose
barren plateaus?

## Method

`ProbeWren` makes QK-norm a static flag. The 24 runs cover depths
{4, 8, 16, 32} × QK-norm {on, off} × seeds {0, 1, 2} under s4's protocol:
Home-Assistant-Requests, dim 256, seq 128, batch 32, and Muon at lr 3e-3 for
2000 steps, with the same lr at every depth. At init and every 100 steps,
block-matrix gradients are measured on 8 fixed probe batches shared by every
run. Per layer, they split into signal (squared mean) and noise (variance
across batches); their sum is the second moment E[g²] that barren-plateau
plots show. Attention entropy is normalized per query position by
log(pos+1). The pre-softmax logit decomposes exactly over RoPE's 32
frequency pairs, so each pair's logit-power share and gradient
signal/noise are recorded too.

## Findings

- No barren plateau appears. Per-parameter E[g²] at init falls as
  depth^(−1.04), while the sum over layers stays depth-invariant
  (8.5–9.5e-5): depth dilutes a fixed gradient budget without destroying
  it. At init, QK-norm is nearly a no-op.
- There is no trainability cliff and no saturation. Every configuration
  trains, and loss at budget improves with depth (d4 0.24–0.30 → d32
  0.20–0.23), with on and off inside seed spread. QK-norm's one
  seed-separated effect is sharper trained attention (normalized entropy at
  d32: 0.479–0.484 on, 0.538–0.552 off).
- The bands differ in role, not in noise. The under-trained band's
  coefficient gradients have the highest SNR (0.33–0.41, against 0.27–0.32
  in-band). Slots 0–1 (periods 6–8 B) carry ~14% of logit power and slots
  11–13 (149–265 B) ~25%. This met the go criterion for study s6
  (`dev/s6_banded_heads`).
- QK-norm stays in `wren/`; nothing else was folded back. Caveats: lr is
  fixed at 3e-3, Muon is itself a strong conditioner, and 2000 steps may not
  reach entropy collapse.

## Reproduce

```sh
uv run fluksio sync dev/s5_qknorm
uv run fluksio sweep s5_qknorm --param depth=4,8,16,32 --param qk_norm=on,off --param seed=0,1,2
uv run python dev/s5_qknorm/plot.py
```

`uv run python dev/s5_qknorm/study.py` runs the parity and decomposition
self-checks. The project-local engine (`uv run fluksio serve --gpus 1`)
records the runs; the figure is written to the gitignored `figures/`
directory.
