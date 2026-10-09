# s4 — Learnable RoPE frequencies

## Question

Does learning RoPE's position-encoding spectrum help at byte level, and does the
loss allow a Rotosolve-style analytic coordinate step once parameters appear in
sinusoidal terms?

## Method

`SpectralWren` shares one log-parameterized spectrum (ω = exp θ) across
layers and heads, trained by its own Adam at `freq_lr`; everything else uses
Muon at lr 3e-3. All 11 runs use Home-Assistant-Requests, dim 256, 2000
steps, batch 32, and train seq 128, with held-out bits/byte at windows 128,
192, and 256. After each run, a probe sweeps slot 8 (stock ω = 0.1) over one
full period with the rest frozen.

| arm | spectrum | runs |
| --- | --- | --- |
| fixed | stock, frozen | seeds 0–2 |
| learned | trainable, stock init | `freq_lr` {1e-3, 3e-3, 1e-2} at seed 0; 3e-3 at seeds 1–2 |
| fope | ω < 2π/128 zeroed, frozen | seeds 0–2 |

`redundancy_probe.py` evaluates saved weights. On the fixed and learned@3e-3
seed-0 weights, it evaluates held-out loss along slot 8 (in-band) and slot 18
(undertrained) over ω ∈ [1e-3, π), at windows 128 and 256.

## Findings

- Learned spectra give comparable results at the training length but worsen
  length generalization. Bits/byte at window 128 is 0.329–0.331 learned against
  0.328–0.332 fixed; at window 256 it is 0.35–0.45 against 0.296–0.307. FoPE
  zeroing neither helps nor hurts (0.307–0.329).
- The spectrum migrates consistently, but only within the undertrained band;
  nothing moves into byte-scale periods.
- The slot-8 slice is smooth (4–8 harmonics carry ~99% of its variation, and a
  single sinusoid ~93%): Rotosolve's premise fails, but a coordinate jump from
  9–17 loss evaluations is viable. The slice minimum sits at the current value
  in every arm, indicating that frequency locking comes from co-adaptation.
- Redundancy probe: below 2π/128, frequencies are interchangeable at the
  training window (spans 0.0006–0.005 nats), while the window-256 loss varies
  6–22× more. Directions that are redundant in-distribution are the ones that
  break length generalization. The fixed spectrum remains Wren's default.

## Reproduce

```sh
uv run fluksio sync dev/s4_learnable_freqs
uv run fluksio sweep s4_learnable_freqs --param mode=fixed --param seed=0,1,2
uv run fluksio sweep s4_learnable_freqs --param mode=learned --param freq_lr=1e-3,3e-3,1e-2 --param seed=0
uv run fluksio sweep s4_learnable_freqs --param mode=learned --param freq_lr=3e-3 --param seed=1,2
uv run fluksio sweep s4_learnable_freqs --param mode=fope --param seed=0,1,2
uv run python dev/s4_learnable_freqs/plot.py
uv run python dev/s4_learnable_freqs/redundancy_probe.py
```

The local Fluksio engine (`uv run fluksio serve --gpus 1`) records the runs;
both scripts write their figures to the gitignored `figures/` directory.
