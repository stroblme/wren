# s6 — Banded attention heads

## Question

Does giving each attention head its own contiguous band of the RoPE spectrum
improve loss, length generalization, or attention structure at byte level?

## Method

The arms differ only in how the RoPE spectrum is spread across the four heads
of the stock model (dim 256, 8 layers, 32 frequency pairs per head). Band
edges sit at periods 32, 128, and 1024 B; outer edges match the stock
spectrum, so only the allocation changes. Training follows the s4 protocol
(`dev/s4_learnable_freqs`): Home-Assistant-Requests, seq 128, batch 32,
2000 steps, Muon at 3e-3, seeds {0, 1, 2}; 9 runs. Held-out bits/byte is
measured at windows {128, 192, 256}, and per-head attention entropy and
distance profiles at 128 and 256.

| arm | spectrum |
| --- | --- |
| `shared` | stock spectrum in every head, frozen (control) |
| `banded` | one band per head, frozen |
| `banded_learned` | banded; slots with ω < 2π/128 frozen, rest trained |

## Findings

- Banding has no effect on loss. At window 128, banded reaches
  0.332–0.334 bpb and shared 0.328–0.332; at window 256 they tie
  (0.296–0.304 vs 0.296–0.308).
- The heads specialize anyway. At window 256, banded entropy rises from byte
  to long head (0.33 / 0.40 / 0.47 / 0.64), while the shared heads are
  interchangeable (0.41–0.44). The long head's mass at ≥ 32 B rises from 0.14
  to 0.44.
- Learning in-band frequencies still harms length generalization.
  `banded_learned` reaches 0.331–0.332 at window 128 but 0.306–0.314 at 256,
  worse than both fixed arms in every seed: about 0.008 bpb of damage, against
  0.05–0.15 for s4's free drift.
- The fixed shared spectrum stays the default. This closes the
  learnable-spectrum thread: the prior picks the spectrum, the data only the
  coefficients.

## Reproduce

Training requires one CUDA card (`uv sync --group cuda12`):

```sh
uv run python dev/s6_banded_heads/study.py   # self-checks
uv run fluksio sync dev/s6_banded_heads
uv run fluksio sweep s6_banded_heads --param mode=shared,banded,banded_learned --param seed=0,1,2
uv run python dev/s6_banded_heads/plot.py
```

The engine (`uv run fluksio serve --gpus 1`) records runs; the figure goes to
the gitignored `figures/` directory.
