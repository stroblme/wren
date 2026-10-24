# s6b — Narrow attention heads

## Question

Does per-head banding start to pay off once heads are too narrow to cover the
spectrum on their own?

## Method

Study s6 (`dev/s6_banded_heads`) attributed its comparable results to each
32-pair head covering the whole spectrum. Here n_heads rises through {4, 8, 16,
32} at fixed dim 256 (head_dim 64 → 8), otherwise under s6's protocol; 2 arms ×
4 head counts × 3 seeds = 24 runs. The difference between arms at each head
count isolates the effect of spectral resolution.

| arm | per-head spectrum (frozen) |
| --- | --- |
| `shared` | stock spectrum, from 32 down to 4 frequencies per head |
| `banded` | s6's four bands split into n_heads/4 sub-bands; union always 128 frequencies |

## Findings

- At the training window length, spectral resolution does not limit loss. At
  window 128, bpb rises only 0.330 → 0.355 as shared falls from 32 to 4
  frequencies, and banding is slightly worse at every head count (+0.002 at 4,
  +0.007 at 32). These results do not support reducing model width through
  narrow banded heads.
- At window 256 the arms diverge (shared/banded): 0.30/0.30 at 4 heads,
  0.34/0.33 at 8, 1.09/0.46 at 16, and 0.44/1.05 at 32, with differences larger
  than seed variation.
- The cause is the wrap zone: periods in (128, 256] B never complete a cycle in
  training and wrap during evaluation. At 16 heads the shared period-199 slot
  sits in every head, each placing 0.17–0.19 of its attention mass at distances
  of 128–255 B; banding confines it to two heads. At 32 heads, the shared
  spectrum skips the zone (periods jump from 63 to 628 B), while banded
  sub-heads fail.
- Banding can isolate failures to individual heads, while shared spectra fail
  together. It does not improve loss at the training window length. The rule
  carried forward: keep spectral mass out of the wrap zone. Config stays at 4
  heads.

## Reproduce

The commands below run at tag `v1.1.0` (and `v1.0.0`).
Training requires one CUDA GPU (`uv sync --group cuda12`):

```sh
uv run python dev/s6b_head_pressure/study.py   # self-checks
uv run fluksio sweep s6b_head_pressure --param mode=shared,banded --param n_heads=4,8,16,32 --param seed=0,1,2
uv run python dev/s6b_head_pressure/plot.py
```

The engine (`uv run fluksio serve --gpus 1`) records runs; the figure is written
to the gitignored `figures/` directory.
