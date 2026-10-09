# s6b — Head-dim pressure

## Question

Does per-head banding start to pay off once heads are too narrow to cover the
spectrum on their own?

## Method

Study s6 (`dev/s6_banded_heads`) explained its tie by each 32-pair head
covering the whole spectrum. Here n_heads rises through {4, 8, 16, 32} at
fixed dim 256 (head_dim 64 → 8), otherwise under s6's protocol; 2 arms × 4
head counts × 3 seeds = 24 runs. Only the arm gap at each head count isolates
spectral resolution.

| arm | per-head spectrum (frozen) |
| --- | --- |
| `shared` | stock spectrum, from 32 down to 4 frequencies per head |
| `banded` | s6's four bands split into n_heads/4 sub-bands; union always 128 frequencies |

## Findings

- In-window, resolution never binds. At window 128, bpb rises only
  0.330 → 0.355 as shared falls from 32 to 4 frequencies, and banding is
  slightly worse at every head count (+0.002 at 4, +0.007 at 32). Narrow
  banded heads free no parameters.
- At window 256 the arms diverge (shared/banded): 0.30/0.30 at 4 heads,
  0.34/0.33 at 8, 1.09/0.46 at 16, and 0.44/1.05 at 32, all seed-separated.
- The cause is the wrap zone: periods in (128, 256] B never complete a cycle in
  training and wrap at eval. At 16 heads the shared period-199 slot sits in
  every head, each leaking 0.17–0.19 of its mass to 128–255 B; banding
  confines it to two heads. At 32 heads, the shared spectrum skips the zone
  (periods jump from 63 to 628 B), while banded sub-heads fail.
- Banding is a robustness prior, not a loss prior: partitioned spectra fail
  in isolation, shared spectra together. The rule carried forward: keep
  spectral mass out of the wrap zone. Config stays at 4 heads.

## Reproduce

Training requires one CUDA card (`uv sync --group cuda12`):

```sh
uv run python dev/s6b_head_pressure/study.py   # self-checks
uv run fluksio sync dev/s6b_head_pressure
uv run fluksio sweep s6b_head_pressure --param mode=shared,banded --param n_heads=4,8,16,32 --param seed=0,1,2
uv run python dev/s6b_head_pressure/plot.py
```

The engine (`uv run fluksio serve --gpus 1`) records runs; the figure goes to
the gitignored `figures/` directory.
