# s8b — Copy-pathway spectrum probe

## Question

Which heads carry the retrieval that formed in s8, and which RoPE frequency
bands carry the match signal at the copy edges?

## Method

This eval-only probe uses trained fp32 weights from s8
(`dev/s8_state_quant`). It teacher-forces needs-copy eval samples through the
stock forward pass and records each layer's queries, keys, and attention. The
main numbers come from 96 samples on the run with 256k samples and 16000
steps; the 64k-sample checkpoint serves as comparison.

| instrument | measures |
| --- | --- |
| copy mass | attention per (layer, head) from the id-emitting positions to the id's state-block span, compared with a same-format control id |
| slot decomposition | exact split of the peak copy-edge logit over RoPE's 32 frequency pairs, summed into s6's bands |

## Findings

- A single matcher head performs the retrieval. L3h3 puts 0.198 of its
  attention on the id's span and 0.001 on the control, a 200× content-over-
  format contrast. It feeds L6h0 and L6h2 (0.05/0.04 vs 0.005/0.003). L3h3
  already led at the 64k checkpoint (0.147), so the circuit strengthened in
  place.
- The match lies in the near-DC band. L3h3's band sums are DC 94.0, sample
  18.4, word 0.5, and byte −2.7. These slots barely rotate at
  Δ ≈ 100–300 B, so they act as position-independent content directions. The
  sample band (128–1024 B) acts as a coarse look-back gate.
- Induction here is content matching, not phase-aligned retrieval. It uses the
  band that s4 (`dev/s4_learnable_freqs`) found under-trained and that s5
  (`dev/s5_qknorm`) measured carrying half of the logit power.
- For spectrum design, keep the near-DC slots: zeroing them, as FoPE does
  (arXiv:2412.17739), should remove the matcher, a testable prediction.

## Reproduce

Runs on CPU:

```sh
uv run fluksio artifacts 1788349170374-a97fddf7 s8_state_quant.weights -o weights.npz
JAX_PLATFORMS=cpu uv run python dev/s8b_copy_spectrum/probe.py weights.npz --n-train 256000
```

`--n-train` must match the run so the eval ids stay unseen. Because the
generator now prunes state blocks, the replayed prompts differ from the probed
ones. The engine (`uv run fluksio serve`) serves the weights; the figure goes
to the gitignored `figures/` directory.
