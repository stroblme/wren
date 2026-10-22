# s8b — Copy-pathway spectrum probe

## Question

Which heads carry the retrieval that formed in s8, and which RoPE frequency
bands carry the match signal at the copy edges?

## Method

This eval-only probe uses trained fp32 weights from s8 (`dev/s8_state_quant`).
It teacher-forces evaluation samples that require copying through the stock
forward pass and records each layer's queries, keys, and attention. The main
numbers come from the run with 256k samples and 16000 steps, where 96 of the
default 128 samples passed the span and context filters at the time; the
64k-sample checkpoint serves as comparison.

| instrument | measures |
| --- | --- |
| copy mass | attention per (layer, head) from the positions emitting ID bytes to the ID's span in the state block, compared with a control ID in the same format |
| slot decomposition | exact split of the peak copy-edge logit over RoPE's 32 frequency pairs, summed into s6's bands |

## Findings

- A single matcher head performs the retrieval. L3h3 puts 0.198 of its attention
  on the ID's span and 0.001 on the control, a 200× difference between the
  matching ID and the control. It feeds L6h0 and L6h2 (0.05/0.04 vs
  0.005/0.003). L3h3 already led at the 64k checkpoint (0.147), so the circuit
  strengthened in place.
- The match lies in the near-DC band. L3h3's band sums are DC 94.0, sample
  18.4, word 0.5, and byte −2.7. These slots barely rotate at
  Δ ≈ 100–300 B, so they act as position-independent content directions. The
  sample band (128–1024 B) acts as a coarse look-back gate.
- The copying circuit uses content matching in the near-DC band, which s4
  (`dev/s4_learnable_freqs`) found undertrained and that s5 (`dev/s5_qknorm`)
  measured carrying half of the logit power.
- For spectrum design, keep the near-DC slots: zeroing them, as FoPE does
  (arXiv:2412.17739), should remove the matching head's function, a testable
  prediction.

## Reproduce

Runs on CPU:

```sh
uv run fluksio artifacts 1788349170374-a97fddf7 s8_state_quant.weights -o /tmp/s8b-weights.npz
JAX_PLATFORMS=cpu uv run python dev/s8b_copy_spectrum/probe.py /tmp/s8b-weights.npz --n-train 256000
```

Run ID `1788349170374-a97fddf7` is the original local run; run the
256k/16000-step s8 sweep first and use your own fp32 run ID.

`--n-train` must match the run so evaluation IDs remain unseen. Because the
generator now prunes state blocks, the replayed prompts differ from the probed
ones, and all 128 samples pass the filters today. The engine
(`uv run fluksio serve`) serves the weights; the figure is written to the
gitignored `figures/` directory.
