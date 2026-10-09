# s2 — Mobile-actions benchmark

## Question

What does the architecture score on a public function-calling benchmark with
a real held-out split, under needle's strict exact-match protocol?

## Method

The model trains on google/mobile-actions (CC-BY-4.0) in implicit-tool mode:
the seven-function surface lives in the weights, and the prompt is the
timestamp line plus the request. Training uses the 8,693-sample train split,
context 768, seq 512, batch 16, and AdamW at lr 3e-3 for 2000 steps at seed 0.
All 961 eval samples are decoded greedily with a KV cache and scored by
ordered strict exact match, well-formed rate, name accuracy, argument
micro-F1, and bits-per-byte. Later runs add hallucination rate, irrelevance
accuracy on held-out off-domain prompts, ECE, and system metrics.

| run | params | training stream |
| --- | --- | --- |
| first | 0.56M (dim 128) | mobile-actions |
| extended metrics | 0.56M (dim 128) | + ~2% synthetic negatives |
| re-baseline | 2.17M (dim 256) | + ~2% synthetic negatives |

## Findings

- The dim-256 run is the reference for studies s3–s5: 15.1% exact match,
  78.1% well-formed, 71.9% name accuracy, 11.2% arg F1, 0.782 bits/byte,
  0.0% hallucination, 90% irrelevance accuracy, and ECE 0.75, at 509 B/s and
  a peak RSS of 1.4 GB (one seed, lr not re-tuned). needle 2 (45M params,
  tools in context) scores 63.7% under the same protocol.
- Selection works; argument values are the bottleneck. Four times the
  parameters bought name accuracy (+14 pts) and irrelevance (+25 pts), but
  exact match (14.0 → 15.1%) and arg F1 (9.7 → 11.2%) barely moved. Arg F1
  became the sensitive metric for later studies.
- Hallucination is exactly zero: the model fails on values, not vocabulary.
  At dim 128, teaching abstention cost well-formedness (79.0 → 69.7%)
  through false abstentions. ECE stays at 0.74–0.75: severely overconfident.

## Reproduce

```sh
uv run fluksio run s2_mobile_actions --follow
```

The flow's defaults (seed 0, 2000 steps, lr 3e-3) at the current default
width match the dim-256 configuration; the dim-128 runs predate the width
change. The project-local engine (`uv run fluksio serve --gpus 1`) records
the runs and their weights artifacts; s2 produces no figure.
