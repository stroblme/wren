# s1 — Baseline

## Question

What loss and tool-call accuracy does the starting architecture (attention-only,
8 layers, dim 128, AdamW) reach on the home corpus, and with what training
budget and learning rate?

## Method

The `s1_baseline` flow has two nodes. The corpus node builds the synthetic home
corpus used at the time: about 200 unique command/tool-call pairs over 10 areas,
including indirect phrasings, repeated as 40 shuffled copies. The training node
runs AdamW at lr 3e-3, batch 32, and seq 128 for 1000 steps at seed 0. It then
decodes six seen prompts, one per device domain, at temperature 0.05; a hit
requires the exact tool call (for example, `it's too dark in the office` →
`lights.on('office')`). Two variations were checked against this setting: 600
steps, and lr 1e-3. The study also set up the fluksio workflow that every later
study uses.

## Findings

- The baseline reaches a final loss of 0.135 and 6/6 tool-call accuracy on
  the seen prompts.
- At 600 steps, accuracy is 5/6. At 1000 steps, lr 1e-3 gives comparable results
  (loss 0.127). The library's default training budget was raised to 1000 steps
  at the time.
- The ~0.13 loss floor reflects corpus entropy: commands are unpredictable,
  while the model has learned their tool calls. Further tuning would not lower
  it.
- The model memorizes the training examples and produces unreliable output for
  unseen phrasings. Study s2 (`dev/s2_mobile_actions`) adds a public benchmark
  with a real held-out split.

## Reproduce

The commands below run at tag `v1.1.0` (and `v1.0.0`).
Training requires one CUDA GPU (`uv sync --group cuda12`):

```sh
uv run fluksio run s1_baseline --defaults --follow
```

The corpus node rebuilds the template corpus byte-identically and the model
shape is pinned to dim 128 and 8 layers, but the model code has changed since,
so the numbers may not reproduce exactly. s1 produces no figure.
