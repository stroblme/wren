# s1 — Baseline

## Question

What loss and tool-call accuracy does the starting architecture
(attention-only, 8 layers, dim 128, AdamW) reach on the home corpus, and with
which step budget and learning rate?

## Method

The `s1_baseline` flow has two nodes. The corpus node builds the synthetic
home corpus of that time: about 200 unique command/tool-call pairs over 10
areas, including indirect phrasings, repeated as 40 shuffled copies. The
training node runs AdamW at lr 3e-3, batch 32, and seq 128 for 1000 steps at
seed 0. It then decodes six seen prompts, one per device domain, at
temperature 0.05; a hit requires the exact tool call (for example,
`it's too dark in the office` → `lights.on('office')`). Two variations were
checked against this setting: 600 steps, and lr 1e-3. The study also set up
the fluksio workflow that every later study uses.

## Findings

- The baseline reaches a final loss of 0.135 and 6/6 tool-call accuracy on
  the seen prompts.
- 600 steps leave accuracy at 5/6, and lr 1e-3 at 1000 steps is equivalent
  (0.127). The library's default training budget was raised to 1000 steps
  at the time.
- The ~0.13 loss floor is corpus entropy: the command halves are
  unpredictable, while the tool-call halves are fully learned. Further tuning
  would not lower it.
- The model memorizes, and unseen phrasings produce noise. Study s2
  (`dev/s2_mobile_actions`) adds a public benchmark with a real held-out
  split.

## Reproduce

```sh
uv run fluksio run s1_baseline --follow
```

Commit `8764546` replaced the template corpus with the state-conditioned
generator, so the flow runs only at `8764546~1` or earlier. Commit `d04dded`
raised the default width to 256, so only checkouts before it keep the
recorded dim 128. The project-local engine (`uv run fluksio serve --gpus 1`)
records the run and its weights artifact; s1 produces no figure.
