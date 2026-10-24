# s10 — The fluksio tail

## Question

Which commands that fluksio hands to its language model today can Wren take
over with a wider grammar and house-shaped data, and what does that cost the
commands it already handles?

In fluksio's voice chain Wren answers a direct command in about 0.1 s on a CPU
and passes the rest to an LLM that needs 2-4 s and a GPU. fluksio's node
already handles "all"/"every" by rule and turns number words into digits, so
this study covers what only the model can learn:

| group | example | wanted |
| --- | --- | --- |
| switch | "turn on the plug in the kitchen" | a new `switch` kind, `switches.on/off` |
| pair | "turn off the lights in the kitchen and the bathroom" | a call per area |
| area-less | "set the heating to 22 degrees" | the user's area, else the living room |
| unknown | "turn on the hall light", "switch on the appliances" | no call |
| noise | "lower the office shower" | what the recogniser makes of a command |

## Method

The corpus generator (`wren.dataset.home`) gains the `switch` kind (ids
start with `p`), the pair and unknown builders, area-less commands for every
kind with the user away half the time, and a few words the 7B panel set used
that the s8 corpus lacked ("shutters", "tv", "sprinkler"). A switch id starts
with `p` (plug), since `s` is the shutter's and the letter is a cue for copying
the right id. A pair calls each area in the order said; two calls fit the
96-byte answer. "attic" and "garage door" stay out of training as probes.

Recogniser noise: `speech.py noisy` speaks each of the 3121 distinct commands
(without delay) once with Kokoro (six voices, 0.9-1.1x), puts it in a small
room (RT60 0.25-0.6 s, brown noise at 12-25 dB SNR) and
transcribes it with faster-whisper `small.en` as `parrot serve`'s final pass
does. A training command is replaced by its transcript with probability
`share`; the delay is said after it as "in N minutes", the form people use.

Training keeps s8's configuration, its pruned ternary run: dim 256 × 8
layers × 4 heads, context 384, Muon 3e-3, batch 32, 256k unique samples, seed
0. Arms:

| arm | share | steps |
| --- | --- | --- |
| A | 0 | 16000 |
| B | 0.5 | 16000 |
| B32 | 0.5 | 32000 |

`regression.py` scores packed models through the numpy runtime, as fluksio
runs them. On the 28 commands recorded on the 7B panel (`~/spikes/voice/real`,
as small.en heard them) plus 16 typed probes it calls `act` with a live-like
house: ten lit areas, no presence, the node's own preprocessing and aliases.
`--generated` scores held-out samples per builder and s8's evaluation split
(its generator from HEAD). Outcomes at a threshold: ok (applied and right, or
nothing applied when nothing was wanted), wrong (applied and not what was
asked), abstained (passed to the LLM).

## Findings

- The broader corpus needs twice the budget for copying. At 16000 steps both
  arms copy unseen ids at 90% (A 90.4%, B 89.7%, s8 99.7%) and lose
  5-9 points on s8's own split, mostly area-less commands that blend two ids
  (`lzrhj` + `lddnv` -> `lzrnv`). At 32000 steps (B32) copying is back at
  98.2% and s8's split at s8's 91.0%, consistent with s8's abrupt
  copy transition arriving later when the corpus is wider.
- B32 takes over every s10 group on held-out samples and keeps s8's:

  | set (exact) | s8 | A | B | B32 |
  | --- | --- | --- | --- | --- |
  | s8 split (400) | 91.0% | 86.0% | 82.0% | 91.0% |
  | s10 mix (400) | 68.2% | 86.0% | 82.5% | 91.0% |
  | command (47) | 70.2% | 85.1% | 95.7% | 100% |
  | area-less (33) | 69.7% | 97.0% | 87.9% | 100% |
  | pair (80) | 0% | 95.0% | 87.5% | 100% |
  | unknown (80) | 53.8% | 100% | 98.8% | 100% |
  | missing (79) | 83.5% | 82.3% | 67.1% | 93.7% |
  | builders at 0.93, ok/wrong/abstained | 69/7/24 | 81/0/19 | 76/0/24 | 97/0/3 |
  | s8 split at 0.93 | 83/0/17 | 69/0/30 | 56/0/44 | 84/0/16 |

- On the 7B panel set as small.en heard it, plus the probes (44 commands),
  at fluksio's 0.93: s8 59/2/39% (ok/wrong/abstained), A 57/2/41, B
  61/0/39, B32 77/2/20. B32 handles every pair, area-less and unknown probe;
  its one wrong answer is the sprinkler (known issue below). Said rather than
  heard, "water the garden for 10 minutes" (no durations in the grammar) gives
  two `lights.on` at 0.97, where s8 abstained.
- Synthetic noise is mild. Kokoro speech across a small room changed only 216
  of the 3121 commands under small.en (7%), mostly "roller shutters" and the
  unknown words ("cellar" -> "seller"), none of the panel's house-word errors
  ("patio" -> "page too"). What the noise arm gains on spoken commands (B 82%
  against A 73% handled at 0.8, no wrong ones) comes mostly from the delay said
  as "in N minutes": A reads "in 5 minutes" at 0.66, B at 0.95.
- Confidence moves down with the wider corpus, so the threshold should too.
  B32 makes no wrong call on the generated sets from 0.7 up, and the panel set
  keeps its one (at 1.00) at every threshold. At 0.8 B32 handles 84% of the
  panel set and probes (77% at 0.93), 99% of the builders, 88% of s8's split.
- B32 is stricter about house shape: a garden with water and no light (never
  generated) abstains at 0.45 where s8 watered it. fluksio's
  node always shows a light per area.
- One seed per arm. The panel set informed a few phrasings ("shutters", "tv",
  "sprinkler"), so its gains on those words are optimistic.
- Known issue: "turn on the sprinkler in the garden" gives `watering.stop` at
  1.00. The added phrasings pair "switch on" with start and "turn off" with
  stop, so "turn on" reads as stop. Symmetric phrasings fix it; B32 ships
  without that rerun.
- B32 was the shipped model of Wren 1.1.0: `runtime/weights-qat.npz`, packed
  into `wren/wren.bin`, with a recommended threshold of 0.8. The fp32
  `wren/weights.npz` the JAX CLI loads stays s8's, since no fp32 arm was
  trained.

## Reproduce

The commands below run at tag `v1.1.0`.
Training requires one CUDA GPU (`uv sync --group cuda12`), the transcripts
need parrot's serve venv (`~/venvs/parrot-serve`). `shipped.bin` is s8's pack
(`wren/wren.bin` at v1.0.0):

```sh
CT2_CUDA_ALLOW_FP16=1 ~/venvs/parrot-serve/bin/python dev/s10_fluksio_tail/speech.py real
CT2_CUDA_ALLOW_FP16=1 ~/venvs/parrot-serve/bin/python dev/s10_fluksio_tail/speech.py noisy
uv run python dev/s10_fluksio_tail/study.py   # self-checks
uv run fluksio sweep s10_fluksio_tail --param share=0.0
uv run fluksio sweep s10_fluksio_tail --param share=0.5 --param noisy=$PWD/data/s10/noisy.jsonl --param steps=16000,32000
uv run python -m wren.export weights.npz data/s10/<arm>.bin
uv run python dev/s10_fluksio_tail/regression.py data/s10/shipped.bin data/s10/<arm>.bin
uv run python dev/s10_fluksio_tail/regression.py --generated data/s10/shipped.bin data/s10/<arm>.bin
```

The engine (`uv run fluksio serve --gpus 1`) records runs. On a card shared
with other processes the train step needs about 3 GB free.
