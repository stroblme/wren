# Reference

The public package exports `Target`, `Call`, `act` and `load_pack`.
Inference needs only NumPy. Import training functions from `wren.train`
after installing the `train` extra.

## Target

`Target(name, type, value, values=None, range=None, description="", unit=None, readonly=False)`

| Field | Meaning |
|---|---|
| `name` | Unique snake_case name. Its words help select the target. |
| `type` | `"bool"`, `"int"`, `"float"` or `"str"`. |
| `value` | Current value, matching the target's type. Supply it on every request. |
| `values` | Allowed string values, case-sensitive. `None` leaves strings unrestricted at parsing time; the generator needs a nonempty list to generate string-setting commands. |
| `range` | Inclusive `[low, high]` for numbers. Without it, parsing allows any value of the numeric type; generated values stay near the listed value. |
| `description` | Words used for selection. The text before its first comma is also a generated name when it has at most five words. |
| `unit` | Metadata for your code; absent from the prompt. |
| `readonly` | Defaults to `False`. When `True`, setting the target rejects the answer; generated commands ask for readings. |

The dataclass stores these fields without validating the installation.
Use valid, distinct names and consistent types, values and constraints.
Descriptions help selection and training; the prompt itself contains only
`name=value` lines and the command.

## Call

`Call(target, value=None, delay_s=None, confidence=1.0)`

| Field | Meaning |
|---|---|
| `target` | Name from the supplied targets. |
| `value` | Value to set, or `None` to request a reading. |
| `delay_s` | Delay in seconds, or `None` to run now. Generated delays use whole seconds. |
| `confidence` | Minimum byte probability on this call's line and the answer's closing blank line. |

Calls describe work for your application. Wren does not execute them.

## act and load_pack

```python
from wren import Target, act, load_pack

targets = [Target("kitchen_light", "bool", False)]
calls = act("turn on the kitchen light", targets)
```

`act(command, targets, *, model=None) -> list[Call]` selects targets, builds
a prompt, decodes greedily and parses the answer. With no `model`, it loads
and caches the packaged demo model. A custom model is the dictionary
returned by `load_pack(path)`.

`act` does not filter by confidence or verify that the supplied target
definitions match the model's training metadata. `[]` can mean abstention
or rejection of an invalid or incomplete answer.

`load_pack(path) -> dict` accepts a string or `pathlib.Path` and loads
configuration (`cfg`), NumPy weight arrays and metadata (`meta`). It expands
the packed attention weights to float32 in memory, so pack size is not the
runtime memory footprint. Packs without metadata have `meta == {}`.

## Prompt and answer limits

The default model has a 384-byte context. `act` reserves 96 bytes for the
answer, leaving 288 bytes for the command, newlines and target lines.
These are UTF-8 bytes, not characters; long names and non-ASCII text use
more of the budget.

Selection uses words from target names and descriptions, lowercased and
split on underscores and punctuation, with common function words removed
and a simple plural `s` removal. Targets sharing the most words with the
command get space first; ties use input order. The selected targets appear
in their original order. Keep commands short: an overlong prompt is
truncated from the front by the runtime.

From `wren.targets`, `select(targets, command, budget=288)` and
`prompt(selected_targets, command)` expose these steps. For example:

```text
canopy=UP
close the awning in 10 minutes
```

Answers contain one call per line and a closing blank line:

```text
>> set(canopy, DOWN) in 10m

```

Readings use `>> read(outdoor_temperature)`. Delays use integer `h`, `m`
or `s` suffixes. An answer consisting only of a blank line means abstention.
An answer that is malformed, incomplete, names a target outside the supplied
list, sets a read-only target or violates a value constraint is rejected
in full. Parsing checks against the complete supplied list, including
targets omitted from the prompt by selection.

## fit

```python
from wren.train import fit

result = fit(targets, out="home.bin", examples=[], tests=[])
```

`out` is required. `examples` and `tests` default to empty sequences and use
the [installation record format](guide.md#add-examples). All options after
`targets` are keyword-only.

| Python option | CLI flag | Default | Meaning |
|---|---|---|---|
| `dim` | `--dim` | `256` | Model width. |
| `n_layers` | `--n-layers` | `4` | Attention-only block count. |
| `n_heads` | `--n-heads` | `4` | Attention heads; `dim / n_heads` must be an even integer. |
| `steps` | `--steps` | `4000` | Optimizer updates. |
| `batch_size` | `--batch-size` | `16` | 384-byte windows per update. |
| `lr` | `--lr` | `0.003` | Initial learning rate, decaying to zero along a cosine. |
| `n_train` | `--n-train` | `200000` | Generated training samples. |
| `n_eval` | `--n-eval` | `1000` | Generated evaluation samples. |
| `precision` | `--precision` | `0.99` | Requested call precision for threshold selection. |
| `seed` | `--seed` | `0` | Model and training-corpus seed; evaluation uses `seed + 1`. |

Keep counts positive and provide enough training text for a full window.
The default installation model has about 1.12M parameters and a 535 kB pack.

Two callbacks are available through Python only:

- `on_log(step, loss)` runs after each training step, with a zero-based step
  and a float loss.
- `paraphrase(command) -> list[str]` supplies alternate training phrasings.
  Preserve meaning and numbers as digits. Results are cached per distinct
  base command during corpus generation, before generated delays are added.
  Evaluation does not use the callback.

### Results

`fit` returns a dictionary:

| Key | Meaning |
|---|---|
| `exact_match` | Fraction of generated evaluation samples with exactly the expected calls before confidence filtering. |
| `threshold` | Lowest confidence at which applied calls reach the precision on the generated evaluation calls and on the installation test calls, each on its own. |
| `coverage` | Fraction of generated evaluation samples with exactly the expected calls after filtering. Includes correct abstentions. |
| `test_exact_match`, `test_coverage` | The same sample metrics on supplied tests; absent when no tests are given. |
| `final_loss` | Last training step's mean next-byte cross-entropy, in nats. |
| `n_params` | Parameter count. |
| `seconds` | End-to-end elapsed time, including corpus generation, evaluation and export. |

Installation metrics compare target, value and delay, ignoring call order
and confidence.

A nonempty test set containing only abstention cases cannot yield a finite
threshold at the default precision: calibration needs correctly predicted
calls. Include positive test cases as well.

Current calibration takes the higher of two independently selected
thresholds without rechecking precision there. A higher cutoff can reduce
precision on either set, so the saved threshold does not guarantee the
requested precision even on calibration data. Evaluate the saved threshold
and inspect the accepted calls for your installation.

The pack's metadata stores `targets` (without current values), `threshold`,
`precision`, `steps`, `date` (UTC date), and `test_exact_match` when tests are
supplied. It does not store the examples, test records, complete training
configuration or optimizer state. Keep the installation JSON and training
settings with the pack. `fit` always starts a new model; it does not resume
from a pack.

### Evaluate an existing pack

Use installation tests to check a saved model without retraining:

```python
from wren.dataset import as_case
from wren.eval import coverage

cases = [as_case(targets, record) for record in home["tests"]]
predicted = [act(command, state, model=model) for command, state, _ in cases]
expected = [calls for _, _, calls in cases]
print(coverage(predicted, expected, model["meta"]["threshold"]))
```

Here `home`, `targets` and `model` are loaded as in the
[How-to](guide.md#run-a-command), and the test list must be nonempty.
`as_case` uses listed values for state keys not specified in the test.

## C runtime

In a checkout, compile with a C compiler and the math library:

```sh
cc -O3 -o wren-rt runtime/wren.c -lm
printf 'kitchen_light=false\nturn on the kitchen light\n' | ./wren-rt wren/wren.bin
# >> set(kitchen_light, true)
```

`./wren-rt PACK [-n BYTES] [PROMPT]` accepts a prompt argument or reads
stdin. The default output budget is 96 bytes. A prompt argument gets a
trailing newline automatically; stdin must include it. For the Python
prompt builder, use `wren.targets.prompt(wren.targets.select(targets,
command), command)`.

The C runtime runs single-threaded and prints raw model output. It does
not select targets, validate calls, provide confidence scores or use pack
metadata. Your integration must handle those steps if needed. The pack
format is version 1; Wren's package version is independent.

For custom low-level training, export a saved Wren `.npz` checkpoint with
`python -m wren.export weights.npz out.bin`. This NumPy-only command adds no
metadata or calibrated threshold; `fit` handles both automatically.
