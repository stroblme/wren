"""S8b — copy-pathway spectrum probe (eval-only, follows s8).

s8's addendum showed copying switch on with data diversity. This probe
asks the QFM-native question on the trained weights: *which heads* carry
the retrieval, and *which spectral bands* of the RoPE encoding carry the
match signal at the copy edges.

Method: teacher-force each needs-copy eval sample's reference text and
capture, per layer and head, the pre-RoPE (post-QK-norm) queries/keys
and the attention. Two instruments:

- copy mass: attention from the positions writing the id bytes back to
  the id's span in the state block, against a control span (a different
  device's id — same format, wrong content), per (layer, head). The
  contrast isolates content matching from format attention.
- slot decomposition: the pre-softmax logit at the peak copy edge
  decomposes exactly over RoPE's 2D frequency pairs (qfm-relation.md §1):
  c_i = (qa·ka + qb·kb) cos(Δω_i) + (qa·kb − qb·ka) sin(Δω_i).
  The per-slot means over copy vs control edges say which bands do the
  matching. Both identities are asserted against the stock forward on
  the first batch.

Run on a saved weights file (pull one with
`uv run fluksio artifacts <run> s8_state_quant.weights`); pass the run's
n_train so the eval split replays with genuinely unseen ids:

    JAX_PLATFORMS=cpu uv run python dev/s8b_copy_spectrum/probe.py \
        weights.npz --n-train 64000

Writes docs/s8b-copy-spectrum.svg and prints the tables.
"""

import argparse
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from wren import tokenizer
from wren.dataset import home, render_call
from wren.model import Wren, load, rope

ID = re.compile(r"^[lmhsw][a-z2-9]{4}$")
# s6's scale bands over RoPE periods (bytes); edges at {32, 128, 1024}.
BANDS = [("byte", 2, 32), ("word", 32, 128), ("sample", 128, 1024), ("dc", 1024, 1e9)]


def captured_forward(model: Wren, tokens: jnp.ndarray) -> tuple[list[dict], jnp.ndarray]:
    """The stock forward, keeping per layer: pre-RoPE q/k (post-QK-norm)
    and the attention matrix. Returns (records, logits); logits are
    asserted equal to model(tokens) on the first call."""
    _, seq = tokens.shape
    x = model.embed(tokens)
    records = []
    for block in model.blocks:
        h = block.norm(x)
        q, k, v = jnp.split(block.wqkv(h), 3, axis=-1)
        shape = (1, seq, block.n_heads, block.head_dim)
        qh = block.q_norm(q.reshape(shape))
        kh = block.k_norm(k.reshape(shape))
        v = v.reshape(shape)
        qr, kr = rope(qh), rope(kh)
        scores = jnp.einsum("bqhd,bkhd->bhqk", qr, kr) / jnp.sqrt(block.head_dim)
        scores = jnp.where(jnp.tril(jnp.ones((seq, seq), dtype=bool)), scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        records.append({"q": qh, "k": kh, "attn": attn})
        out = jnp.einsum("bhqk,bkhd->bqhd", attn, v).reshape(1, seq, -1)
        x = x + block.wo(out)
    return records, model.embed.attend(model.norm(x))


def slot_contributions(
    qh: jnp.ndarray, kh: jnp.ndarray, q_pos: int, k_pos: int, head: int
) -> np.ndarray:
    """Per-frequency-pair contributions to the (q_pos <- k_pos) logit of
    one head; their sum equals the roped dot product (asserted below)."""
    half = qh.shape[-1] // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(half) / half))
    qa, qb = qh[0, q_pos, head, :half], qh[0, q_pos, head, half:]
    ka, kb = kh[0, k_pos, head, :half], kh[0, k_pos, head, half:]
    angle = (q_pos - k_pos) * freqs
    c = (qa * ka + qb * kb) * jnp.cos(angle) + (qa * kb - qb * ka) * jnp.sin(angle)
    return np.asarray(c)


def spans(text: str, value: str, control: str) -> tuple[range, range, range] | None:
    """Byte spans: the value in the state block (source), the value in the
    call line (emit), and the control id in the state block."""
    if not text.isascii():
        return None
    source = text.find(value)
    emit = text.rfind(value)
    ctrl = text.find(control)
    if source < 0 or ctrl < 0 or emit <= source:
        return None
    return (
        range(source, source + len(value)),
        range(emit, emit + len(value)),
        range(ctrl, ctrl + len(control)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=pathlib.Path)
    parser.add_argument("--n-train", type=int, default=64000)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--out", type=pathlib.Path, default="docs/s8b-copy-spectrum.svg")
    args = parser.parse_args()

    model = load(args.weights)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    half = model.cfg.dim // model.cfg.n_heads // 2
    _, evals = home(n_train=args.n_train)

    probes = []  # (text, value, control)
    for prompt, calls in evals:
        ids = [str(v) for _, a in calls for v in a.values() if ID.match(str(v))]
        others = [
            w for line in prompt.splitlines() for w in line.split() if ID.match(w)
        ]
        control = next((o for o in others if o not in ids), None)
        if not ids or control is None:
            continue
        text = prompt + "\n".join(render_call(n, a) for n, a in calls) + "\n"
        probes.append((text, ids[0], control))
    probes = probes[: args.n_samples]
    assert probes, "no copy-requiring samples found"

    copy_mass = np.zeros((n_layers, n_heads))
    ctrl_mass = np.zeros((n_layers, n_heads))
    copy_slots = np.zeros((n_layers, n_heads, half))
    ctrl_slots = np.zeros((n_layers, n_heads, half))
    n_edges = 0
    for pi, (text, value, control) in enumerate(probes):
        got = spans(text, value, control)
        if got is None:
            continue
        source, emit, ctrl = got
        toks = jnp.array([tokenizer.encode(text)[: model.cfg.context]], dtype=jnp.int32)
        if emit[-1] >= toks.shape[1]:
            continue
        records, logits = captured_forward(model, toks)
        if pi == 0:  # parity with the stock forward
            assert jnp.allclose(logits, model(toks), atol=1e-4)
        queries = [p - 1 for p in emit]  # position p is predicted at p-1
        n_edges += 1
        for li, rec in enumerate(records):
            attn = np.asarray(rec["attn"])[0]  # (heads, T, T)
            copy_mass[li] += attn[:, queries][:, :, list(source)].sum(-1).mean(1)
            ctrl_mass[li] += attn[:, queries][:, :, list(ctrl)].sum(-1).mean(1)
            for h in range(n_heads):
                sub = attn[h][np.ix_(queries, list(source))]
                q_pos, k_pos = queries[sub.max(1).argmax()], source[sub.max(0).argmax()]
                c = slot_contributions(rec["q"], rec["k"], q_pos, k_pos, h)
                if pi == 0 and li == 0 and h == 0:  # decomposition is exact
                    dot = float(
                        rope(rec["q"])[0, q_pos, h] @ rope(rec["k"])[0, k_pos, h]
                    )
                    assert abs(float(c.sum()) - dot) < 1e-3 * max(1, abs(dot))
                copy_slots[li, h] += c
                subc = attn[h][np.ix_(queries, list(ctrl))]
                qc, kc = queries[subc.max(1).argmax()], ctrl[subc.max(0).argmax()]
                ctrl_slots[li, h] += slot_contributions(rec["q"], rec["k"], qc, kc, h)

    copy_mass /= n_edges
    ctrl_mass /= n_edges
    copy_slots /= n_edges
    ctrl_slots /= n_edges

    print(f"{n_edges} probed samples; copy vs control attention mass per head:")
    for li in range(n_layers):
        row = "  ".join(
            f"h{h}: {copy_mass[li, h]:.3f}/{ctrl_mass[li, h]:.3f}"
            for h in range(n_heads)
        )
        print(f"  layer {li}: {row}")

    # The copy heads: largest copy-over-control contrast.
    contrast = copy_mass - ctrl_mass
    top = np.dstack(np.unravel_index(np.argsort(-contrast, axis=None), contrast.shape))[0][:3]
    freqs = 1.0 / (10000.0 ** (np.arange(half) / half))
    periods = 2 * np.pi / freqs
    print("top copy heads and their matching bands (mean slot contribution):")
    for li, h in top:
        shares = copy_slots[li, h] - ctrl_slots[li, h]
        band_sums = {
            name: float(shares[(periods >= lo) & (periods < hi)].sum())
            for name, lo, hi in BANDS
        }
        print(f"  L{li}h{h} contrast {contrast[li, h]:.3f}: {band_sums}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    im = ax1.imshow(contrast, cmap="RdBu_r", vmin=-abs(contrast).max(), vmax=abs(contrast).max())
    ax1.set_xlabel("head"), ax1.set_ylabel("layer")
    ax1.set_title("copy − control attention mass")
    fig.colorbar(im, ax=ax1)
    for li, h in top:
        ax2.plot(periods, copy_slots[li, h] - ctrl_slots[li, h], label=f"L{li}h{h}")
    for _, lo, _hi in BANDS[1:]:
        ax2.axvline(lo, color="gray", lw=0.5, ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("RoPE period (bytes)"), ax2.set_ylabel("slot contribution, copy − control")
    ax2.set_title("match signal by spectral slot (top heads)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
