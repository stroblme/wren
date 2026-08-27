"""Byte-level tokenizer: every token is one UTF-8 byte, vocab size 256.

There is no vocabulary to build and no out-of-vocabulary token: any text
(or binary) round-trips exactly. The trade-off is that sequences are ~4x
longer than with a learned BPE vocabulary, which matters little at Wren's
small context sizes. Needle makes the same trade for the same reason.
"""

VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")
