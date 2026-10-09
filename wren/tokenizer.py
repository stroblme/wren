"""Encode text as UTF-8 byte IDs (0–255); replace invalid UTF-8 when decoding."""

VOCAB_SIZE = 256


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")
