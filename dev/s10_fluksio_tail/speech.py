"""The speech side of s10, run from parrot's serve venv (Kokoro, faster-whisper).

    CT2_CUDA_ALLOW_FP16=1 ~/venvs/parrot-serve/bin/python dev/s10_fluksio_tail/speech.py real
    CT2_CUDA_ALLOW_FP16=1 ~/venvs/parrot-serve/bin/python dev/s10_fluksio_tail/speech.py noisy

`real` transcribes the 7B panel set (`~/spikes/voice/real`) as `parrot serve`'s
final pass does. `noisy` speaks every command of the training corpus once with
Kokoro, adds room reverb and noise as VS5's synthetic set did, and transcribes
it the same way. Both write to `data/s10/`, the transcripts as fluksio's wren
node hands them to the model: lower case, no sentence marks, digits.
"""

import json
import pathlib
import re
import sys

import numpy as np

REAL = pathlib.Path.home() / "spikes/voice/real"
OUT = pathlib.Path("data/s10")
VOICES = ["af_heart", "am_michael", "bf_emma", "am_adam", "af_bella", "bm_george"]

_UNITS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_NUMBER = re.compile(
    rf"\b(?:({'|'.join(_TENS)})(?:[\s-]({'|'.join(_UNITS[1:10])}))?|({'|'.join(_UNITS)}))\b"
)


def plain(text: str) -> str:
    """A transcript as the node passes it on: "Twenty two." -> "22"."""
    text = re.sub(r"[.,!?;:]+(?=\s|$)", "", text).strip().lower()

    def number(m: re.Match) -> str:
        if m[1]:
            return str(
                10 * (_TENS.index(m[1]) + 2) + (_UNITS.index(m[2]) if m[2] else 0)
            )
        return str(_UNITS.index(m[3]))

    return _NUMBER.sub(number, text)


def whisper():
    from faster_whisper import WhisperModel  # ty: ignore[unresolved-import]

    model = WhisperModel("small.en", device="cuda", compute_type="float16")

    def transcribe(audio: np.ndarray) -> str:
        # parrot serve's final pass (parrot/backends.py).
        segments, _ = model.transcribe(
            audio,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    return transcribe


def real() -> None:
    import soundfile as sf  # ty: ignore[unresolved-import]

    transcribe = whisper()
    rows = [r.split("\t") for r in (REAL / "labels.tsv").read_text().splitlines()[1:]]
    out = []
    for name, *_, text in rows:
        heard = transcribe(sf.read(REAL / name, dtype="float32")[0])
        out.append({"file": name, "said": text, "heard": heard, "text": plain(heard)})
        print(name, repr(heard))
    (OUT / "real.json").write_text(json.dumps(out, indent=1))


def room(rng: np.random.Generator, rt60: float, sr: int = 16000) -> np.ndarray:
    """A small room: the direct path plus exponentially decaying noise (VS5)."""
    t = np.arange(int(sr * rt60)) / sr
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * 0.25
    tail[: int(0.004 * sr)] = 0.0
    tail[0] = 1.0
    return tail


def far_field(rng: np.random.Generator, audio: np.ndarray) -> np.ndarray:
    """Kokoro's 24 kHz speech as a panel across the room hears it (VS5's recipe)."""
    from scipy.signal import fftconvolve, resample_poly

    audio = resample_poly(audio, 2, 3).astype(np.float32)
    wet = fftconvolve(audio, room(rng, float(rng.uniform(0.25, 0.6))))[: len(audio)]
    audio = audio + 0.4 * wet / (np.abs(wet).max() + 1e-9) * np.abs(audio).max()
    audio = np.concatenate(
        [np.zeros(4800, np.float32), audio, np.zeros(9600, np.float32)]
    )
    speech = np.sqrt(np.mean(audio[4800:-9600] ** 2))
    noise = np.cumsum(rng.standard_normal(len(audio)))
    noise -= np.convolve(noise, np.ones(400) / 400, "same")  # brownish, no DC
    snr_db = float(rng.uniform(12, 25))
    noise *= speech / (np.sqrt(np.mean(noise**2)) * 10 ** (snr_db / 20))
    audio = audio + noise
    return (audio / max(1.0, np.abs(audio).max() / 0.9)).astype(np.float32)


def noisy() -> None:
    """Append `{"command", "text"}` lines to noisy.jsonl, resuming where it stopped."""
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    from kokoro import KPipeline  # ty: ignore[unresolved-import]

    from wren.dataset import commands, spoken

    path = OUT / "noisy.jsonl"
    done = (
        {json.loads(line)["command"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    todo = [c for c in commands() if c not in done]
    print(f"{len(done)} done, {len(todo)} to speak")
    rng = np.random.default_rng(len(done))
    pipes = {
        lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", device="cuda")
        for lang in "ab"
    }
    transcribe = whisper()
    with path.open("a") as out:
        for i, command in enumerate(todo):
            voice = str(rng.choice(VOICES))
            speed = float(rng.uniform(0.9, 1.1))
            chunks = pipes[voice[0]](spoken(command), voice=voice, speed=speed)
            audio = np.concatenate([r.audio.numpy() for r in chunks])
            heard = transcribe(far_field(rng, audio))
            out.write(json.dumps({"command": command, "text": plain(heard)}) + "\n")
            if i % 200 == 0:
                out.flush()
                print(i, command, "->", plain(heard), flush=True)


if __name__ == "__main__":
    assert (
        plain("Set the heating to twenty-two degrees.")
        == "set the heating to 22 degrees"
    )
    assert plain("In five minutes, nineteen") == "in 5 minutes 19"
    assert plain("seventeen seven") == "17 7"
    OUT.mkdir(parents=True, exist_ok=True)
    {"real": real, "noisy": noisy}[sys.argv[1]]()
