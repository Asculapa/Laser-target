"""Record the story game's voices and make its sound effects.

Run through build_story_audio.sh, which sets up what this needs. The lines
come from laserapp/story_script.py, one set per language, and are spoken by
text-to-speech models that run on this machine:

* English - Kokoro (model and voices Apache-2.0)
* Ukrainian - StyleTTS2 trained on Ukrainian voices, by patriotyk (MIT),
  which also works out where the stress falls in each word

Both licences allow the recordings to be shipped with the app. The effects
are plain synthesis, a few sines and some noise.

Only lines whose text has changed since they were recorded are done again;
--force redoes everything, --only picks lines by id.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from unicodedata import normalize

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from laserapp import story_script as script  # noqa: E402  (no OpenCV needed)

OUT = ROOT / "laserapp" / "assets" / "story"
RATE = 24000

# Per language and speaker: voice, speaking speed, pitch (1 = as recorded),
# echo. Pitch is changed by playing the recording faster or slower, so a
# raised voice is also told to speak more slowly, and a lowered one faster.
CAST = {
    "en": {
        script.NARRATOR: ("bm_george", 0.92, 1.00, 0.0),
        script.MAREN: ("bf_emma", 0.84, 0.95, 0.0),
        script.PIP: ("af_heart", 0.86, 1.20, 0.0),
        script.GLOAM: ("am_onyx", 0.98, 0.76, 0.4),
    },
    "uk": {
        # Voices of the model's own set, picked by pitch and pace: a calm low
        # woman for the old keeper, a boy for the spark, the deepest and
        # flattest man there is for the fog.
        script.NARRATOR: ("Кирило Татарченко", 0.95, 1.00, 0.0),
        script.MAREN: ("Марта Мольфар", 1.1, 0.97, 0.0),
        script.PIP: ("Поліна Еккерт(хлопчик)", 1.15, 1.08, 0.0),
        script.GLOAM: ("Юрій Вихованець", 1.05, 0.86, 0.4),
    },
}
KOKORO_LANG = {"bm": "en-gb", "bf": "en-gb", "am": "en-us", "af": "en-us"}


def write(folder: Path, name: str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(folder / f"{name}.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())


# -- voices -----------------------------------------------------------------
def resample(x: np.ndarray, factor: float) -> np.ndarray:
    """Play `x` `factor` times faster: higher and shorter, or lower and longer."""
    n = int(len(x) / factor)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def trim(x: np.ndarray, pad: float = 0.06) -> np.ndarray:
    loud = np.flatnonzero(np.abs(x) > 0.01 * np.abs(x).max())
    if not len(loud):
        return x
    edge = int(pad * RATE)
    return x[max(0, loud[0] - edge):loud[-1] + edge]


def finish(x: np.ndarray, pitch: float, echo: float) -> np.ndarray:
    """What every voice gets after the model: pitch, echo, an even level."""
    x = trim(resample(x, pitch) if pitch != 1.0 else x)
    if echo:
        delay = int(0.11 * RATE)
        wet = np.zeros(len(x) + 3 * delay)
        for k in range(3):
            wet[k * delay:k * delay + len(x)] += x * echo ** k
        x = wet
    return x * (0.89 / max(1e-6, np.abs(x).max()))


def kokoro(args):
    """English. Returns the function that speaks a line."""
    from kokoro_onnx import Kokoro
    tts = Kokoro(args.model, args.voices)

    def speak(text: str, voice: str, speed: float) -> np.ndarray:
        x, rate = tts.create(text, voice=voice, speed=speed, lang=KOKORO_LANG[voice[:2]])
        x = np.asarray(x, np.float64)
        return x if rate == RATE else resample(x, rate / RATE)
    return speak


def styletts(args):
    """Ukrainian. A "+" after a vowel in the text puts the stress there."""
    import torch
    from ipa_uk import ipa
    from styletts2_inference.models import StyleTTS2
    from ukrainian_word_stress import Stressifier, StressSymbol
    stressify = Stressifier()
    model = StyleTTS2(hf_path="patriotyk/styletts2_ukrainian_multispeaker", device="cpu")
    styles = {}
    gap = np.zeros(int(0.22 * RATE))

    def speak(text: str, voice: str, speed: float) -> np.ndarray:
        if voice not in styles:
            styles[voice] = torch.load(Path(args.voices) / f"{voice}.pt")
        parts = []
        # A sentence at a time, as the model was trained.
        for sentence in re.split(r"(?<=[.?!:])\s+", text):
            sentence = sentence.replace("+", StressSymbol.CombiningAcuteAccent)
            sentence = normalize("NFKC", sentence).replace("'", "ʼ")
            sentence = re.sub(r" - ", ": ", sentence).replace("...", ".")
            phonemes = ipa(stressify(sentence))
            if not phonemes:
                continue
            tokens = model.tokenizer.encode(phonemes)
            wav = model(tokens, speed=speed, s_prev=styles[voice]).cpu().numpy()
            parts += [trim(np.asarray(wav, np.float64), pad=0.02), gap]
        return np.concatenate(parts[:-1])
    return speak


ENGINES = {"en": kokoro, "uk": styletts}


def stressed(language: script.Language, text: str) -> str:
    """`text` with the language's stress marks put in, for the speech model."""
    for word in sorted(language.stress, key=len, reverse=True):
        marked = language.stress[word]
        text = re.sub(rf"(?<![\w+]){re.escape(word)}(?![\w+])",
                      lambda m: m.group()[0] + marked[1:], text, flags=re.IGNORECASE)
    return text


def record(language: script.Language, args) -> None:
    folder = OUT / language.code
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "voices.json"
    try:
        done = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        done = {}
    only = {s for s in args.only.split(",") if s}
    lines = language.all_lines()
    todo = [l for l in lines if l.id in only] if only else \
        [l for l in lines if args.force or done.get(l.id) != l.text
         or not (folder / f"{l.id}.wav").exists()]
    if todo:
        speak = ENGINES[language.code](args)
        for i, line in enumerate(todo):
            voice, speed, pitch, echo = CAST[language.code][line.who]
            x = finish(speak(stressed(language, line.text), voice, speed), pitch, echo)
            write(folder, line.id, x)
            done[line.id] = line.text
            print(f"  [{i + 1}/{len(todo)}] {line.id} {line.who:9} {len(x) / RATE:4.1f}s")
    # Recordings of lines that are no longer in the story go.
    ids = {l.id for l in lines}
    for stale in [k for k in done if k not in ids]:
        (folder / f"{stale}.wav").unlink(missing_ok=True)
        del done[stale]
    manifest_path.write_text(
        json.dumps(done, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    size = sum(f.stat().st_size for f in folder.glob("*.wav"))
    print(f"{language.name}: {len(todo)} recorded, {len(lines)} lines, {size / 1e6:.1f} MB")


# -- effects ----------------------------------------------------------------
def tone(freq, seconds: float, decay: float = 6.0, overtone: float = 0.0) -> np.ndarray:
    """A sine (or a glide, if `freq` is a pair) that dies away."""
    t = np.arange(int(seconds * RATE)) / RATE
    f = np.linspace(freq[0], freq[1], len(t)) if isinstance(freq, tuple) else \
        np.full(len(t), float(freq))
    phase = 2 * np.pi * np.cumsum(f) / RATE
    x = np.sin(phase) + overtone * np.sin(2.01 * phase)
    attack = np.minimum(1.0, t / 0.004)
    return x * attack * np.exp(-decay * t / seconds)


def mix(parts) -> np.ndarray:
    """Sounds laid at (start time, samples) over each other."""
    n = max(int(at * RATE) + len(x) for at, x in parts)
    out = np.zeros(n)
    for at, x in parts:
        i = int(at * RATE)
        out[i:i + len(x)] += x
    return out


def effects() -> None:
    rng = np.random.default_rng(1)
    rumble = np.cumsum(rng.standard_normal(int(1.1 * RATE)))
    rumble -= np.convolve(rumble, np.ones(1500) / 1500, mode="same")    # no drift
    rumble *= np.exp(-4.0 * np.arange(len(rumble)) / len(rumble))
    made = {
        "hit": (0.35, mix([(0, tone(880, 0.09)), (0.05, tone(1320, 0.12))])),
        "gold": (0.35, mix([(i * 0.06, tone(f, 0.22, overtone=0.3))
                            for i, f in enumerate((1047, 1319, 1568, 2093))])),
        "bad": (0.4, tone((190, 110), 0.28, decay=2.5, overtone=0.6)),
        "lost": (0.4, tone((520, 150), 0.5, decay=3.0)),
        "star": (0.35, tone(1175, 0.7, decay=5.0, overtone=0.35)),
        "win": (0.4, mix([(i * 0.11, tone(f, 0.7, decay=5.0, overtone=0.3))
                          for i, f in enumerate((523, 659, 784, 1047, 1319))])),
        "boom": (0.6, mix([(0, rumble / np.abs(rumble).max()),
                           (0, tone((120, 45), 0.9, decay=4.0))])),
        "tick": (0.25, tone(660, 0.05)),
    }
    for name, (level, x) in made.items():
        write(OUT, name, x * (level / np.abs(x).max()))
    print(f"effects: {', '.join(made)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--lang", choices=sorted(ENGINES), help="the language to record")
    p.add_argument("--model", default="kokoro-v1.0.onnx", help="en: the Kokoro model")
    p.add_argument("--voices", default="voices-v1.0.bin",
                   help="en: Kokoro's voices file; uk: the folder of voice styles")
    p.add_argument("--force", action="store_true", help="record every line again")
    p.add_argument("--only", default="", help="comma-separated line ids to record")
    p.add_argument("--stress", action="store_true",
                   help="uk: record nothing, print each line as it will be stressed")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stress:
        from ukrainian_word_stress import Stressifier
        mark = Stressifier()
        language = script.LANGUAGES["uk"]
        for line in language.all_lines():
            print(line.id, mark(stressed(language, line.text).replace("+", "\u0301")))
    elif args.lang:
        record(script.LANGUAGES[args.lang], args)
    else:
        effects()
    return 0


if __name__ == "__main__":
    sys.exit(main())
