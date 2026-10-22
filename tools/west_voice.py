"""Record the Wild West's announcer, from laserapp/west_words.py.

    ./build_story_audio.sh west            # lines that changed
    ./build_story_audio.sh west --force    # all of them again

Spoken by the same model as the story's Ukrainian voices - StyleTTS2 trained
on Ukrainian speech, by patriotyk (MIT) - which works out where the stress
falls in each word; west_words.STRESS puts it right where it guesses wrong.
The announcer is one of the model's own voices, a man's with a wide swing
in pitch, said in one of two deliveries (west_words.mood): a call, quicker
and with the street's echo, or calm and slower for the tutorial.

Each line is said a few times, and the take kept is the one that fits:
of the takes whose length is believable for the words, the liveliest for a
call and the middle one for a calm line. Only lines whose text has changed
since they were recorded are done again.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from laserapp import west_words as words  # noqa: E402
from story_audio import RATE, finish, styletts, write  # noqa: E402

OUT = ROOT / "laserapp" / "assets" / "west" / "voice"
VOICE = "Роман Куліш"
# Each delivery: its speed, its pitch (1 = as recorded), its echo, which take wins.
MANNERS = {
    "call": (1.05, 0.97, 0.12, max),
    "calm": (0.90, 1.00, 0.05, None),       # None: the middle take
}
TAKES = 3


def liveliness(x: np.ndarray) -> float:
    """Pitch spread (semitones) plus loudness swing (dB/10), voiced frames only."""
    n = int(0.03 * RATE)
    f0, db = [], []
    for i in range(0, len(x) - n, n // 2):
        frame = x[i:i + n] * np.hanning(n)
        e = np.sqrt(np.mean(frame ** 2))
        if e < 0.01:
            continue
        ac = np.correlate(frame, frame, "full")[n - 1:]
        lo, hi = RATE // 400, RATE // 70
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.45 * ac[0]:
            f0.append(RATE / lag)
            db.append(20 * np.log10(e))
    if len(f0) < 5:
        return 0.0
    semis = 12 * np.log2(np.array(f0) / np.median(f0))
    return float(np.std(semis) + np.std(db) / 10)


def believable(x: np.ndarray, text: str) -> bool:
    """A take that is far too short or long for its words has gone wrong."""
    letters = sum(c.isalpha() for c in text)
    return 0.04 <= len(x) / RATE / max(1, letters) <= 0.16


def pick(takes, how):
    good = [t for t in takes if t[2]] or takes
    good.sort(key=lambda t: t[0])
    return how(good, key=lambda t: t[0]) if how else good[len(good) // 2]


def stressed(text: str) -> str:
    """`text` with west_words.STRESS put in: a "+" after the stressed vowel."""
    for word, marked in words.STRESS.items():
        text = re.sub(rf"(?<![\w+]){re.escape(word)}(?![\w+])",
                      lambda m: m.group()[0] + marked[1:], text, flags=re.IGNORECASE)
    return text


def save(manifest: Path, done: dict) -> None:
    manifest.write_text(json.dumps(done, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", required=True, help="the folder of StyleTTS2 voice styles")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "voices.json"
    try:
        done = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        done = {}
    only = {s for s in args.only.split(",") if s}
    todo = [k for k in words.LINES if k in only] if only else \
        [k for k, text in words.LINES.items()
         if args.force or done.get(k) != text or not (OUT / f"{k}.wav").exists()]

    if todo:
        import torch
        speak = styletts(SimpleNamespace(voices=args.voices))
        for i, key in enumerate(todo):
            manner = words.mood(key)
            speed, pitch, echo, how = MANNERS[manner]
            text = words.LINES[key]
            t0 = time.time()
            takes = []
            for seed in range(TAKES):
                torch.manual_seed(7 + seed)
                x = speak(stressed(text), VOICE, speed)
                takes.append((liveliness(x), seed, believable(x, text), x))
            score, seed, ok, x = pick(takes, how)
            x = finish(x, pitch, echo)
            write(OUT, key, x)
            done[key] = text
            save(manifest, done)                 # as it goes: a long run may be stopped
            print(f"  [{i + 1}/{len(todo)}] {key:12} {manner}  take {seed} "
                  f"({score:.2f}{'' if ok else ', length off'})  {len(x) / RATE:4.1f}s  "
                  f"{time.time() - t0:3.0f}s  {text}", flush=True)

    for stale in [k for k in done if k not in words.LINES]:
        (OUT / f"{stale}.wav").unlink(missing_ok=True)
        del done[stale]
    save(manifest, done)
    print(f"west announcer: {len(todo)} recorded, {len(words.LINES)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
