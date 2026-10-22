"""Record the Wild West's announcer, from laserapp/west_words.py.

    ./build_story_audio.sh west            # lines that changed
    ./build_story_audio.sh west --force    # all of them again

Spoken by OmniVoice (k2-fsa/OmniVoice, run from a VoiceStudio checkout:
--voicestudio, or $VOICESTUDIO), which copies the *delivery* of the clip it
clones as well as the voice. So the announcer has two references, both in one
voice and kept in tools/voices/: a call, shouted over the street, and a calm
one for the tutorial (west_words.mood). They are made once, from the clip the
voice was chosen by (west_seed.wav): several takes of a line in each manner,
the liveliest kept for the call and the steadiest for the calm. Delete them
to choose again.

Each line is then said a few times, and the take kept is the one that fits:
of the takes whose length is believable for the words, the liveliest for a
call and the middle one for a calm line. Only lines whose text has changed
since they were recorded are done again.

The model's weights are CC-BY-NC: the recordings are for non-commercial use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from laserapp import west_words as words  # noqa: E402
from story_audio import RATE, finish, write  # noqa: E402

OUT = ROOT / "laserapp" / "assets" / "west" / "voice"
REFS = ROOT / "tools" / "voices"
SEED = REFS / "west_seed.wav"               # the voice, as first chosen
SEED_TEXT = "Ватажок виходить на дуель!"
# Each manner: what its reference says, its pace, its echo, and which take wins.
MANNERS = {
    "call": ("Гей, ви там! Банда вже в місті! Усі по домівках, а ти — за мною!",
             1.05, 0.12, max),
    "calm": ("Не поспішай. Подивися уважно, а тоді стріляй. У тебе все вийде.",
             0.95, 0.05, None),             # None: the middle take
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


def save(manifest: Path, done: dict) -> None:
    manifest.write_text(json.dumps(done, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voicestudio", default=os.environ.get(
        "VOICESTUDIO", str(Path.home() / "prj" / "VoiceStudio")))
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
        import soundfile as sf
        import torch
        sys.path.insert(0, args.voicestudio)
        from omnivoice.models.omnivoice import OmniVoice
        torch.set_num_threads(os.cpu_count() or 4)
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu",
                                          dtype=torch.float32)

        def say(text, ref, ref_text, seed, speed=1.0):
            torch.manual_seed(seed)
            audio = model.generate(text=text, language="uk", ref_audio=str(ref),
                                   ref_text=ref_text, speed=speed)
            return audio[0].squeeze(0).cpu().numpy().astype(np.float64)

        # The two manners, made once from the seed clip.
        for manner, (text, _, _, how) in MANNERS.items():
            ref = REFS / f"west_{manner}.wav"
            if ref.exists():
                continue
            takes = []
            for seed in range(4):
                x = say(text, SEED, SEED_TEXT, 100 + seed)
                takes.append((liveliness(x), seed, believable(x, text), x))
            score, seed, _, x = pick(takes, how or min)
            sf.write(str(ref), x, RATE, subtype="PCM_16")
            print(f"reference {manner}: take {seed} of 4, liveliness {score:.2f}", flush=True)

        for i, key in enumerate(todo):
            manner = words.mood(key)
            text, speed, echo, how = MANNERS[manner]
            ref = REFS / f"west_{manner}.wav"
            t0 = time.time()
            takes = []
            for seed in range(TAKES):
                x = say(words.LINES[key], ref, text, 7 + seed, speed)
                takes.append((liveliness(x), seed, believable(x, words.LINES[key]), x))
            score, seed, ok, x = pick(takes, how)
            write(OUT, key, finish(x, 1.0, echo))
            done[key] = words.LINES[key]
            save(manifest, done)                 # as it goes: a long run may be stopped
            print(f"  [{i + 1}/{len(todo)}] {key:12} {manner}  take {seed} "
                  f"({score:.2f}{'' if ok else ', length off'})  {len(x) / RATE:4.1f}s  "
                  f"{time.time() - t0:3.0f}s  {words.LINES[key]}", flush=True)

    for stale in [k for k in done if k not in words.LINES]:
        (OUT / f"{stale}.wav").unlink(missing_ok=True)
        del done[stale]
    save(manifest, done)
    print(f"west announcer: {len(todo)} recorded, {len(words.LINES)} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
