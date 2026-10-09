"""Make the jar shoot's music and sound effects.

    .venv/bin/python tools/jars_audio.py

Plain synthesis, numpy only, on the instruments of tools/duel_audio.py. The
music is for the jar range and lasts a round, countdown included: a bouncing
two-step in G - bass on the beat, a piano chord off it - that a whistled
tune joins after sixteen seconds, and that goes up a tone for the last
twelve. The quick draw has no music; the wait is the point of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from duel_audio import PLUCK, SAW, SOFT, clock, hat, kick, pitch, rng, snare, voice
from story_audio import RATE, mix, tone, write

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from laserapp.jars import JarRange  # noqa: E402

OUT = ROOT / "laserapp" / "assets" / "jars"

BEAT = 0.5                          # 120 to the minute
BAR = 4 * BEAT
INTRO = JarRange.COUNTDOWN          # six beats
BARS = int(JarRange.DURATION / BAR)
LIFT = BARS - 6                     # the bar where everything goes up a tone
TAIL = 3.0

G = 49.0                            # the key: G major
CHORDS = (0, 5, 0, 7)               # G, C, G, D - in semitones from G
# The tune, one row a bar: (eighth it starts on, eighths it lasts, semitones
# from the G above middle C).
TUNE = (
    ((0, 1, 0), (1, 1, 2), (2, 2, 4), (4, 2, 7), (6, 2, 4)),
    ((0, 2, 9), (2, 1, 7), (3, 1, 9), (4, 4, 12)),
    ((0, 1, 7), (1, 1, 4), (2, 2, 2), (4, 2, 0), (6, 2, 4)),
    ((0, 2, 2), (2, 2, 7), (4, 4, 11)),
)
WHISTLE = ((1, 1.0), (2, 0.08))


def piano(semitones: float, seconds: float) -> np.ndarray:
    """A slightly out-of-tune upright: two strings a hair apart."""
    f = pitch(semitones, G)
    return voice(f, seconds, PLUCK, 6.0) + voice(f * 1.004, seconds, PLUCK, 6.0) * 0.7


def music() -> np.ndarray:
    drums, bass, keys, lead = [], [], [], []

    for beat in range(int(INTRO / BEAT)):
        drums.append((beat * BEAT, kick() * 0.7))
        drums.append((beat * BEAT + BEAT / 2, hat(0.2)))
    keys.append((0.0, tone((pitch(24, G), pitch(36, G)), INTRO, decay=-1.5) * 0.1))

    for bar in range(BARS):
        t0 = INTRO + bar * BAR
        lift = 2 if bar >= LIFT else 0
        root = CHORDS[bar % 4] + lift
        last = bar == BARS - 1

        for beat in range(4):
            drums.append((t0 + beat * BEAT, kick() * (1.0 if beat % 2 == 0 else 0.5)))
            if beat % 2:
                drums.append((t0 + beat * BEAT, snare() * (0.5 if bar >= 8 else 0.3)))
            drums.append((t0 + beat * BEAT + BEAT / 2, hat(0.35)))
        if last or bar == LIFT - 1:
            for s in range(8):
                drums.append((t0 + BAR / 2 + s * BEAT / 4, snare() * (0.25 + 0.05 * s)))

        # Oom-pah: root and fifth on the beats, the chord between.
        for beat in range(4):
            if beat % 2 == 0:
                note = root + (0 if beat == 0 else 7)
                bass.append((t0 + beat * BEAT, voice(pitch(note, G), BEAT, SAW, 4.0) * 0.5))
            else:
                for k in (0, 4, 7):
                    keys.append((t0 + beat * BEAT, piano(root + 24 + k, BEAT * 0.9) * 0.12))
        if bar >= 8:                                    # an extra stab on the "and"s
            for beat in (1, 3):
                for k in (0, 4, 7):
                    keys.append((t0 + beat * BEAT + BEAT / 2, piano(root + 36 + k, BEAT / 2) * 0.06))

        if bar >= 8:
            for start, eighths, step in TUNE[bar % 4]:
                f = pitch(step + lift, 392.0)
                lead.append((t0 + start * BEAT / 2,
                             voice(f, eighths * BEAT / 2, WHISTLE, 1.6, attack=0.02,
                                   vibrato=0.25) * 0.3))

    end = INTRO + BARS * BAR
    drums.append((end, kick()))
    for k in (0, 4, 7, 12):
        keys.append((end, piano(k + 2 + 24, TAIL) * 0.2))
    bass.append((end, voice(pitch(2, G), TAIL, SAW, 4.0) * 0.5))
    keys.append((end, voice(pitch(2 + 36, G), TAIL, SOFT, 3.0) * 0.1))

    x = mix([(0, mix(drums) * 0.8), (0, mix(bass)), (0, mix(keys)), (0, mix(lead))])
    x = x[:int((end + TAIL) * RATE)]
    x = np.tanh(2.0 * x / np.abs(x).max())
    fade = int(0.5 * RATE)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)
    return x


def smash() -> np.ndarray:
    """Glass breaking: a crack, a hiss of pieces, and the pieces ringing."""
    t = clock(0.6)
    hiss = np.diff(rng.standard_normal(len(t) + 1))
    noise = hiss * (np.exp(-30 * t) + 0.25 * np.exp(-8 * t))
    rings = []
    for _ in range(9):
        at = rng.uniform(0.0, 0.25)
        f = rng.uniform(2600, 6200)
        rings.append((at, tone(f, 0.18, decay=7.0) * rng.uniform(0.15, 0.4)))
    return mix([(0, noise), (0, tone((180, 70), 0.08, decay=5.0) * 0.6)] + rings)


def effects() -> dict:
    thud = clock(0.12)
    chime = mix([(i * 0.07, tone(f, 0.3, decay=5.0, overtone=0.3))
                 for i, f in enumerate((1047, 1319, 1568, 2093))])
    buzz = clock(0.4)
    return {
        "hit": (0.5, smash()),
        "gold": (0.5, mix([(0, smash()), (0.05, chime * 0.8)])),
        "miss": (0.3, rng.standard_normal(len(thud)) * np.exp(-45 * thud)
                 + tone((110, 60), 0.12, decay=4.0)),
        "bad": (0.4, np.sign(np.sin(2 * np.pi * 140 * buzz)) * np.exp(-2.5 * buzz) * 0.6
                + tone(70, 0.4, decay=2.0)),
        "point": (0.4, mix([(0, tone(988, 0.25, decay=4.0, overtone=0.2)),
                            (0.1, tone(1319, 0.5, decay=4.0, overtone=0.2))])),
        "win": (0.45, mix([(i * 0.13, tone(f, 0.9 if i == 4 else 0.3, decay=4.0, overtone=0.3))
                           for i, f in enumerate((392, 494, 587, 784, 988))])),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    x = music()
    write(OUT, "music", x * 0.4)
    print(f"music: {len(x) / RATE:.1f}s")
    made = effects()
    for name, (level, x) in made.items():
        write(OUT, name, x * (level / np.abs(x).max()))
    print(f"effects: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
