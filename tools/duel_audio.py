"""Make the range duel's music and sound effects.

    .venv/bin/python tools/duel_audio.py

All of it is plain synthesis and needs only numpy, so nothing here has a
licence to worry about. The music is one piece as long as a round - the
countdown, the minute, and a last chord - rather than a short loop, so it
can build as the clock runs down: drums and bass first, then an arpeggio,
then a tune, and for the last twelve seconds all of it a tone higher. The
app starts it with the countdown, so the two stay in step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from story_audio import RATE, mix, tone, write

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from laserapp.duel import Duel  # noqa: E402

OUT = ROOT / "laserapp" / "assets" / "duel"

BEAT = 0.5                          # 120 to the minute
BAR = 4 * BEAT
INTRO = Duel.COUNTDOWN              # six beats: the music turns with the "go"
BARS = int(Duel.DURATION / BAR)
LIFT = BARS - 6                     # the bar where everything goes up a tone
TAIL = 3.0

A = 55.0                            # the key: A minor
# Root of each bar's chord, in semitones from A, and whether it is major.
CHORDS = ((0, False), (0, False), (-4, True), (-2, True))
# The tune, one row a bar: (eighth it starts on, eighths it lasts, semitones
# from the A above middle C).
TUNE = (
    ((0, 2, 0), (2, 2, 3), (4, 4, 7)),
    ((0, 1, 5), (1, 1, 3), (2, 2, 0), (5, 3, 3)),
    ((0, 2, 0), (2, 2, 3), (4, 4, 8)),
    ((0, 2, 7), (2, 2, 5), (4, 4, 2)),
)

rng = np.random.default_rng(7)


def pitch(semitones: float, base: float = A) -> float:
    return base * 2.0 ** (semitones / 12.0)


def clock(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def voice(freq: float, seconds: float, harmonics, decay: float, attack: float = 0.004,
          vibrato: float = 0.0) -> np.ndarray:
    """A note built from its harmonics - so nothing in it is above what the
    sample rate can carry - that fades over its length."""
    t = clock(seconds)
    phase = 2 * np.pi * freq * t
    if vibrato:
        phase += vibrato * np.sin(2 * np.pi * 5.5 * t)
    x = sum(a * np.sin(n * phase) for n, a in harmonics if n * freq < RATE * 0.45)
    release = np.minimum(1.0, (seconds - t) / 0.02)
    return x * np.minimum(1.0, t / attack) * np.exp(-decay * t / seconds) * release


SAW = tuple((n, 1.0 / n) for n in range(1, 9))
SQUARE = tuple((n, 1.0 / n) for n in (1, 3, 5, 7))
PLUCK = ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12))
SOFT = ((1, 1.0), (2, 0.25), (3, 0.08))


def kick() -> np.ndarray:
    return tone((130, 42), 0.22, decay=5.0)


def snare() -> np.ndarray:
    t = clock(0.16)
    return rng.standard_normal(len(t)) * np.exp(-28 * t) * 0.8 + tone(185, 0.16, decay=9.0) * 0.5


def hat(level: float) -> np.ndarray:
    t = clock(0.05)
    hiss = np.diff(rng.standard_normal(len(t) + 1))        # what is left is the top
    return hiss * np.exp(-90 * t) * level


def music() -> np.ndarray:
    drums, bass, chords, lead = [], [], [], []

    # The countdown: one kick a beat, and a note that climbs to the start.
    for beat in range(int(INTRO / BEAT)):
        drums.append((beat * BEAT, kick() * 0.8))
        drums.append((beat * BEAT + BEAT / 2, hat(0.25)))
    chords.append((0.0, tone((pitch(12), pitch(24)), INTRO, decay=-1.5) * 0.12))

    for bar in range(BARS):
        t0 = INTRO + bar * BAR
        lift = 2 if bar >= LIFT else 0
        root, major = CHORDS[bar % 4]
        root += lift
        third = 4 if major else 3
        last = bar == BARS - 1

        for beat in range(4):
            drums.append((t0 + beat * BEAT, kick()))
            if bar >= 8 and beat % 2:
                drums.append((t0 + beat * BEAT, snare() * 0.55))
        # Offbeat hats, then sixteenths once the tune is in.
        steps = 16 if bar >= 16 else 8
        for s in range(steps):
            off = s % (steps // 4) == steps // 8
            drums.append((t0 + s * BAR / steps, hat(0.5 if off else 0.22)))
        if last or bar == LIFT - 1:                         # a roll into what follows
            for s in range(8):
                drums.append((t0 + BAR / 2 + s * BEAT / 4, snare() * (0.25 + 0.05 * s)))

        # Bass: the root on the beat, its octave between.
        for s in range(8):
            f = pitch(root + (12 if s % 2 else 0))
            bass.append((t0 + s * BEAT / 2, voice(f, BEAT / 2, SAW, 3.0) * 0.5))

        # A held chord under it all, and from the second section its arpeggio.
        for step in (0, third, 7):
            chords.append((t0, voice(pitch(root + 24 + step), BAR, SOFT, 0.6, attack=0.15) * 0.09))
        if bar >= 8:
            notes = (0, third, 7, 12, 7, third, 0, 7)
            for s in range(16):
                f = pitch(root + 36 + notes[s % 8])
                chords.append((t0 + s * BEAT / 4, voice(f, BEAT / 2, PLUCK, 7.0) * 0.22))

        if bar >= 16:
            for start, eighths, step in TUNE[bar % 4]:
                f = pitch(step + lift, 440.0)
                lead.append((t0 + start * BEAT / 2,
                             voice(f, eighths * BEAT / 2, SQUARE, 2.2, attack=0.01,
                                   vibrato=0.12) * 0.28))

    # Time: one last chord, left to ring.
    end = INTRO + BARS * BAR
    drums.append((end, kick()))
    drums.append((end, hat(0.5)))
    for step in (0, 3, 7, 12):
        chords.append((end, voice(pitch(step + 2 + 24), TAIL, PLUCK, 4.0) * 0.2))
    bass.append((end, voice(pitch(2), TAIL, SAW, 4.0) * 0.5))

    # The arpeggio and the tune get an echo, a dotted eighth behind.
    wet = mix(chords + lead)
    delay = int(0.75 * BEAT * RATE)
    echo = np.zeros(len(wet) + 2 * delay)
    for k, gain in enumerate((1.0, 0.35, 0.12)):
        echo[k * delay:k * delay + len(wet)] += wet * gain
    x = mix([(0, mix(drums) * 0.9), (0, mix(bass)), (0, echo)])
    x = x[:int((end + TAIL) * RATE)]
    x = np.tanh(2.2 * x / np.abs(x).max())                  # pressed together a little
    fade = int(0.5 * RATE)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)
    return x


def effects() -> dict:
    t = clock(0.3)
    # A steel plate rings on notes that do not belong together.
    plate = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-d * t)
                for f, a, d in ((1870, 1.0, 16), (2930, 0.6, 22), (4410, 0.35, 30)))
    click = rng.standard_normal(len(t)) * np.exp(-400 * t) * 0.6
    thud = clock(0.12)
    return {
        "hit": (0.4, plate + click),
        "bad": (0.45, mix([(0, tone((200, 120), 0.3, decay=2.0, overtone=0.7)),
                           (0.16, tone((150, 90), 0.34, decay=2.5, overtone=0.7))])),
        "miss": (0.3, rng.standard_normal(len(thud)) * np.exp(-45 * thud)
                 + tone((110, 60), 0.12, decay=4.0)),
        "win": (0.45, mix([(i * 0.13, tone(f, 0.9 if i == 4 else 0.3, decay=4.0, overtone=0.3))
                           for i, f in enumerate((523, 523, 659, 784, 1047))])),
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
