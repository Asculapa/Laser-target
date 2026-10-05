"""Make the Wild West game's music and sound effects.

    .venv/bin/python tools/west_audio.py

Synthesised with numpy, on the instruments of tools/duel_audio.py, like the
other games' sound. Each street has a piece of its own - its own tune, key,
rhythm and band - and each is livelier than the one before:

1. The Lonesome Trail - a slow ballad in three, harmonica over a guitar;
2. Saloon Rag - ragtime on the saloon piano, with banjo and fiddle;
3. The Chase - a Morricone gallop: guitar riff, whip, anvil, choir, trumpet.

The showdown has no music: a bell, then quiet, then the shout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from duel_audio import PLUCK, SAW, clock, hat, kick, pitch, rng, snare, voice
from story_audio import RATE, mix, tone, write

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "laserapp" / "assets" / "west" / "sound"

E2 = 82.41
WHISTLE = ((1, 1.0), (2, 0.05))
BRASS = tuple((n, 1.0 / n ** 1.3) for n in range(1, 10))
REED = ((1, 1.0), (2, 0.7), (3, 0.55), (4, 0.3), (5, 0.2), (6, 0.1))     # a harmonica
BOW = tuple((n, 0.9 / n) for n in range(1, 9))                           # a fiddle
CHOIR = ((1, 1.0), (2, 0.35), (3, 0.2), (4, 0.06))
BANJO = tuple((n, 1.0 / n) for n in range(1, 9))


def at(semitones: float, base: float) -> float:
    return base * 2.0 ** (semitones / 12.0)


def hoof(level: float) -> np.ndarray:
    """A clop: a knock of filtered noise."""
    t = clock(0.07)
    knock = np.convolve(rng.standard_normal(len(t)), np.ones(12) / 12, mode="same")
    return (knock * 3 + tone(420, 0.07, decay=8.0) * 0.4) * np.exp(-60 * t) * level


def twang(semitones: float, seconds: float, base: float = E2) -> np.ndarray:
    """A plucked string that bends up into the note."""
    t = clock(seconds)
    f = pitch(semitones, base) * (1.0 - 0.03 * np.exp(-25 * t))
    phase = 2 * np.pi * np.cumsum(f) / RATE
    x = sum(a * np.sin(n * phase) for n, a in PLUCK + ((5, 0.08), (6, 0.05)))
    return x * np.exp(-3.5 * t / seconds) * np.minimum(1.0, t / 0.003)


def triad(root: int, minor: bool = False, seventh: bool = False) -> tuple:
    return (root, root + (3 if minor else 4), root + 7) + ((root + 10,) if seventh else ())


def echo(x: np.ndarray, delay: float, gains=(1.0, 0.4, 0.18, 0.07)) -> np.ndarray:
    d = int(delay * RATE)
    out = np.zeros(len(x) + (len(gains) - 1) * d)
    for k, g in enumerate(gains):
        out[k * d:k * d + len(x)] += x * g
    return out


def loop(parts, seconds: float, drive: float) -> np.ndarray:
    """Lay the parts down, fold the tails round to the start, and press it together."""
    x = mix(parts)
    n = int(seconds * RATE)
    if len(x) < n:
        x = np.concatenate([x, np.zeros(n - len(x))])
    x[:len(x) - n] += x[n:]
    x = x[:n]
    return np.tanh(drive * x / np.abs(x).max())


# -- 1. Main Street: The Lonesome Trail ---------------------------------------
# A slow ballad in three, in D: a guitar going boom-chick-chick, a horse
# walking, and a harmonica with the tune once the guitar has set off.
TRAIL_CHORDS = ((0, False), (0, False), (7, False), (7, False), (-3, True), (5, False),
                (7, False), (0, False),
                (5, False), (0, False), (7, False), (7, False), (-3, True), (5, False),
                (7, False), (0, False))
# Six eighths a bar; semitones from D5.
TRAIL = (
    ((0, 4, 0), (4, 2, 4)), ((0, 4, 7), (4, 2, 9)), ((0, 6, 7),),
    ((0, 2, 4), (2, 2, 2), (4, 2, 0)), ((0, 4, -3), (4, 2, -1)), ((0, 6, 2),),
    ((0, 2, 4), (2, 2, 2), (4, 2, -1)), ((0, 6, 0),),
    ((0, 4, 7), (4, 2, 9)), ((0, 4, 12), (4, 2, 11)), ((0, 6, 9),),
    ((0, 2, 7), (2, 2, 9), (4, 2, 7)), ((0, 4, 4), (4, 2, 2)), ((0, 4, 4), (4, 2, 7)),
    ((0, 2, 4), (2, 2, 2), (4, 2, -1)), ((0, 6, 0),),
)


def trail() -> np.ndarray:
    beat = 60.0 / 90
    bar = 3 * beat
    D2 = at(-2, E2)                                 # the key: D
    parts = []
    for b, (root, minor) in enumerate(TRAIL_CHORDS):
        t0 = b * bar
        tones = triad(root, minor)
        parts.append((t0, twang(root, beat * 1.5, D2) * 0.55))               # boom
        for k in (1, 2):                                                     # chick, chick
            for n, step in enumerate(tones):
                parts.append((t0 + k * beat + n * 0.01, twang(step + 12, beat * 0.8, D2) * 0.12))
        for k in range(3):                                                   # a horse, walking
            parts.append((t0 + k * beat, hoof(0.45)))
            parts.append((t0 + k * beat + beat * 0.45, hoof(0.25)))
        if b >= 2:
            for start, eighths, step in TRAIL[b]:
                parts.append((t0 + start * beat / 2,
                              echo(voice(at(step, 587.33), eighths * beat / 2, REED, 1.0,
                                         attack=0.05, vibrato=0.25) * 0.24, 0.75 * beat,
                                   (1.0, 0.3, 0.1))))
    return loop(parts, 16 * bar, 1.6)


# -- 2. The Inn: Saloon Rag --------------------------------------------------
# Ragtime in C on the upright piano - the left hand striding, bass and chord,
# the right hand skipping ahead of the beat - with a banjo, a washboard, and
# the fiddle joining the tune the second time round.
RAG_CHORDS = ((0, False), (0, False), (5, False), (0, False), (7, True), (7, True),
              (0, False), (0, False)) * 2      # (root from C, a seventh on it)
RAG = (                                         # eight eighths a bar; semitones from C5
    ((0, 1, 7), (1, 1, 9), (2, 2, 12), (4, 1, 9), (5, 1, 7), (6, 2, 4)),
    ((0, 1, 5), (1, 1, 4), (2, 2, 0), (4, 4, 4)),
    ((0, 1, 9), (1, 1, 12), (2, 2, 9), (4, 1, 5), (5, 1, 9), (6, 2, 12)),
    ((0, 2, 7), (2, 2, 4), (4, 4, 0)),
    ((0, 1, 11), (1, 1, 14), (2, 2, 11), (4, 1, 7), (5, 1, 11), (6, 2, 14)),
    ((0, 2, 12), (2, 1, 11), (3, 1, 9), (4, 4, 7)),
    ((0, 1, 4), (1, 1, 7), (2, 1, 12), (3, 1, 16), (4, 2, 14), (6, 2, 11)),
    ((0, 6, 12),),
)


def piano(semitones: float, seconds: float, base: float) -> np.ndarray:
    f = at(semitones, base)
    return voice(f, seconds, PLUCK, 5.0) + voice(f * 1.005, seconds, PLUCK, 5.0) * 0.7


def rag() -> np.ndarray:
    beat = 60.0 / 126
    bar = 4 * beat
    C3 = 130.81
    parts = []
    for b in range(16):
        t0 = b * bar
        root, seventh = RAG_CHORDS[b]
        tones = triad(root, seventh=seventh)
        for k in range(4):
            if k % 2 == 0:                                                   # stride: bass...
                parts.append((t0 + k * beat, piano(root - 12 + (0 if k == 0 else 7), beat, C3) * 0.4))
            else:                                                            # ...and chord
                for step in tones:
                    parts.append((t0 + k * beat, piano(step, beat * 0.7, C3) * 0.13))
            parts.append((t0 + k * beat + beat / 2, hat(0.35)))              # washboard
            parts.append((t0 + k * beat, hat(0.15)))
            for n, step in enumerate(tones[:3]):                             # banjo, off the beat
                parts.append((t0 + k * beat + beat / 2 + n * 0.008,
                              voice(at(step + 12, C3), beat / 2, BANJO, 9.0) * 0.07))
        for start, eighths, step in RAG[b % 8]:
            t = t0 + start * beat / 2
            parts.append((t, piano(step + 12, eighths * beat / 2 + 0.1, C3) * 0.3))
            if b >= 8:
                parts.append((t, voice(at(step, 523.25), eighths * beat / 2, BOW, 0.6,
                                       attack=0.05, vibrato=0.4) * 0.14))
    return loop(parts, 16 * bar, 1.9)


# -- 3. Calico: The Chase ------------------------------------------------------
# A chase in A minor, the way Morricone wrote them: a low, twanging guitar riff
# that never lets up, drums at a gallop, a whip, an anvil, a choir - and a
# trumpet with the tune once the riff has gone round four times.
CHASE_CHORDS = (0, 0, 1, 0, 0, 0, -2, -5, 0, 0, 1, 0, -4, -4, -5, -5)   # from A; -5 is E major
RIFF = (0, 0, 12, 0, 1, 0, 10, 7)                                       # eighths, from the chord's root
CHASE = (                                                               # semitones from A4
    (), (), (), (),
    ((0, 6, 0), (6, 2, 3)), ((0, 4, 1), (4, 4, 0)), ((0, 2, -2), (2, 2, 0), (4, 4, 3)),
    ((0, 8, -1),),
    ((0, 6, 7), (6, 2, 8)), ((0, 4, 7), (4, 4, 5)), ((0, 2, 3), (2, 2, 5), (4, 4, 3)),
    ((0, 8, 0),),
    ((0, 4, -1), (4, 4, 0)), ((0, 4, 3), (4, 4, 7)), ((0, 4, 8), (4, 4, 7)), ((0, 8, -1),),
)


def whip() -> np.ndarray:
    t = clock(0.12)
    return np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-35 * t) \
        + tone((3000, 900), 0.12, decay=4.0) * 0.5


def anvil() -> np.ndarray:
    return sum(tone(f, 1.2, decay=d) * a
               for f, a, d in ((880, 1.0, 6), (2110, 0.6, 9), (3170, 0.4, 12)))


def chase() -> np.ndarray:
    beat = 60.0 / 150
    bar = 4 * beat
    A2 = 110.0
    parts = []
    for b in range(16):
        t0 = b * bar
        root = CHASE_CHORDS[b]
        for k in range(4):
            parts.append((t0 + k * beat, kick() * (1.0 if k % 2 == 0 else 0.6)))
            if k % 2:
                parts.append((t0 + k * beat, snare() * 0.6))
            for n in range(4):                                               # hooves at a gallop
                parts.append((t0 + k * beat + n * beat / 4, hoof(0.8 if n == 3 else 0.35)))
        if b % 4 == 3:                                                       # a whip, and a roll
            parts.append((t0 + 3.5 * beat, whip() * 0.8))
            for n in range(8):
                parts.append((t0 + 2 * beat + n * beat / 4, snare() * (0.2 + 0.07 * n)))
        if b % 2 == 0:
            parts.append((t0, anvil() * 0.2))
        for n, step in enumerate(RIFF):                                      # the riff
            parts.append((t0 + n * beat / 2, twang(root + step, beat / 2 + 0.05, A2) * 0.45))
        for step in triad(root, minor=root == 0):    # the choir: Am, else Bb, G, F, E major
            for detune in (1.0, 1.004, 0.996):
                parts.append((t0, voice(at(step + 12, A2) * detune, bar, CHOIR, 0.5,
                                        attack=0.25) * 0.05))
        for start, eighths, step in CHASE[b]:
            parts.append((t0 + start * beat / 2,
                          echo(voice(at(step, 440.0), eighths * beat / 2, BRASS, 0.7,
                                     attack=0.03, vibrato=0.2) * 0.22, 0.75 * beat,
                               (1.0, 0.3, 0.1))))
    return loop(parts, 16 * bar, 2.4)


STREETS = (("The Lonesome Trail", trail), ("Saloon Rag", rag), ("The Chase", chase))


def _smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    """A one-pole low-pass: `seconds` is its time constant."""
    n = max(2, int(seconds * RATE * 5))
    kernel = np.exp(-np.arange(n) / (seconds * RATE))
    return np.convolve(x, kernel / kernel.sum())[:len(x)]


def shot(size: float = 1.0, far: float = 0.0) -> np.ndarray:
    """A revolver going off, the way a western sounds: a hard crack, the
    blast behind it, a thump in the chest, and the report coming back off
    the street's wooden fronts. `size` makes it bigger and lower, `far` puts
    it further off - softer at the front, more echo."""
    t = clock(1.1)
    noise = rng.standard_normal(len(t))
    crack = np.diff(noise, prepend=0.0) * np.exp(-t / 0.0025)        # the muzzle: all top
    blast = _smooth(noise, 0.00012 * size) * np.exp(-t / (0.045 * size))
    blast /= np.abs(blast).max()
    thump = np.zeros(len(t))
    thump[:int(0.35 * RATE)] = tone((150 / size, 48 / size), 0.35, decay=7.0)
    front = crack * 0.9 * (1 - far) + blast * 1.0 + thump * 0.9
    front = np.tanh(2.2 * front) / np.tanh(2.2)                        # driven, like a real report
    # The street answers: dull, late copies, and a tail that rolls away.
    dull = _smooth(front, 0.00045)
    dull /= np.abs(dull).max()
    out = front * (1 - 0.45 * far)
    for delay, gain in ((0.105, 0.30), (0.24, 0.17), (0.41, 0.09)):
        i = int(delay * RATE)
        out[i:] += dull[:len(out) - i] * gain * (1 + far)
    roll = _smooth(rng.standard_normal(len(t)), 0.0008) * np.exp(-t / 0.28) * np.minimum(1, t / 0.03)
    return out + roll / np.abs(roll).max() * 0.12 * (1 + far)


def effects() -> dict:
    """Every sound of a bullet fired starts with the shot itself: one sound
    per shot, so a player (and winsound's single channel) never hears the
    report and what it hit fight over who plays."""
    thud = clock(0.2)
    ricochet = tone((2600, 1300), 0.45, decay=3.0) * (1 + 0.3 * np.sin(2 * np.pi * 30 * clock(0.45)))
    bell = mix([(0, sum(tone(220 * p, 3.0, decay=d) * a
                        for p, a, d in ((1, 1.0, 4), (2.0, 0.6, 5), (2.4, 0.4, 7),
                                        (3.0, 0.3, 8), (4.2, 0.2, 10))))])
    glass = sum(tone(f, 0.5, decay=d) * a for f, a, d in
                ((2900, 0.5, 9), (4100, 0.4, 12), (5300, 0.3, 14), (3500, 0.3, 11)))
    shards = mix([(0.012 * i + 0.03 * rng.random(), tone(3000 + 2500 * rng.random(), 0.12, decay=8.0)
                   * 0.25) for i in range(14)])
    clang = sum(tone(f, 0.5, decay=d) * a for f, a, d in ((620, 1.0, 6), (1480, 0.6, 9), (2330, 0.4, 12)))
    return {
        "hit": (0.9, mix([(0, shot()), (0.03, rng.standard_normal(len(thud))
                                        * np.exp(-30 * thud) * 0.35
                                        + tone((140, 70), 0.2, decay=5.0) * 0.6)])),
        "miss": (0.9, mix([(0, shot()), (0.07, ricochet * 0.35)])),
        "enemy": (0.95, mix([(0, shot(1.5, far=0.5)), (0.3, shot(1.5, far=1.0) * 0.25)])),
        "hurt": (0.9, mix([(0, shot(1.5, far=0.3)), (0.12, tone((300, 180), 0.3, decay=3.0,
                                                                  overtone=0.4) * 0.35)])),
        "oops": (0.9, mix([(0, shot()), (0.2, tone(520, 0.2, decay=3.0, overtone=0.3) * 0.4),
                           (0.43, tone(390, 0.35, decay=3.0, overtone=0.3) * 0.4)])),
        "bottle": (0.9, mix([(0, shot()), (0.03, glass * 0.35), (0.04, shards)])),
        "clang": (0.9, mix([(0, shot()), (0.03, clang * 0.45)])),
        "coin": (0.9, mix([(0, shot()), (0.05, tone(1319, 0.12, decay=3.0, overtone=0.3) * 0.45),
                           (0.13, tone(1976, 0.4, decay=4.0, overtone=0.3) * 0.45)])),
        "star": (0.9, mix([(0, shot())] + [(0.05 + i * 0.07, tone(f, 0.35, decay=4.0, overtone=0.3) * 0.4)
                                          for i, f in enumerate((784, 988, 1175, 1568))])),
        "bell": (0.45, bell),
        "draw": (0.5, tone((1500, 2600), 0.35, decay=1.5)),       # a whistle: now!
        "click": (0.45, mix([(0, tone(2400, 0.03, decay=6.0)), (0, tone(900, 0.04, decay=8.0) * 0.5)])),
        "reload": (0.4, mix([(i * 0.07, tone(1800 + 150 * i, 0.03, decay=6.0)) for i in range(6)]
                            + [(0.45, tone((700, 1100), 0.08, decay=3.0))])),
        "clear": (0.45, mix([(i * 0.14, tone(f, 0.9 if i == 4 else 0.3, decay=4.0, overtone=0.3))
                             for i, f in enumerate((330, 392, 494, 659, 784))])),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for level, (title, piece) in enumerate(STREETS):
        x = piece()
        loud = (0.06, 0.075, 0.09)[level]               # and each louder than the last
        write(OUT, f"music{level + 1}", x * (loud / np.sqrt(np.mean(x ** 2))))
        print(f"music{level + 1}: {title}, {len(x) / RATE:.1f}s")
    (OUT / "music.wav").unlink(missing_ok=True)
    made = effects()
    for name, (level, x) in made.items():
        write(OUT, name, x * (level / np.abs(x).max()))
    print(f"effects: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
