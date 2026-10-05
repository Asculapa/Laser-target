"""Make the story game's music: a piece for each chapter.

    .venv/bin/python tools/story_music.py

Plain synthesis, like the duel's (duel_audio.py, whose instruments these
are), so there is no licence to worry about. A chapter is not timed, so each
piece is a loop of about forty seconds that ends where it can begin again:

  1  Sparks on the Wind       light and hopeful: bells over a slow major round
  2  What the Dark Wants      a heartbeat and a creeping bass, closing in
  3  Ships in the Dark        a slow rocking in six-eight, with the sea under it
  4  The Keepers' Stars       hardly there: single bells over a held chord
  5  The Heart of the Storm   drums, thunder, and a tune that fights back
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from duel_audio import PLUCK, SAW, SOFT, SQUARE, clock, hat, kick, pitch, snare, voice
from story_audio import RATE, mix, write

OUT = Path(__file__).resolve().parent.parent / "laserapp" / "assets" / "story" / "music"

BELL = ((1, 1.0), (2.76, 0.3), (5.4, 0.12))         # partials that do not line up
FLUTE = ((1, 1.0), (2, 0.2), (3, 0.05))

Parts = List[Tuple[float, np.ndarray]]
Chord = Tuple[int, bool]                # root in semitones from the key, and major?
Note = Tuple[float, float, int]         # start and length in steps, semitones

rng = np.random.default_rng(11)


def third(major: bool) -> int:
    return 4 if major else 3


def echo(x: np.ndarray, delay: float, gains: Sequence[float] = (1.0, 0.4, 0.16)) -> np.ndarray:
    d = int(delay * RATE)
    out = np.zeros(len(x) + d * (len(gains) - 1))
    for k, gain in enumerate(gains):
        out[k * d:k * d + len(x)] += x * gain
    return out


def pad(parts: Parts, at: float, seconds: float, chord: Chord, base: float, level: float,
        attack: float = 0.4, vibrato: float = 0.0) -> None:
    root, major = chord
    for step in (0, third(major), 7):
        parts.append((at, voice(pitch(root + step, base), seconds, SOFT, 0.5, attack=attack,
                                vibrato=vibrato) * level))


def tune(parts: Parts, at: float, step: float, notes: Sequence[Note], base: float,
         harmonics, level: float, decay: float = 2.0, vibrato: float = 0.1) -> None:
    for start, length, semitones in notes:
        parts.append((at + start * step,
                      voice(pitch(semitones, base), length * step, harmonics, decay,
                            attack=0.02, vibrato=vibrato) * level))


def rumble(seconds: float) -> np.ndarray:
    x = np.cumsum(rng.standard_normal(int(seconds * RATE)))
    x -= np.convolve(x, np.ones(1500) / 1500, mode="same")      # no drift
    x *= np.exp(-3.5 * np.arange(len(x)) / len(x))
    return x / np.abs(x).max()


def finish(layers: Sequence[Tuple[Parts, float]], seconds: float, level: float,
           drive: float = 1.8) -> np.ndarray:
    """The layers - each (its notes, the echo it gets) - as one loop, cut to
    length: what rings on past the end would only be cut off by the start."""
    n = int(seconds * RATE)
    x = np.zeros(n)
    for parts, delay in layers:
        if parts:
            layer = echo(mix(parts), delay) if delay else mix(parts)
            m = min(n, len(layer))
            x[:m] += layer[:m]
    x = np.tanh(drive * x / np.abs(x).max()) * level
    fade = int(0.08 * RATE)
    x[-fade:] *= np.linspace(1.0, 0.0, fade)
    return x


# -- 1: Sparks on the Wind --------------------------------------------------
def sparks() -> np.ndarray:
    D = 293.66
    beat = 60 / 96.0
    bar = 4 * beat
    chords = ((0, True), (-3, False), (-7, True), (-5, True))       # D  Bm  G  A
    melody = (((0, 2, 4), (2, 2, 7), (4, 4, 12)),
              ((0, 2, 9), (2, 2, 7), (4, 4, 4)),
              ((0, 2, 5), (2, 2, 9), (4, 4, 12)),
              ((0, 2, 11), (2, 2, 9), (4, 4, 7)))
    low, bells, lead = [], [], []
    for b in range(16):
        at, (root, major) = b * bar, chords[b % 4]
        pad(low, at, bar, chords[b % 4], D, 0.06)
        for beat_ in (0, 2):
            low.append((at + beat_ * beat, voice(pitch(root - 24, D), 2 * beat, SOFT, 2.0) * 0.3))
        t3 = third(major)
        for s, step in enumerate((0, t3, 7, 12, 12 + t3, 12, 7, t3)):
            bells.append((at + s * beat / 2,
                          voice(pitch(root + 12 + step, D), beat, BELL, 5.0) * 0.11))
        if b >= 4:
            tune(lead, at, beat / 2, melody[b % 4], 2 * D, FLUTE, 0.17)
            for _ in range(2):                          # the sparks themselves
                when = at + rng.integers(0, 8) * beat / 2 + beat / 4
                note = int(rng.choice((0, 2, 4, 7, 9))) + 24
                bells.append((when, voice(pitch(note, 2 * D), 0.6, BELL, 6.0) * 0.05))
    return finish(((low, 0), (bells, 0.75 * beat), (lead, 0.75 * beat)), 16 * bar, 0.3)


# -- 2: What the Dark Wants -------------------------------------------------
def siege() -> np.ndarray:
    E = 82.41
    beat = 60 / 108.0
    bar = 4 * beat
    chords = ((0, False), (0, False), (-4, True), (-2, True), (0, False), (-5, True))
    melody = (((0, 3, 7), (3, 1, 5), (4, 4, 3)),                    # Em Em C D Em B
              ((0, 3, 3), (3, 1, 2), (4, 4, 0)),
              ((0, 4, 8), (4, 4, 3)),
              ((0, 4, 10), (4, 4, 5)),
              ((0, 6, 7), (6, 2, 3)),
              ((0, 4, 7), (4, 4, 11)))
    drums, low, mid, lead = [], [], [], []
    for b in range(18):
        at, (root, major), section = b * bar, chords[b % 6], b // 6
        for beat_ in (0, 2):                            # a heartbeat
            drums.append((at + beat_ * beat, kick()))
            drums.append((at + (beat_ + 0.7) * beat, kick() * 0.55))
        for s, step in enumerate((0, 0, 12, 0, 0, 12, 0, 7)):
            low.append((at + s * beat / 2,
                        voice(pitch(root + step, E), beat / 2, SAW, 4.0) * 0.4))
        pad(mid, at, bar, chords[b % 6], 4 * E, 0.045, vibrato=0.25)
        if section >= 1:
            for s in range(16):
                drums.append((at + s * beat / 4, hat(0.3 if s % 4 == 2 else 0.12)))
            for s, step in enumerate((0, third(major), 7, third(major)) * 2):
                mid.append((at + s * beat / 2,
                            voice(pitch(root + step, 4 * E), beat / 2, PLUCK, 6.0) * 0.12))
        if section >= 2:
            for beat_ in (1, 3):
                drums.append((at + beat_ * beat, snare() * 0.3))
            tune(lead, at, beat / 2, melody[b % 6], 4 * E, SQUARE, 0.14)
    return finish(((drums, 0), (low, 0), (mid, 0.75 * beat), (lead, 0.75 * beat)),
                  18 * bar, 0.32, drive=2.4)


# -- 3: Ships in the Dark ---------------------------------------------------
def ships() -> np.ndarray:
    A = 110.0
    step = 0.36                                         # an eighth; six to the bar
    bar = 6 * step
    Am, F, C, G, E = (0, False), (-4, True), (3, True), (-2, True), (-5, True)
    chords = (Am, F, C, G, Am, F, G, Am) * 2 + (F, E)
    melody = (((0, 3, 7), (3, 3, 12)), ((0, 3, 12), (3, 3, 8)), ((0, 3, 10), (3, 3, 7)),
              ((0, 6, 5),), ((0, 3, 3), (3, 3, 7)), ((0, 3, 8), (3, 3, 12)),
              ((0, 3, 14), (3, 3, 10)), ((0, 6, 12),))
    ending = (((0, 6, 8),), ((0, 3, 7), (3, 3, 11)))
    low, lead = [], []
    for b, (root, major) in enumerate(chords):
        at = b * bar
        t3 = third(major)
        for s, note in enumerate((0, 7, 12, 12 + t3, 12, 7)):       # the rocking
            low.append((at + s * step,
                        voice(pitch(root + note, A), 3 * step, SOFT, 2.5) * 0.24))
        pad(low, at, bar, (root, major), 2 * A, 0.05, attack=0.5)
        if b >= 16:
            tune(lead, at, step, ending[b - 16], 4 * A, FLUTE, 0.16, decay=1.2, vibrato=0.15)
        elif b >= 8:
            tune(lead, at, step, melody[b % 8], 4 * A, FLUTE, 0.16, decay=1.2, vibrato=0.15)
    # The sea: a hiss that comes and goes, once every two bars.
    t = clock(len(chords) * bar)
    sea = np.convolve(rng.standard_normal(len(t)), np.ones(40) / 40, mode="same")
    sea *= (0.5 - 0.5 * np.cos(2 * np.pi * t / (2 * bar))) ** 2
    return finish(((low, 0), (lead, 2 * step), ([(0.0, sea / np.abs(sea).max() * 0.09)], 0)),
                  len(chords) * bar, 0.3)


# -- 4: The Keepers' Stars --------------------------------------------------
def stars() -> np.ndarray:
    D = 146.83
    bar = 5.0
    D_, G, Bm, A = (0, True), (-7, True), (-3, False), (-5, True)
    chords = (D_, D_, G, D_, Bm, G, A, D_)
    # One bell every beat and a quarter, on notes of the chord under it - so
    # whichever stars the player lights, their own bell sits well with it.
    bells_ = ((12, 7, 4, 7), (16, 12, 9, 7), (12, 9, 5, 9), (4, 7, 12, 16),
              (9, 12, 16, 12), (17, 12, 9, 5), (14, 11, 7, 11), (12, 7, 4, 0))
    low, bells = [], []
    for b, (root, major) in enumerate(chords):
        at = b * bar
        for note in (0, 7, 12 + third(major), 19):
            low.append((at, voice(pitch(root + note, D), bar, SOFT, 0.4, attack=1.2) * 0.07))
        for s, note in enumerate(bells_[b]):
            bells.append((at + s * bar / 4, voice(pitch(note, 4 * D), 2.5, BELL, 4.0) * 0.14))
    return finish(((low, 0), (bells, 0.42)), len(chords) * bar, 0.26, drive=1.4)


# -- 5: The Heart of the Storm ----------------------------------------------
def storm() -> np.ndarray:
    D = 73.42
    beat = 60 / 132.0
    bar = 4 * beat
    Dm, Bb, C, A = (0, False), (-4, True), (-2, True), (-5, True)
    order = tuple(range(8)) * 2 + (0, 1, 2, 3, 4, 7)    # which bar of the round is played
    chords = (Dm, Dm, Bb, C, Dm, Dm, Bb, A)
    melody = (((0, 3, 0), (3, 1, 3), (4, 4, 7)), ((0, 2, 8), (2, 2, 7), (4, 4, 3)),
              ((0, 3, 8), (3, 1, 12), (4, 4, 15)), ((0, 2, 14), (2, 2, 12), (4, 4, 10)),
              ((0, 3, 12), (3, 1, 15), (4, 4, 19)), ((0, 2, 17), (2, 2, 15), (4, 4, 12)),
              ((0, 4, 15), (4, 4, 12)), ((0, 4, 11), (4, 4, 14)))
    drums, low, mid, lead = [], [], [], []
    for b, k in enumerate(order):
        at, (root, major) = b * bar, chords[k]
        if b % 8 == 0:
            drums.append((at, rumble(2.5) * 0.9))       # thunder
        for beat_ in range(4):
            drums.append((at + beat_ * beat, kick()))
            if beat_ % 2:
                drums.append((at + beat_ * beat, snare() * 0.5))
        for s in range(16):
            drums.append((at + s * beat / 4, hat(0.4 if s % 4 == 2 else 0.18)))
        if b == len(order) - 1:                         # a roll back to the top
            for s in range(8):
                drums.append((at + bar / 2 + s * beat / 4, snare() * (0.25 + 0.05 * s)))
        for s in range(8):
            low.append((at + s * beat / 2,
                        voice(pitch(root + (12 if s % 2 else 0), D), beat / 2, SAW, 3.0) * 0.5))
        pad(mid, at, bar, chords[k], 4 * D, 0.06, attack=0.1)
        if b >= 4:
            t3 = third(major)
            for s in range(16):
                note = (0, t3, 7, 12, 7, t3, 0, 7)[s % 8]
                mid.append((at + s * beat / 4,
                            voice(pitch(root + 24 + note, D), beat / 2, PLUCK, 7.0) * 0.18))
        if b >= 8:
            tune(lead, at, beat / 2, melody[k], 8 * D, SQUARE, 0.24, vibrato=0.12)
    return finish(((drums, 0), (low, 0), (mid, 0.75 * beat), (lead, 0.75 * beat)),
                  len(order) * bar, 0.38, drive=2.2)


PIECES = (sparks, siege, ships, stars, storm)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for n, piece in enumerate(PIECES, 1):
        x = piece()
        write(OUT, f"c{n}", x)
        print(f"c{n} {piece.__name__:7} {len(x) / RATE:5.1f}s  "
              f"level {np.sqrt((x * x).mean()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
