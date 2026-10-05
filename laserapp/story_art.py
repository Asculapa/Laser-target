"""Pictures for the story game - all of them drawn, none of them loaded.

Two rules shape how everything here looks, and both come from the camera
that is watching this screen (see overlay.py):

* no red. Night, sea and fog are blues and cool greys; light is lime and
  cyan. Every colour keeps its red well under its green.
* nothing large is bright. A laser dot on a lit patch has nothing to stand
  out against, so scenery stays dim and only outlines and small details glow.

A scene is a still picture, rendered once, plus a little movement drawn over
it every frame.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import cv2
import numpy as np

from . import overlay

SPARK = overlay.YELLOW              # the lamp's light
FOG = (170, 150, 110)               # the Gloam
STONE = (46, 44, 36)
SEA = (30, 16, 0)
WAVE = (72, 44, 8)

# The bay of chapter 3, as shares of the screen: boats cross it left to
# right, and the reef has one gap.
HORIZON = 0.22
REEF_X = 0.70
CHANNEL = (0.53, 0.67)              # top and bottom of the gap
HARBOUR_X = 0.93


def dim(colour, k: float) -> Tuple[int, int, int]:
    return tuple(int(c * k) for c in colour)


def _poly(points) -> np.ndarray:
    return np.array(points, np.int32).reshape(-1, 1, 2)


# ---------------------------------------------------------------------------
# things: drawn at a place, at a size, at a time
# ---------------------------------------------------------------------------
def spark(img, x: float, y: float, r: float, t: float, colour=SPARK,
          heading: Tuple[float, float] = (0.0, 0.0)) -> None:
    """A piece of the lamp: a small bright core, turning rays, a faint halo."""
    c = (int(x), int(y))
    cv2.circle(img, c, int(r), dim(colour, 0.14), -1, cv2.LINE_AA)
    cv2.circle(img, c, int(r), dim(colour, 0.7), 2, cv2.LINE_AA)
    if heading != (0.0, 0.0):
        tail = (int(x - heading[0] * r * 2.4), int(y - heading[1] * r * 2.4))
        cv2.line(img, c, tail, dim(colour, 0.35), 2, cv2.LINE_AA)
    for k in range(4):
        a = t * 1.7 + k * math.pi / 4
        reach = r * (0.78 if k % 2 else 0.5)
        d = (math.cos(a) * reach, math.sin(a) * reach)
        cv2.line(img, (int(x - d[0]), int(y - d[1])), (int(x + d[0]), int(y + d[1])),
                 colour, 1, cv2.LINE_AA)
    cv2.circle(img, c, max(2, int(r * 0.2)), colour, -1, cv2.LINE_AA)


def gloamling(img, x: float, y: float, r: float, t: float, phase: float = 0.0,
              look: Tuple[float, float] = (0.0, 1.0), eyes: int = 2,
              strength: float = 1.0) -> None:
    """A scrap of the fog: a shape that never holds still, and its eyes."""
    points = []
    for k in range(18):
        a = 2 * math.pi * k / 18
        rr = r * (1 + 0.13 * math.sin(3 * a + t * 3.1 + phase)
                  + 0.07 * math.sin(5 * a - t * 2.3 + phase))
        points.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    shape = _poly(points)
    cv2.fillPoly(img, [shape], dim(FOG, 0.2 * strength), cv2.LINE_AA)
    cv2.polylines(img, [shape], True, dim(FOG, strength), 2, cv2.LINE_AA)
    norm = math.hypot(*look) or 1.0
    lx, ly = look[0] / norm, look[1] / norm
    for k in range(eyes):
        side = (k - (eyes - 1) / 2) * 0.62
        ex = x + lx * r * 0.22 - ly * r * side
        ey = y + ly * r * 0.22 + lx * r * side
        size = max(2, int(r * 0.17))
        cv2.circle(img, (int(ex), int(ey)), size, dim(overlay.WHITE, strength), -1, cv2.LINE_AA)
        cv2.circle(img, (int(ex + lx * size * 0.4), int(ey + ly * size * 0.4)),
                   max(1, size // 2), (0, 0, 0), -1, cv2.LINE_AA)


def moth(img, x: float, y: float, r: float, t: float, phase: float = 0.0) -> None:
    """A glow-moth: a friend. Green, winged, and nothing like a gloamling."""
    colour = overlay.GREEN
    flap = 0.35 + 0.65 * abs(math.sin(t * 9.0 + phase))
    for side in (-1, 1):
        upper = [(x, y - r * 0.15), (x + side * r * flap, y - r * 0.8),
                 (x + side * r * 1.05 * flap, y - r * 0.1), (x, y + r * 0.1)]
        lower = [(x, y + r * 0.05), (x + side * r * 0.75 * flap, y + r * 0.7),
                 (x, y + r * 0.35)]
        for wing in (upper, lower):
            cv2.fillPoly(img, [_poly(wing)], dim(colour, 0.18), cv2.LINE_AA)
            cv2.polylines(img, [_poly(wing)], True, colour, 2, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(y)), (max(2, int(r * 0.13)), int(r * 0.5)), 0, 0, 360,
                colour, -1, cv2.LINE_AA)
    for side in (-1, 1):
        cv2.line(img, (int(x), int(y - r * 0.45)),
                 (int(x + side * r * 0.3), int(y - r * 0.95)), colour, 1, cv2.LINE_AA)


def boat(img, x: float, y: float, s: float, colour, t: float, phase: float = 0.0) -> None:
    """A fishing boat, seen from the side, with a lantern at the masthead."""
    tilt = 0.07 * math.sin(t * 1.6 + phase)
    ca, sa = math.cos(tilt), math.sin(tilt)

    def at(u: float, v: float) -> Tuple[int, int]:
        return int(x + (u * ca - v * sa) * s), int(y + (u * sa + v * ca) * s)

    hull = [at(-1.0, 0.0), at(1.1, 0.0), at(0.7, 0.42), at(-0.75, 0.42)]
    cv2.fillPoly(img, [_poly(hull)], dim(colour, 0.2), cv2.LINE_AA)
    cv2.polylines(img, [_poly(hull)], True, colour, 2, cv2.LINE_AA)
    cv2.line(img, at(0.0, 0.0), at(0.0, -1.25), colour, 2, cv2.LINE_AA)
    sail = [at(0.1, -1.1), at(0.1, -0.15), at(0.85, -0.15)]
    cv2.polylines(img, [_poly(sail)], True, colour, 2, cv2.LINE_AA)
    cv2.circle(img, at(0.0, -1.35), max(2, int(s * 0.11)), SPARK, -1, cv2.LINE_AA)


def star(img, x: float, y: float, r: float, colour, filled: bool = True) -> None:
    """A four-pointed star."""
    points = []
    for k in range(8):
        a = k * math.pi / 4 - math.pi / 2
        rr = r if k % 2 == 0 else r * 0.3
        points.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    if filled:
        cv2.fillPoly(img, [_poly(points)], colour, cv2.LINE_AA)
    else:
        cv2.polylines(img, [_poly(points)], True, colour, 1, cv2.LINE_AA)


def mist(img, x: float, y: float, r: float, t: float, seed: float = 0.0) -> None:
    """A bank of fog. It dims what is under it rather than covering it."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x - r * 2.4)), min(w, int(x + r * 2.4))
    y0, y1 = max(0, int(y - r * 0.8)), min(h, int(y + r * 0.8))
    if x0 >= x1 or y0 >= y1:
        return
    region = img[y0:y1, x0:x1]
    layer = region.copy()
    for k in range(5):
        cx = x + (k - 2) * r * 0.75 + math.sin(t * 0.4 + k + seed) * r * 0.1
        cy = y + math.cos(t * 0.3 + k * 1.7 + seed) * r * 0.12
        cv2.ellipse(layer, (int(cx - x0), int(cy - y0)),
                    (int(r * 0.8), int(r * (0.42 + 0.12 * (k % 2)))), 0, 0, 360,
                    dim(FOG, 0.3), -1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.65, region, 0.35, 0, region)


def cage(img, x: float, y: float, r: float, sparks: int, t: float,
         stand: float = 0.0) -> None:
    """The lamp's glass cage, with whatever light there is to keep in it."""
    ring = [(x + r * math.cos(a), y + r * math.sin(a))
            for a in (math.pi / 8 + k * math.pi / 4 for k in range(8))]
    inner = [(x + (px - x) * 0.72, y + (py - y) * 0.72) for px, py in ring]
    if stand:
        base = [(x - r * 0.45, y + r * 0.92), (x + r * 0.45, y + r * 0.92),
                (x + r * 0.7, y + stand), (x - r * 0.7, y + stand)]
        cv2.fillPoly(img, [_poly(base)], dim(STONE, 0.6), cv2.LINE_AA)
        cv2.polylines(img, [_poly(base)], True, overlay.DIM, 1, cv2.LINE_AA)
    cv2.fillPoly(img, [_poly(ring)], (20, 16, 6), cv2.LINE_AA)
    for a, b in zip(ring, inner):
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), overlay.DIM, 1, cv2.LINE_AA)
    cv2.polylines(img, [_poly(inner)], True, overlay.DIM, 1, cv2.LINE_AA)
    cv2.polylines(img, [_poly(ring)], True, overlay.GREY, 2, cv2.LINE_AA)
    for k in range(sparks):
        a = t * 0.5 + k * 2.4                   # the golden angle: evenly spread
        d = r * 0.5 * math.sqrt((k + 0.5) / max(1, sparks))
        twinkle = 0.65 + 0.35 * math.sin(t * 3.0 + k * 1.9)
        cv2.circle(img, (int(x + d * math.cos(a)), int(y + d * math.sin(a))),
                   max(2, int(r * 0.055)), dim(SPARK, twinkle), -1, cv2.LINE_AA)


def tower(img, x: float, base: float, height: float, lit: bool = False) -> Tuple[int, int]:
    """The lighthouse, standing on (x, base). Returns where its lamp is."""
    H = height

    def at(u: float, v: float) -> Tuple[int, int]:
        return int(x + u * H), int(base - v * H)

    def half(v: float) -> float:
        return 0.11 - (0.11 - 0.065) * v / 0.74

    body = [at(-0.11, 0), at(0.11, 0), at(0.065, 0.74), at(-0.065, 0.74)]
    cv2.fillPoly(img, [_poly(body)], STONE, cv2.LINE_AA)
    for lo, hi in ((0.18, 0.32), (0.48, 0.60)):
        band = [at(-half(lo), lo), at(half(lo), lo), at(half(hi), hi), at(-half(hi), hi)]
        cv2.fillPoly(img, [_poly(band)], dim(STONE, 0.45), cv2.LINE_AA)
    cv2.polylines(img, [_poly(body)], True, overlay.GREY, 1, cv2.LINE_AA)
    cv2.rectangle(img, at(-0.025, 0.09), at(0.025, 0), (0, 0, 0), -1)
    for v in (0.40, 0.66):
        cv2.rectangle(img, at(-0.012, v + 0.03), at(0.012, v), (0, 0, 0), -1)
    cv2.rectangle(img, at(-0.10, 0.775), at(0.10, 0.74), dim(STONE, 1.3), -1)
    cv2.rectangle(img, at(-0.10, 0.775), at(0.10, 0.74), overlay.GREY, 1)
    glass = dim(SPARK, 0.55) if lit else (22, 18, 8)
    cv2.rectangle(img, at(-0.05, 0.90), at(0.05, 0.775), glass, -1)
    cv2.rectangle(img, at(-0.05, 0.90), at(0.05, 0.775), overlay.GREY, 1)
    for u in (-0.017, 0.017):
        cv2.line(img, at(u, 0.90), at(u, 0.775), overlay.DIM, 1, cv2.LINE_AA)
    roof = [at(-0.065, 0.90), at(0.065, 0.90), at(0, 1.0)]
    cv2.fillPoly(img, [_poly(roof)], dim(STONE, 0.8), cv2.LINE_AA)
    cv2.polylines(img, [_poly(roof)], True, overlay.GREY, 1, cv2.LINE_AA)
    cv2.line(img, at(0, 1.0), at(0, 1.05), overlay.GREY, 1, cv2.LINE_AA)
    return at(0, 0.8375)


def pip(img, x: float, y: float, r: float, t: float) -> None:
    """Pip: a spark with a face."""
    y += r * 0.08 * math.sin(t * 2.2)
    for k in range(8):
        a = t * 0.8 + k * math.pi / 4
        r0, r1 = r * 0.72, r * (1.0 if k % 2 else 0.86)
        cv2.line(img, (int(x + r0 * math.cos(a)), int(y + r0 * math.sin(a))),
                 (int(x + r1 * math.cos(a)), int(y + r1 * math.sin(a))),
                 SPARK, 2, cv2.LINE_AA)
    c = (int(x), int(y))
    cv2.circle(img, c, int(r * 0.6), dim(SPARK, 0.5), -1, cv2.LINE_AA)
    cv2.circle(img, c, int(r * 0.6), SPARK, 2, cv2.LINE_AA)
    blink = math.sin(t * 0.9) > 0.97
    for side in (-1, 1):
        eye = (int(x + side * r * 0.22), int(y - r * 0.1))
        if blink:
            cv2.line(img, (eye[0] - int(r * 0.08), eye[1]), (eye[0] + int(r * 0.08), eye[1]),
                     (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.ellipse(img, eye, (max(2, int(r * 0.07)), max(2, int(r * 0.11))), 0, 0, 360,
                        (0, 0, 0), -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(y + r * 0.12)), (int(r * 0.22), int(r * 0.16)), 0, 20, 160,
                (0, 0, 0), 2, cv2.LINE_AA)


def maren(img, x: float, y: float, r: float, t: float) -> None:
    """Maren, the old keeper: a sou'wester, a scarf and a weathered smile."""
    line, soft = overlay.CYAN, dim(overlay.CYAN, 0.5)
    shoulders = [(x - r * 0.95, y + r), (x - r * 0.8, y + r * 0.62), (x - r * 0.28, y + r * 0.46),
                 (x + r * 0.28, y + r * 0.46), (x + r * 0.8, y + r * 0.62), (x + r * 0.95, y + r)]
    cv2.fillPoly(img, [_poly(shoulders)], dim(overlay.CYAN, 0.12), cv2.LINE_AA)
    cv2.polylines(img, [_poly(shoulders)], False, line, 2, cv2.LINE_AA)
    for k in range(3):                              # the scarf
        yy = int(y + r * (0.44 + 0.07 * k))
        cv2.line(img, (int(x - r * 0.27), yy), (int(x + r * 0.27), yy), soft, 2, cv2.LINE_AA)
    head = (int(x), int(y + r * 0.08))
    cv2.circle(img, head, int(r * 0.36), (14, 12, 4), -1, cv2.LINE_AA)
    cv2.circle(img, head, int(r * 0.36), line, 2, cv2.LINE_AA)
    crown = (int(x), int(y - r * 0.2))
    cv2.ellipse(img, crown, (int(r * 0.4), int(r * 0.42)), 0, 180, 360,
                dim(overlay.CYAN, 0.22), -1, cv2.LINE_AA)
    cv2.ellipse(img, crown, (int(r * 0.4), int(r * 0.42)), 0, 180, 360, line, 2, cv2.LINE_AA)
    cv2.ellipse(img, crown, (int(r * 0.74), int(r * 0.13)), -6, 0, 360,
                dim(overlay.CYAN, 0.22), -1, cv2.LINE_AA)
    cv2.ellipse(img, crown, (int(r * 0.74), int(r * 0.13)), -6, 0, 360, line, 2, cv2.LINE_AA)
    for side in (-1, 1):
        eye = (int(x + side * r * 0.14), int(y + r * 0.06))
        cv2.circle(img, eye, max(2, int(r * 0.035)), overlay.WHITE, -1, cv2.LINE_AA)
        cv2.line(img, (eye[0] + side * int(r * 0.08), eye[1] - int(r * 0.02)),
                 (eye[0] + side * int(r * 0.15), eye[1] - int(r * 0.06)), soft, 1, cv2.LINE_AA)
    cv2.ellipse(img, (int(x), int(y + r * 0.17)), (int(r * 0.13), int(r * 0.08)), 0, 15, 165,
                overlay.WHITE, 2, cv2.LINE_AA)


def gloam(img, x: float, y: float, r: float, t: float) -> None:
    """The Gloam itself, as it looks when it speaks."""
    gloamling(img, x, y, r * 0.8, t * 0.6, 0.0, (0.0, 0.2), eyes=3)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
class Backdrop:
    """The scenes of the story, by name. Each is rendered when first shown."""

    # name: (still picture, movement over it)
    SCENES = {
        "night": ("tower", "rain"),
        "storm": ("tower", "storm"),
        "lit": ("tower-stars", "beam"),
        "dawn": ("tower-dawn", "beam"),
        "gale": ("gale", "wind"),
        "room": ("room", "glow"),
        "fog": ("room", "creep"),
        "bay": ("bay", "swell"),
        "sky": ("sky", "twinkle"),
        "dark": ("dark", "rain"),
    }

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.unit = float(min(size))
        self._still: Dict[str, np.ndarray] = {}
        rng = random.Random(11)
        self._specks = [(rng.random(), rng.random(), rng.random()) for _ in range(90)]
        self.lamp = (0, 0)                  # where the tower's lamp is, once drawn

    def draw(self, canvas, name: str, t: float) -> None:
        still, moving = self.SCENES.get(name, self.SCENES["dark"])
        if still not in self._still:
            img = overlay.blank(self.size)
            getattr(self, "_still_" + still.split("-")[0])(img, still)
            self._still[still] = img
        np.copyto(canvas, self._still[still])
        getattr(self, "_move_" + moving)(canvas, t)

    # -- pieces -------------------------------------------------------------
    def _sky(self, img, until: float, top=(40, 18, 0)) -> None:
        w, h = self.size
        rows = int(h * until)
        fade = np.linspace(1.0, 0.25, rows)[:, None]
        img[:rows] = (fade * np.array(top, np.float64)).astype(np.uint8)[:, None, :]

    def _stars(self, img, count: int, until: float, seed: int = 3) -> None:
        w, h = self.size
        rng = random.Random(seed)
        for _ in range(count):
            x, y = rng.randrange(w), rng.randrange(max(1, int(h * until)))
            colour = dim(rng.choice((overlay.WHITE, overlay.CYAN, overlay.GREY)),
                         rng.uniform(0.25, 0.75))
            cv2.circle(img, (x, y), rng.choice((1, 1, 1, 2)), colour, -1, cv2.LINE_AA)

    def _sea(self, img, top: float, seed: int = 5) -> None:
        w, h = self.size
        y0 = int(h * top)
        img[y0:] = SEA
        rng = random.Random(seed)
        for _ in range(int(160 * (1 - top) / 0.4)):
            x, y = rng.randrange(w), rng.randrange(y0 + 6, h)
            length = int(self.unit * rng.uniform(0.015, 0.05) * (0.4 + (y - y0) / (h - y0)))
            cv2.line(img, (x, y), (x + length, y), WAVE, 1, cv2.LINE_AA)
        cv2.line(img, (0, y0), (w, y0), dim(WAVE, 1.3), 1, cv2.LINE_AA)

    def _rock(self, img, x: float, top: float, width: float, seed: int = 9) -> None:
        w, h = self.size
        rng = random.Random(seed)
        ridge = []
        steps = 9
        for k in range(steps + 1):
            u = k / steps
            lift = math.sin(u * math.pi) ** 0.6
            ridge.append((x - width + 2 * width * u,
                          top + (1 - lift) * self.unit * 0.16 + rng.uniform(-1, 1) * self.unit * 0.012))
        cv2.fillPoly(img, [_poly(ridge)], dim(STONE, 0.6), cv2.LINE_AA)
        cv2.polylines(img, [_poly(ridge)], False, overlay.DIM, 2, cv2.LINE_AA)

    # -- still pictures -----------------------------------------------------
    def _still_tower(self, img, name: str) -> None:
        w, h = self.size
        dawn, clear = name.endswith("dawn"), name != "tower"
        self._sky(img, 0.66, (58, 34, 4) if dawn else (40, 18, 0))
        self._stars(img, 30 if dawn else 110 if clear else 12, 0.5)
        if dawn:
            # Morning, without a trace of red in it: a pale band on the water.
            rows = int(h * 0.2)
            y1 = int(h * 0.66)
            glow = (np.linspace(0.0, 1.0, rows) ** 2)[:, None] * np.array((120, 110, 50.0))
            band = img[y1 - rows:y1].astype(np.float64) + glow[:, None, :]
            img[y1 - rows:y1] = np.clip(band, 0, 255).astype(np.uint8)
        self._sea(img, 0.66)
        self._rock(img, w * 0.70, h * 0.62, self.unit * 0.30)
        self.lamp = tower(img, w * 0.70, h * 0.635, h * 0.46, lit=clear)

    def _still_gale(self, img, name: str) -> None:
        w, h = self.size
        self._sky(img, 0.9)
        self._stars(img, 10, 0.6, seed=4)
        self._sea(img, 0.9)
        # The gallery rail the keeper is standing at.
        y = int(h * 0.955)
        cv2.line(img, (0, y), (w, y), overlay.DIM, 3, cv2.LINE_AA)
        for x in range(0, w, int(self.unit * 0.09)):
            cv2.line(img, (x, y), (x, h), overlay.DIM, 2, cv2.LINE_AA)

    def _still_room(self, img, name: str) -> None:
        w, h = self.size
        floor = int(h * 0.82)
        self._sky(img, 0.82, (34, 15, 0))
        self._stars(img, 16, 0.7, seed=6)
        img[floor:] = dim(STONE, 0.4)
        for k in range(-6, 7):                      # floorboards, running away
            cv2.line(img, (w // 2 + int(k * w * 0.05), floor),
                     (w // 2 + int(k * w * 0.13), h), dim(STONE, 0.75), 1, cv2.LINE_AA)
        # The lamp room is glass all round: the panes' frames.
        for k in range(9):
            x = int(w * k / 8)
            cv2.line(img, (x, 0), (x, floor), dim(STONE, 0.9), 5, cv2.LINE_AA)
        for y in (int(h * 0.05), floor):
            cv2.line(img, (0, y), (w, y), dim(STONE, 1.1), 5, cv2.LINE_AA)
        cage(img, w / 2, h / 2, self.unit * 0.13, 0, 0.0, stand=floor - h / 2)

    def _still_bay(self, img, name: str) -> None:
        w, h = self.size
        self._sky(img, HORIZON)
        self._stars(img, 14, HORIZON * 0.9, seed=8)
        # The headland, and the lighthouse on it with its lamp out.
        self._sea(img, HORIZON)
        head = [(w * 0.60, h * HORIZON), (w * 0.74, h * (HORIZON - 0.035)),
                (w * 0.90, h * (HORIZON - 0.05)), (w, h * (HORIZON - 0.03)), (w, h * HORIZON)]
        cv2.fillPoly(img, [_poly(head)], dim(STONE, 0.7), cv2.LINE_AA)
        tower(img, w * 0.88, h * (HORIZON - 0.045), h * 0.12)
        # The Teeth.
        rng = random.Random(21)
        x = w * REEF_X
        y = h * (HORIZON + 0.07)
        while y < h * 0.96:
            size = self.unit * rng.uniform(0.028, 0.05)
            if not h * CHANNEL[0] - size < y < h * CHANNEL[1] + size * 0.3:
                lean = rng.uniform(-0.4, 0.4) * size
                tooth = [(x - size * 0.7, y), (x + lean, y - size * 1.5), (x + size * 0.7, y)]
                cv2.fillPoly(img, [_poly(tooth)], STONE, cv2.LINE_AA)
                cv2.polylines(img, [_poly(tooth)], True, overlay.GREY, 1, cv2.LINE_AA)
                cv2.line(img, (int(x - size), int(y + 2)), (int(x + size), int(y + 2)),
                         dim(overlay.WHITE, 0.3), 1, cv2.LINE_AA)
            y += size * 1.25
        for edge in CHANNEL:                        # the buoys marking the gap
            c = (int(x), int(h * edge))
            cv2.line(img, c, (c[0], c[1] - int(self.unit * 0.03)), overlay.GREEN, 2, cv2.LINE_AA)
            cv2.circle(img, (c[0], c[1] - int(self.unit * 0.034)), 5, overlay.GREEN, -1, cv2.LINE_AA)
        # The harbour wall, and the houses behind it.
        x0 = int(w * HARBOUR_X)
        mid = h * sum(CHANNEL) / 2
        cv2.rectangle(img, (x0, int(mid - h * 0.16)), (w, int(mid + h * 0.16)), dim(STONE, 0.7), -1)
        cv2.rectangle(img, (x0, int(mid - h * 0.16)), (w, int(mid + h * 0.16)), overlay.GREY, 1)
        for k in range(4):
            yy = int(mid - h * 0.12 + k * h * 0.07)
            cv2.rectangle(img, (x0 + 14, yy), (x0 + 14 + int(self.unit * 0.035), yy + int(h * 0.035)),
                          (0, 0, 0), -1)
            cv2.rectangle(img, (x0 + 22, yy + 8), (x0 + 30, yy + 16), SPARK, -1)

    def _still_sky(self, img, name: str) -> None:
        w, h = self.size
        self._sky(img, 1.0, (30, 12, 0))
        self._stars(img, 260, 1.0, seed=13)
        # The hole in the clouds: what is left of them, round the edges.
        rng = random.Random(14)
        for _ in range(46):
            a = rng.uniform(0, 2 * math.pi)
            x = w / 2 + math.cos(a) * w * rng.uniform(0.50, 0.62)
            y = h / 2 + math.sin(a) * h * rng.uniform(0.50, 0.66)
            axes = (int(self.unit * rng.uniform(0.10, 0.2)), int(self.unit * rng.uniform(0.05, 0.1)))
            cv2.ellipse(img, (int(x), int(y)), axes, 0, 0, 360, (26, 22, 14), -1, cv2.LINE_AA)
            cv2.ellipse(img, (int(x), int(y)), axes, 0, 0, 360, (44, 38, 24), 1, cv2.LINE_AA)

    def _still_dark(self, img, name: str) -> None:
        self._sky(img, 1.0, (20, 9, 0))

    # -- movement -----------------------------------------------------------
    def _move_rain(self, canvas, t: float, lightning: bool = True) -> None:
        w, h = self.size
        for u, v, s in self._specks[:70]:
            fall = (v + t * (0.9 + s)) % 1.0
            x, y = int((u + fall * 0.25) % 1.0 * w), int(fall * h)
            d = int(self.unit * (0.012 + 0.014 * s))
            cv2.line(canvas, (x, y), (x + d // 3, y + d), (70, 52, 22), 1, cv2.LINE_AA)
        if not lightning:
            return
        k = (t % 6.5) / 0.22
        if k < 1.0:                                 # a flash, kept blue and brief
            cv2.add(canvas, dim((46, 30, 6), 1.0 - k) + (0,), dst=canvas)
            rng = random.Random(int(t / 6.5))
            x, y, bolt = w * rng.uniform(0.1, 0.5), 0.0, []
            while y < h * 0.5:
                bolt.append((x, y))
                x += self.unit * rng.uniform(-0.05, 0.05)
                y += self.unit * rng.uniform(0.03, 0.08)
            cv2.polylines(canvas, [_poly(bolt)], False, dim(overlay.WHITE, 1.0 - k), 2, cv2.LINE_AA)

    def _move_wind(self, canvas, t: float) -> None:
        w, h = self.size
        for u, v, s in self._specks[:46]:
            x = int((u + t * (0.25 + 0.35 * s)) % 1.0 * (w * 1.2) - w * 0.1)
            y = int(v * h * 0.86 + math.sin(t * 2 + u * 20) * self.unit * 0.008)
            cv2.line(canvas, (x, y), (x + int(self.unit * (0.03 + 0.07 * s)), y),
                     (60, 44, 18), 1, cv2.LINE_AA)

    def _move_glow(self, canvas, t: float) -> None:
        w, h = self.size
        cage(canvas, w / 2, h / 2, self.unit * 0.13, 9, t)

    def _move_creep(self, canvas, t: float) -> None:
        w, h = self.size
        cage(canvas, w / 2, h / 2, self.unit * 0.13, 9, t)
        for k, (u, v, s) in enumerate(self._specks[:11]):
            # Up from the stairs, at the sides, and never quite arriving.
            side = -1 if k % 2 else 1
            x = w / 2 + side * w * (0.2 + 0.28 * u) + math.sin(t * 0.7 + k) * self.unit * 0.03
            y = h * (0.2 + 0.6 * v) + math.sin(t * 0.5 + k * 2) * self.unit * 0.04
            gloamling(canvas, x, y, self.unit * (0.035 + 0.05 * s), t, k * 1.7,
                      (w / 2 - x, h / 2 - y), strength=0.45 + 0.4 * s)

    def _move_swell(self, canvas, t: float) -> None:
        w, h = self.size
        top = h * HORIZON
        for u, v, s in self._specks[:40]:
            y = int(top + 12 + v * (h - top - 24))
            x = int((u + 0.012 * math.sin(t * 0.8 + v * 30)) * w)
            glint = 0.5 + 0.5 * math.sin(t * 1.3 + u * 40)
            cv2.line(canvas, (x, y), (x + int(self.unit * 0.03 * (0.5 + s)), y),
                     dim(WAVE, 0.6 + glint), 1, cv2.LINE_AA)

    def _move_twinkle(self, canvas, t: float) -> None:
        w, h = self.size
        for u, v, s in self._specks[:30]:
            k = 0.5 + 0.5 * math.sin(t * (1.0 + 2.5 * s) + u * 50)
            cv2.circle(canvas, (int(u * w), int(v * h)), 2, dim(overlay.WHITE, 0.25 + 0.6 * k),
                       -1, cv2.LINE_AA)

    def _move_storm(self, canvas, t: float) -> None:
        self._move_rain(canvas, t, lightning=False)
        x, y = self.lamp
        r = self.unit * 0.2
        for k in range(7):                          # the fog, wound round the tower
            a = t * 0.35 + k * 2 * math.pi / 7
            gloamling(canvas, x + math.cos(a) * r * 1.5, y + math.sin(a) * r * 0.75,
                      r * 0.42, t, k * 1.3, (-math.cos(a), -math.sin(a)), eyes=0, strength=0.55)
        gloamling(canvas, x, y - r * 0.15, r, t * 0.5, 0.0, (-0.5, 0.6), eyes=1)

    def _move_beam(self, canvas, t: float) -> None:
        w, h = self.size
        x, y = self.lamp
        turn = math.cos(t * 0.55)                   # the lamp going round, seen side on
        layer = np.zeros_like(canvas)
        reach, spread = w * 0.9 * turn, self.unit * (0.05 + 0.10 * abs(turn))
        for side in (1, -1):
            beam = [(x, y - 4), (x + side * reach, y - spread), (x + side * reach, y + spread),
                    (x, y + 4)]
            cv2.fillPoly(layer, [_poly(beam)], dim((70, 92, 44), 0.35 + 0.65 * abs(turn)),
                         cv2.LINE_AA)
        cv2.add(canvas, layer, dst=canvas)
        flare = 1.0 - abs(turn)                     # pointing this way: a glare
        cv2.circle(canvas, (x, y), int(self.unit * (0.012 + 0.03 * flare)),
                   dim(SPARK, 0.5 + 0.5 * flare), -1, cv2.LINE_AA)
