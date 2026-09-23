"""The figures of the range duel: armed ones to shoot, and people to spare.

Drawn rather than loaded, like every other picture here, and in the same
palette for the same reason (see overlay.py): no red, and nothing large is
bright. So these are range targets in outline - dim clothes, pale edges - and
the one bright thing on each is what it holds. That is how they are told
apart, as on a real shoot / no-shoot range - by the hands - though a face
helps: the armed scowl, behind a mask or dark glasses, and the rest do not.

A figure is a head-to-hips cut-out in units of its own height: u across from
the centre line, v down from the top. Besides the picture, rendering gives a
map of whose pixels are whose, because one target has both kinds on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Sequence, Tuple

import cv2
import numpy as np

from . import overlay

FOE, FRIEND = 1, 2                  # whose pixel it is; 0 is the air around them
ASPECT = 0.76                       # width of a figure, in heights
SCALE = 2                           # drawn this much larger, then shrunk: smooth edges

EDGE = (205, 205, 178)
SKIN = (118, 140, 122)
DARK = (44, 44, 38)                 # what the armed wear
MASK = (22, 22, 18)                 # a balaclava; and every dark detail
STEEL = (232, 232, 204)             # a weapon
VEST = (62, 68, 50)
DRESS = (112, 88, 20)
HAIR = (56, 118, 104)               # fair: dark hair would be lost in the dark
CROP = (40, 40, 34)                 # a man's short hair
JACKET = (124, 98, 44)
SHIRT = (88, 112, 66)
COAT = (118, 128, 110)              # a medic's: as near white as a large thing may be
BAG = (60, 92, 64)
SCREEN = (230, 200, 56)             # a phone, lit

Point = Tuple[float, float]


class _Pen:
    """Draws in figure units, and notes whose pixels it has drawn."""

    def __init__(self, height: int) -> None:
        self.h = height * SCALE
        self.w = int(ASPECT * height) * SCALE
        self.img = np.zeros((self.h, self.w, 3), np.uint8)
        self.zone = np.zeros((self.h, self.w), np.uint8)
        self.side = FOE
        self.dx = 0.0                           # where the figure being drawn stands
        self.edge = max(2, int(round(0.007 * self.h)))

    def at(self, u: float, v: float) -> Tuple[int, int]:
        return int(round(self.w / 2 + (u + self.dx) * self.h)), int(round(v * self.h))

    def px(self, k: float) -> int:
        return max(1, int(round(k * self.h)))

    def _points(self, points: Sequence[Point]) -> np.ndarray:
        return np.array([self.at(u, v) for u, v in points], np.int32).reshape(-1, 1, 2)

    def poly(self, points: Sequence[Point], fill, edge=EDGE) -> None:
        p = self._points(points)
        for img, a, b in ((self.img, fill, edge), (self.zone, self.side, self.side)):
            cv2.fillPoly(img, [p], a, cv2.LINE_AA if img is self.img else cv2.LINE_8)
            cv2.polylines(img, [p], True, b, self.edge, cv2.LINE_AA if img is self.img
                          else cv2.LINE_8)

    def oval(self, centre: Point, radii: Point, fill, edge=EDGE, angle: float = 0.0,
             start: float = 0.0, end: float = 360.0) -> None:
        c, r = self.at(*centre), (self.px(radii[0]), self.px(radii[1]))
        cv2.ellipse(self.img, c, r, angle, start, end, fill, -1, cv2.LINE_AA)
        cv2.ellipse(self.zone, c, r, angle, start, end, self.side, -1)
        if edge is not None:
            cv2.ellipse(self.img, c, r, angle, start, end, edge, self.edge, cv2.LINE_AA)
            cv2.ellipse(self.zone, c, r, angle, start, end, self.side, self.edge)

    def limb(self, points: Sequence[Point], width: float, fill, edge=EDGE) -> None:
        """An arm, a strap, a barrel: a thick line with an outline round it."""
        p, w = self._points(points), self.px(width)
        if edge is not None:
            cv2.polylines(self.img, [p], False, edge, w + 2 * self.edge, cv2.LINE_AA)
        cv2.polylines(self.img, [p], False, fill, w, cv2.LINE_AA)
        cv2.polylines(self.zone, [p], False, self.side, w + 2 * self.edge)

    def mark(self, points: Sequence[Point], colour, width: float = 0.0) -> None:
        """A detail drawn on something: it changes no one's outline."""
        cv2.polylines(self.img, [self._points(points)], False, colour,
                      self.px(width) if width else self.edge, cv2.LINE_AA)

    def dot(self, centre: Point, radius: float, colour) -> None:
        cv2.circle(self.img, self.at(*centre), self.px(radius), colour, -1, cv2.LINE_AA)

    def arc(self, centre: Point, radii: Point, start: float, end: float, colour,
            width: float = 0.0) -> None:
        cv2.ellipse(self.img, self.at(*centre), (self.px(radii[0]), self.px(radii[1])), 0,
                    start, end, colour, self.px(width) if width else self.edge, cv2.LINE_AA)


# -- the parts --------------------------------------------------------------
SHOULDER = 0.365                    # how far down the arms begin
HEAD = ((0, 0.14), (0.088, 0.108))


def _bar(a: Point, b: Point, half: float) -> Sequence[Point]:
    """The rectangle `half` to either side of the line from a to b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    k = half / ((dx * dx + dy * dy) ** 0.5 or 1.0)
    nx, ny = -dy * k, dx * k
    return [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny),
            (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]


def _torso(pen: _Pen, cloth, shoulder: float = 0.21, waist: float = 0.17,
           hip: float = 0.19) -> None:
    pen.poly([(-0.034, 0.22), (0.034, 0.22), (0.04, 0.31), (-0.04, 0.31)], SKIN)   # the neck
    right = [(0.07, 0.282), (shoulder, 0.335), (shoulder + 0.012, 0.42),
             (waist, 0.70), (hip, 1.02)]
    pen.poly(right + [(-u, v) for u, v in reversed(right)], cloth)


def _face(pen: _Pen, mood: str = "smile") -> None:
    """Someone whose face can be seen - and what it says."""
    pen.oval(*HEAD, SKIN)
    for side in (-1, 1):
        pen.dot((side * 0.035, 0.132), 0.012, MASK)
        if mood == "scowl":
            pen.mark([(side * 0.062, 0.092), (side * 0.014, 0.112)], MASK, 0.012)
        else:
            pen.arc((side * 0.035, 0.112), (0.02, 0.012), 200, 340, MASK)
    if mood == "smile":
        pen.arc((0, 0.178), (0.034, 0.022), 15, 165, MASK, 0.01)
    elif mood == "scowl":
        pen.arc((0, 0.212), (0.03, 0.02), 200, 340, MASK, 0.01)
    else:                                                       # afraid
        pen.oval((0, 0.195), (0.015, 0.018), MASK, None)


def _balaclava(pen: _Pen) -> None:
    pen.oval(*HEAD, MASK)
    pen.oval((0, 0.13), (0.064, 0.03), SKIN, None)
    for side in (-1, 1):
        pen.dot((side * 0.03, 0.134), 0.011, MASK)
        pen.mark([(side * 0.07, 0.098), (side * 0.01, 0.118)], MASK, 0.02)     # a scowl


def _crop(pen: _Pen) -> None:
    pen.oval((0, 0.09), (0.09, 0.06), CROP, EDGE, 0, 180, 360)


def _arm(pen: _Pen, side: int, cloth, elbow: Point, hand: Point, shoulder: float = 0.21,
         width: float = 0.07, fist: float = 0.038) -> None:
    """From the shoulder on `side` (1: the right of the picture) to a hand."""
    pen.limb([(side * (shoulder - 0.012), SHOULDER), (side * elbow[0], elbow[1]),
              (side * hand[0], hand[1])], width, cloth)
    if fist:
        pen.oval((side * hand[0], hand[1]), (fist, fist), SKIN)


def _arms_down(pen: _Pen, cloth, shoulder: float = 0.21, width: float = 0.07,
               sides: Sequence[int] = (-1, 1)) -> None:
    for side in sides:
        _arm(pen, side, cloth, (shoulder + 0.04, 0.62), (shoulder + 0.03, 0.85),
             shoulder, width)


def _pistol(pen: _Pen, at: Point, aim: Point, down: Point, length: float = 0.25) -> None:
    """A pistol in profile: the slide from `at` along `aim`, the grip from
    `at` along `down` (both unit vectors), and a hand closed on the grip."""
    (gx, gy), (ax, ay), (nx, ny) = at, aim, down

    def on(along: float, below: float) -> Point:
        return gx + ax * along + nx * below, gy + ay * along + ny * below

    pen.poly(_bar(on(-0.005, 0.0), on(-0.04, 0.14), 0.027), STEEL)
    pen.poly(_bar(on(-0.05, 0.0), on(length, 0.0), 0.031), STEEL)
    pen.mark([on(0.02, 0.03), on(0.075, 0.07), on(0.02, 0.078)], STEEL, 0.012)
    pen.mark([on(0.0, -0.008), on(length - 0.03, -0.008)], MASK)
    pen.dot(on(length - 0.022, 0.012), 0.01, MASK)
    pen.oval(on(-0.022, 0.085), (0.04, 0.04), SKIN)


def _cross(pen: _Pen, centre: Point, size: float, colour=overlay.GREEN) -> None:
    u, v = centre
    pen.mark([(u - size, v), (u + size, v)], colour, size * 0.75)
    pen.mark([(u, v - size), (u, v + size)], colour, size * 0.75)


# -- the armed --------------------------------------------------------------
def _gunman(pen: _Pen) -> None:
    """Masked, a pistol held across the chest."""
    _torso(pen, DARK)
    _balaclava(pen)
    _arms_down(pen, DARK, sides=(-1,))
    _arm(pen, 1, DARK, (0.29, 0.61), (0.085, 0.565), fist=0)
    _pistol(pen, (0.09, 0.48), (-0.995, -0.1), (-0.1, 0.995), 0.27)


def _rifleman(pen: _Pen) -> None:
    """Masked, in a vest, a rifle slanted across the body."""
    _torso(pen, DARK)
    pen.poly([(-0.14, 0.35), (0.14, 0.35), (0.155, 0.69), (-0.155, 0.69)], VEST)
    for u in (-0.075, 0.075):
        pen.poly(_bar((u, 0.74), (u, 0.85), 0.05), VEST)
    _balaclava(pen)
    _arm(pen, -1, DARK, (0.29, 0.58), (0.10, 0.64), fist=0)
    _arm(pen, 1, DARK, (0.295, 0.53), (0.125, 0.46), fist=0)
    # The rifle: stock, grip, the curved magazine, body, barrel and its sight.
    pen.poly([(-0.34, 0.735), (-0.295, 0.81), (-0.12, 0.665), (-0.16, 0.60)], STEEL)
    pen.poly(_bar((-0.085, 0.615), (-0.04, 0.705), 0.023), STEEL)
    pen.poly([(-0.012, 0.565), (0.04, 0.525), (0.105, 0.60), (0.15, 0.635), (0.125, 0.69),
              (0.06, 0.655)], STEEL)
    pen.poly(_bar((-0.16, 0.64), (0.135, 0.42), 0.037), STEEL)
    pen.poly(_bar((0.135, 0.42), (0.335, 0.27), 0.014), STEEL)
    pen.poly(_bar((0.29, 0.305), (0.272, 0.28), 0.01), STEEL)
    pen.mark([(-0.13, 0.605), (0.12, 0.42)], MASK)
    pen.oval((-0.09, 0.64), (0.04, 0.04), SKIN)
    pen.oval((0.12, 0.455), (0.04, 0.04), SKIN)


def _thug(pen: _Pen) -> None:
    """No mask: a cap, dark glasses, and a knife held up to strike."""
    _torso(pen, DARK)
    pen.mark([(0, 0.32), (0, 1.0)], EDGE)                       # an open jacket
    _face(pen, "scowl")
    pen.mark([(-0.066, 0.132), (0.066, 0.132)], MASK, 0.036)    # the glasses
    pen.oval((0, 0.105), (0.094, 0.078), DARK, EDGE, 0, 180, 360)
    pen.limb([(-0.095, 0.105), (0.125, 0.105)], 0.014, DARK)    # the peak of the cap
    _arms_down(pen, DARK, sides=(-1,))
    _arm(pen, 1, DARK, (0.305, 0.52), (0.27, 0.32), fist=0)
    pen.poly([(0.238, 0.255), (0.30, 0.255), (0.305, 0.12), (0.262, 0.02), (0.238, 0.12)],
             STEEL)
    pen.mark([(0.265, 0.24), (0.265, 0.08)], MASK)
    pen.poly(_bar((0.215, 0.262), (0.325, 0.262), 0.012), STEEL)
    pen.oval((0.27, 0.305), (0.042, 0.042), SKIN)


# -- the unarmed ------------------------------------------------------------
def _hair(pen: _Pen) -> None:
    pen.poly([(-0.11, 0.14), (0.11, 0.14), (0.15, 0.31), (0.115, 0.39), (-0.115, 0.39),
              (-0.15, 0.31)], HAIR)
    pen.oval((0, 0.14), (0.118, 0.136), HAIR)
    pen.oval((0, 0.21), (0.108, 0.13), HAIR, None)


def _woman_body(pen: _Pen, mood: str) -> None:
    _hair(pen)
    _torso(pen, DRESS, shoulder=0.175, waist=0.118, hip=0.225)
    pen.poly([(-0.05, 0.285), (0.05, 0.285), (0, 0.37)], SKIN)      # the neckline
    pen.mark([(-0.118, 0.70), (0.118, 0.70)], EDGE, 0.014)          # a belt
    _face(pen, mood)
    pen.oval((0, 0.10), (0.09, 0.07), HAIR, EDGE, 0, 180, 360)      # the fringe


def _woman(pen: _Pen) -> None:
    """Long hair, a dress, a handbag on its strap."""
    _woman_body(pen, "smile")
    _arms_down(pen, DRESS, shoulder=0.175, width=0.058)
    pen.mark([(0.13, 0.335), (-0.18, 0.72)], BAG, 0.018)
    pen.poly([(-0.30, 0.71), (-0.10, 0.71), (-0.115, 0.90), (-0.285, 0.90)], BAG)
    pen.mark([(-0.295, 0.775), (-0.105, 0.775)], EDGE)
    pen.dot((-0.20, 0.80), 0.014, STEEL)


def _caller(pen: _Pen) -> None:
    """A man reading his phone: its lit screen is all he holds."""
    _torso(pen, JACKET)
    pen.mark([(0, 0.32), (0, 1.0)], EDGE)
    _face(pen)
    _crop(pen)
    _arms_down(pen, JACKET, sides=(-1,))
    _arm(pen, 1, JACKET, (0.29, 0.62), (0.115, 0.56))
    pen.poly(_bar((0.085, 0.365), (0.105, 0.54), 0.052), MASK)
    pen.poly(_bar((0.088, 0.385), (0.103, 0.515), 0.038), SCREEN, None)
    pen.oval((0.125, 0.545), (0.04, 0.04), SKIN)


def _surrender(pen: _Pen) -> None:
    """Both hands up, and nothing in them."""
    _torso(pen, SHIRT)
    _face(pen, "afraid")
    _crop(pen)
    for side in (-1, 1):
        _arm(pen, side, SHIRT, (0.315, 0.40), (0.30, 0.19), fist=0)
        for k in (-1.5, -0.5, 0.5, 1.5):                        # the fingers, spread
            pen.limb([(side * 0.30 + k * 0.02, 0.14), (side * 0.30 + k * 0.034, 0.06)],
                     0.014, SKIN)
        pen.oval((side * 0.30, 0.16), (0.046, 0.046), SKIN)


def _medic(pen: _Pen) -> None:
    """A pale coat, a cap and a bag, each with a cross on it."""
    _torso(pen, COAT)
    pen.arc((0, 0.33), (0.07, 0.15), 10, 170, MASK, 0.012)      # a stethoscope
    pen.dot((0.0, 0.48), 0.02, STEEL)
    _cross(pen, (0.10, 0.60), 0.05)
    _face(pen)
    pen.oval((0, 0.10), (0.094, 0.075), COAT, EDGE, 0, 180, 360)
    _cross(pen, (0, 0.062), 0.022)
    _arms_down(pen, COAT)
    pen.poly([(0.11, 0.84), (0.34, 0.84), (0.34, 1.02), (0.11, 1.02)], STEEL)
    pen.mark([(0.18, 0.84), (0.18, 0.80), (0.27, 0.80), (0.27, 0.84)], EDGE, 0.012)
    _cross(pen, (0.225, 0.935), 0.045)


# -- both at once -----------------------------------------------------------
def _hostage(pen: _Pen) -> None:
    """A gunman behind the woman he is holding: only what shows of him is
    his, and a shot that lands on her is a shot at her."""
    pen.dx, pen.side = 0.105, FOE
    _torso(pen, DARK)
    _balaclava(pen)
    pen.dx, pen.side = -0.13, FRIEND
    _woman_body(pen, "afraid")
    _arms_down(pen, DRESS, shoulder=0.175, width=0.058)
    pen.dx, pen.side = 0.105, FOE
    _arm(pen, 1, DARK, (0.215, 0.58), (0.10, 0.50), fist=0)
    _pistol(pen, (0.105, 0.42), (-0.8, -0.6), (-0.6, 0.8), 0.19)
    pen.dx = 0.0


@dataclass(frozen=True)
class Look:
    name: str
    foe: bool
    draw: Callable[[_Pen], None]
    mixed: bool = False             # someone to spare stands in the way


LOOKS: Dict[str, Look] = {look.name: look for look in (
    Look("gunman", True, _gunman),
    Look("rifleman", True, _rifleman),
    Look("thug", True, _thug),
    Look("hostage", True, _hostage, mixed=True),
    Look("woman", False, _woman),
    Look("caller", False, _caller),
    Look("surrender", False, _surrender),
    Look("medic", False, _medic),
)}
FOES = tuple(l.name for l in LOOKS.values() if l.foe and not l.mixed)
FRIENDS = tuple(l.name for l in LOOKS.values() if not l.foe)
MIXED = tuple(l.name for l in LOOKS.values() if l.mixed)


@lru_cache(maxsize=64)
def render(name: str, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """The figure `height` pixels tall, and whose pixel each of its pixels is."""
    look = LOOKS[name]
    pen = _Pen(height)
    pen.side = FOE if look.foe else FRIEND
    look.draw(pen)
    size = (pen.w // SCALE, pen.h // SCALE)
    img = cv2.resize(pen.img, size, interpolation=cv2.INTER_AREA)
    zone = cv2.resize(pen.zone, size, interpolation=cv2.INTER_NEAREST)
    return img, zone
