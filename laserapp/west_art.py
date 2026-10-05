"""The pictures of the Wild West game: CC0 sprites and photographs from
OpenGameArt.org (assets/west/CREDITS.md), recoloured as they are loaded.

They come as published, which is no good on this screen as it is: brown
hats, skin, red bandanas, a sunlit town - all of it red to the camera, which
would take it for the laser (see overlay.py). So:

* the photographs become a town by moonlight: grey, tinted blue, and dim,
  because a dot on a lit patch has nothing to stand out against;
* the sprites keep their shapes and their shading, but every hue is moved
  into the half of the colour circle where red is under green - red to deep
  blue, brown to steel blue, yellow to lime - and nothing is let get bright.

A character is drawn at whatever height the place it stands in calls for,
scaled up pixel by pixel, which is how pixel art is meant to be scaled.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets" / "west"
BRIGHT = 110                    # no channel of a sprite above this
SATURATION = 0.3                # and no colour strong: a red dot is found by red over
                                # the stronger of green and blue, and on a pixel strong
                                # in either a weak dot has none to show
SLACK = 14                      # px around a figure that still count as on it: the
                                # calibration is a few px out, and hands shake
EDGE = (120, 118, 92)           # a pale rim round every figure: their own dark
                                # outlines would be lost in the night

# Hue in OpenCV's units (0-180): where each part of the circle goes.
_FROM = [0, 6, 14, 22, 30, 40, 85, 130, 170, 180]
_TO = [116, 108, 98, 90, 36, 50, 78, 104, 112, 116]


def recolour(bgr: np.ndarray) -> np.ndarray:
    """The same picture with no red in it, and nothing bright."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (h < 25) & ((s < 150) | (v > 170))   # skin and pale wood stay pale
    hsv = np.dstack([np.interp(h, _FROM, _TO), np.where(warm, s * 0.3, s * SATURATION),
                     np.minimum(v * 0.8, BRIGHT)])
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out[..., 2] = np.minimum(out[..., 2], out[..., 1] * 0.85)
    return out.astype(np.uint8)


def moonlight(bgr: np.ndarray) -> np.ndarray:
    """A photograph as the town at night: grey, blue, and dim."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = g ** 1.2 * 80.0
    return np.clip(np.dstack([g, g * 0.9, g * 0.62]), 0, 255).astype(np.uint8)


# -- characters -------------------------------------------------------------
Frame = Tuple[np.ndarray, np.ndarray]       # picture, and where it is solid


def _cowboy(name: str, pose: str) -> List[Frame]:
    frames = []
    for i in range(4):
        img = cv2.imread(str(ASSETS / name / f"{pose}_{i}.png"), cv2.IMREAD_UNCHANGED)
        frames.append((img[..., :3], img[..., 3] > 127))
    return frames


def _townsfolk(name: str) -> List[Frame]:
    frames = []
    for path in sorted((ASSETS / "townsfolk").glob(f"{name}_*.png"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        frames.append((img[..., :3], img[..., 3] > 127))
    return frames


@dataclass(frozen=True)
class Look:
    armed: bool
    source: str                     # its folder (a bandit) or its name (townsfolk)


# Who is who must be plain at a glance, so the two sides share nothing: the
# bandits are stocky cowboys with a gun held out, the townsfolk tall people
# by another artist, empty-handed - and nobody appears on both sides.
LOOKS: Dict[str, Look] = {
    "gunslinger": Look(True, "cowboy2"),
    "outlaw": Look(True, "cowboy4"),         # the one with the bandana over his face
    "oldman": Look(False, "oldman"),
    "gentleman": Look(False, "gentleman"),
    "butcher": Look(False, "butcher"),
    "lady": Look(False, "lady"),
}
BANDITS = [n for n, l in LOOKS.items() if l.armed]
TOWNSFOLK = [n for n, l in LOOKS.items() if not l.armed]
BOSS = "gunslinger"


@lru_cache(maxsize=None)
def frames(look: str, pose: str) -> Tuple[Frame, ...]:
    """The recoloured frames of `look` in `pose`: "stand" (waiting, gun out if
    armed), "shoot", "walk" (a bandit crossing the street, gun out), or
    "idle" (hands empty: a boss before the draw)."""
    l = LOOKS[look]
    if l.armed:
        raw = _cowboy(l.source, {"stand": "gun", "shoot": "shoot", "idle": "idle",
                                 "walk": "walk"}[pose])
    else:
        raw = _townsfolk(l.source)
    return tuple((recolour(img), solid) for img, solid in raw)


@lru_cache(maxsize=None)
def body(look: str) -> Tuple[int, int, int, int]:
    """The box round the figure in its frames: x0, y0, x1, y1."""
    solid = np.zeros_like(frames(look, "stand")[0][1])
    for pose in (("stand", "idle") if LOOKS[look].armed else ("stand",)):
        for _, s in frames(look, pose):
            solid |= s
    ys, xs = np.nonzero(solid)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class Sprite:
    img: np.ndarray
    solid: np.ndarray
    near: np.ndarray            # solid, and a little round it: what counts as a hit
    fx: int                     # where the figure's feet are, in the sprite
    fy: int
    muzzle: Tuple[int, int]     # the end of the gun
    box: Tuple[int, int, int, int]  # round the figure itself: x0, y0, x1, y1


@lru_cache(maxsize=512)
def sprite(look: str, pose: str, index: int, height: int, flip: bool) -> Sprite:
    """Frame `index` of `pose`, scaled so the figure is `height` px tall."""
    img, solid = frames(look, pose)[index % len(frames(look, pose))]
    x0, y0, x1, y1 = body(look)
    k = max(1, int(round(height / float(y1 - y0))))
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    solid = cv2.resize(solid.astype(np.uint8), None, fx=k, fy=k,
                       interpolation=cv2.INTER_NEAREST) > 0
    fx, fy = (x0 + x1) * k // 2, y1 * k
    rim = max(2, k // 3)
    grown = cv2.dilate(solid.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim + 1,) * 2)) > 0
    img = img.copy()
    img[grown & ~solid] = EDGE
    solid = grown
    if flip:
        img, solid = img[:, ::-1], solid[:, ::-1]
        fx = img.shape[1] - fx
    near = cv2.dilate(solid.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * SLACK + 1, 2 * SLACK + 1))) > 0
    # The gun's end: the solid pixel furthest out on the side it faces, in
    # the upper half of the figure.
    ys, xs = np.nonzero(solid[:fy - (y1 - y0) * k // 3])
    i = int(np.argmin(xs)) if flip else int(np.argmax(xs))
    muzzle = (int(xs[i]), int(ys[i]))
    ys, xs = np.nonzero(solid)
    return Sprite(np.ascontiguousarray(img), np.ascontiguousarray(solid), near, fx, fy,
                  muzzle, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def blit(canvas, s: Sprite, x: float, foot: float, clip: Optional[float] = None,
         rise: float = 1.0) -> None:
    """Put a sprite down with its feet at (x, foot). Nothing of it shows
    below `clip` (a window sill, a barrel); `rise` < 1 has it still coming up
    from behind that, or out of the ground."""
    h, w = s.img.shape[:2]
    left = int(round(x)) - s.fx
    if clip is not None:
        top = int(round(foot)) - s.fy + int((1.0 - rise) * h)
        bottom = int(clip)
        rows = slice(0, max(0, bottom - top))
        img, solid = s.img[rows], s.solid[rows]
    else:
        # Out of the ground: squashed down to the feet, and growing.
        rows = max(1, int(h * rise))
        img = cv2.resize(s.img, (w, rows), interpolation=cv2.INTER_NEAREST)
        solid = cv2.resize(s.solid.astype(np.uint8), (w, rows),
                           interpolation=cv2.INTER_NEAREST) > 0
        top = int(round(foot)) - s.fy + (h - rows)
    _paste(canvas, img, solid, left, top)


def _paste(canvas, img, solid, left: int, top: int) -> None:
    H, W = canvas.shape[:2]
    h, w = solid.shape[:2]
    a0, b0 = max(0, left), max(0, top)
    a1, b1 = min(W, left + w), min(H, top + h)
    if a0 >= a1 or b0 >= b1:
        return
    part = (slice(b0 - top, b1 - top), slice(a0 - left, a1 - left))
    region = canvas[b0:b1, a0:a1]
    region[solid[part]] = img[part][solid[part]]


def blit_fallen(canvas, s: Sprite, x: float, foot: float, k: float, flip: bool,
                clip: Optional[float] = None) -> None:
    """The figure going over backwards, `k` of the way (0 to 1)."""
    h, w = s.img.shape[:2]
    angle = 90.0 * k * (1 if flip else -1)
    m = cv2.getRotationMatrix2D((float(s.fx), float(s.fy)), angle, 1.0)
    m[1, 2] += 0.25 * h * k * k                     # and sinking out of sight
    size = (w + h, h + w // 2)
    img = cv2.warpAffine(s.img, m, size, flags=cv2.INTER_NEAREST)
    solid = cv2.warpAffine(s.solid.astype(np.uint8), m, size, flags=cv2.INTER_NEAREST) > 0
    left, top = int(round(x)) - s.fx, int(round(foot)) - s.fy
    if clip is not None:
        solid[max(0, int(clip) - top):] = False
    _paste(canvas, img, solid, left, top)


# -- the town ---------------------------------------------------------------
@dataclass(frozen=True)
class Spot:
    """Where someone can appear, in fractions of the photograph: across, the
    line their feet are on, how tall they are, and the line below which they
    are hidden (a sill, a railing, a barrel), if any."""
    u: float
    foot: float
    height: float
    clip: Optional[float] = None


@dataclass(frozen=True)
class Scene:
    name: str
    title: str
    photo: str
    anchor: float                   # which part of the photo is kept, top (0) to bottom (1),
                                    # when the screen is wider than it
    spots: Tuple[Spot, ...]
    showdown: Spot
    runway: Spot                    # the line runners cross the street along (u unused)
    music: int                      # which piece plays here


SCENES = (
    Scene("stelmo", "ГОЛОВНА ВУЛИЦЯ", "stelmo.jpg", 0.6, (
        Spot(0.215, 0.58, 0.20, 0.53),          # windows
        Spot(0.350, 0.595, 0.18, 0.54),
        Spot(0.405, 0.595, 0.18, 0.54),
        Spot(0.545, 0.395, 0.17, 0.35),
        Spot(0.605, 0.39, 0.17, 0.345),
        Spot(0.725, 0.415, 0.16, 0.37),
        Spot(0.770, 0.415, 0.16, 0.375),
        Spot(0.120, 0.59, 0.21),                # doors
        Spot(0.465, 0.60, 0.17),
        Spot(0.705, 0.61, 0.15),
        Spot(0.560, 0.63, 0.17, 0.575),         # behind the fence
        Spot(0.920, 0.62, 0.13),                # by the shed
        Spot(0.150, 0.86, 0.26),                # in the street
        Spot(0.840, 0.84, 0.25),
    ), Spot(0.50, 0.88, 0.34), Spot(0.0, 0.74, 0.22), 1),
    Scene("store", "КРАМНИЦЯ", "store.jpg", 0.5, (
        Spot(0.270, 0.47, 0.12),                # on the roof
        Spot(0.560, 0.47, 0.12, 0.445),         # behind the museum's front
        Spot(0.240, 0.66, 0.15, 0.62),          # at the shop window
        Spot(0.355, 0.69, 0.15),                # in the doorway
        Spot(0.500, 0.70, 0.16, 0.62),          # behind the counter
        Spot(0.710, 0.74, 0.20),                # at the blacksmith's doors
        Spot(0.790, 0.74, 0.20),
        Spot(0.080, 0.70, 0.16, 0.65),          # behind the tarpaulin
        Spot(0.880, 0.76, 0.14, 0.68),          # behind the crates
        Spot(0.150, 0.92, 0.26),                # in the street
        Spot(0.620, 0.90, 0.26),
    ), Spot(0.50, 0.92, 0.32), Spot(0.0, 0.84, 0.22), 1),
    Scene("inn", "ЗАЇЗНИЙ ДВІР", "inn.jpg", 0.7, (
        Spot(0.425, 0.49, 0.17, 0.41),          # on the balcony
        Spot(0.480, 0.49, 0.17, 0.41),
        Spot(0.555, 0.49, 0.17, 0.41),
        Spot(0.650, 0.49, 0.17, 0.40),
        Spot(0.645, 0.79, 0.27),                # under the porch
        Spot(0.765, 0.78, 0.26),
        Spot(0.910, 0.86, 0.27),
        Spot(0.280, 0.69, 0.10),                # up the street
        Spot(0.150, 0.66, 0.08),
        Spot(0.420, 0.80, 0.22),
        Spot(0.180, 0.91, 0.27),
    ), Spot(0.26, 0.92, 0.34), Spot(0.0, 0.84, 0.22), 2),
    Scene("boardwalk", "ДОЩАНИЙ ТРОТУАР", "boardwalk.jpg", 0.55, (
        Spot(0.360, 0.47, 0.13, 0.425),         # on the balcony
        Spot(0.450, 0.47, 0.13, 0.425),
        Spot(0.525, 0.47, 0.12, 0.425),
        Spot(0.640, 0.46, 0.10, 0.42),
        Spot(0.130, 0.55, 0.12),                # in the doorways
        Spot(0.440, 0.57, 0.12),
        Spot(0.530, 0.58, 0.11),
        Spot(0.620, 0.56, 0.09),
        Spot(0.700, 0.55, 0.07),                # far down the street
        Spot(0.780, 0.54, 0.05),
        Spot(0.930, 0.72, 0.20),                # under the awning
        Spot(0.300, 0.80, 0.24),                # in the street
        Spot(0.600, 0.76, 0.22),
    ), Spot(0.45, 0.85, 0.30), Spot(0.0, 0.68, 0.18), 2),
    Scene("hotel", "ГОТЕЛЬ ГЕНКА", "hotel.jpg", 0.5, (
        Spot(0.450, 0.46, 0.14, 0.40),          # on the balcony
        Spot(0.530, 0.46, 0.14, 0.39),
        Spot(0.600, 0.46, 0.14, 0.40),
        Spot(0.330, 0.48, 0.14, 0.44),          # at the windows
        Spot(0.245, 0.52, 0.13, 0.48),
        Spot(0.290, 0.67, 0.13, 0.62),
        Spot(0.430, 0.66, 0.16),                # in the door
        Spot(0.570, 0.68, 0.16, 0.60),          # behind the porch fence
        Spot(0.680, 0.66, 0.15, 0.62),
        Spot(0.770, 0.68, 0.15),                # by the steps
        Spot(0.520, 0.95, 0.26, 0.88),          # behind the fence in front
        Spot(0.080, 0.90, 0.24, 0.84),          # behind the hay
        Spot(0.850, 0.80, 0.22),                # in the yard
    ), Spot(0.70, 0.88, 0.32), Spot(0.0, 0.78, 0.20), 3),
    Scene("calico", "КАЛІКО", "calico.jpg", 0.6, (
        Spot(0.480, 0.62, 0.07),                # far down the street
        Spot(0.560, 0.62, 0.07),
        Spot(0.645, 0.71, 0.15, 0.655),         # behind the barrel
        Spot(0.730, 0.66, 0.14),                # on the porch
        Spot(0.800, 0.67, 0.14),
        Spot(0.880, 0.73, 0.17, 0.655),         # behind the wheels
        Spot(0.330, 0.73, 0.17, 0.64),          # behind the cart
        Spot(0.080, 0.70, 0.17, 0.63),          # behind the fence
        Spot(0.420, 0.76, 0.22),
        Spot(0.220, 0.90, 0.27),
        Spot(0.760, 0.88, 0.26),
    ), Spot(0.50, 0.86, 0.34), Spot(0.0, 0.80, 0.22), 3),
)


def _cover(scene: Scene, size: Tuple[int, int]) -> Tuple[float, float, float]:
    """Scale and offset that make the photo cover the screen."""
    h, w = _photo(scene.photo).shape[:2]
    W, H = size
    k = max(W / float(w), H / float(h))
    return k, (W - w * k) / 2.0, (H - h * k) * scene.anchor


@lru_cache(maxsize=4)
def _photo(name: str) -> np.ndarray:
    return cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)


@lru_cache(maxsize=4)
def backdrop(scene: Scene, size: Tuple[int, int]) -> np.ndarray:
    k, dx, dy = _cover(scene, size)
    img = _photo(scene.photo)
    m = np.float32([[k, 0, dx], [0, k, dy]])
    return moonlight(cv2.warpAffine(img, m, size, flags=cv2.INTER_AREA))


def place(scene: Scene, spot: Spot, size: Tuple[int, int]) -> Tuple[float, float, int,
                                                                     Optional[float]]:
    """A spot on the screen: x, foot, height in px, clip line."""
    h, w = _photo(scene.photo).shape[:2]
    k, dx, dy = _cover(scene, size)
    y = lambda v: dy + v * h * k
    return (dx + spot.u * w * k, y(spot.foot), int(spot.height * h * k),
            None if spot.clip is None else y(spot.clip))
