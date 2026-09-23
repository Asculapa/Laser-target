"""Silhouette range, on the round engine in game.py: a man-shaped target with
numbered scoring rings, as on a shooting range.

Ten targets come up one at a time and each gets one shot. The shot scores the
number of the ring it lands in - 10 in the middle of the chest, down to 5 for
the rest of the figure - so unlike the gallery it matters *where* the target
is hit, not just that it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import overlay
from .game import HighScores, Round, Shot, Target

# The figure in units of its own height: u across from the centre line, v down
# from the top of the head.
HEAD = (0.125, 0.095, 0.125)        # centre v, radius across, radius down
BODY = ((0.05, 0.21), (0.14, 0.27), (0.27, 0.33), (0.31, 0.45), (0.27, 1.0))
HALF_WIDTH = max(u for u, _ in BODY)
CENTRE_V = 0.56                     # the rings are centred on the chest
RING = (0.05, 0.068)                # width of one ring, across and down
BEST, LEAST = 10, 5                 # the innermost ring, and the rest of the figure

# Green on black, like the paper ones - and, like everything else drawn on
# this screen, nowhere near red. The fill is faint so that a laser dot on the
# figure still stands out for the detector.
FILL = tuple(int(c * 0.16) for c in overlay.GREEN)
PAD = 3                             # sprite border, room for the outline


def zone(u: float, v: float) -> int:
    """Points for a shot at (u, v): 0 off the figure, otherwise 5 to 10."""
    cv_, ru, rv = HEAD
    on_head = (u / ru) ** 2 + ((v - cv_) / rv) ** 2 <= 1.0
    on_body = BODY[0][1] <= v <= 1.0 and abs(u) <= float(
        np.interp(v, [p[1] for p in BODY], [p[0] for p in BODY]))
    if not (on_head or on_body):
        return 0
    rings_out = int(math.hypot(u / RING[0], (v - CENTRE_V) / RING[1]))
    return max(LEAST, BEST - rings_out)


def render(height: int) -> Tuple[np.ndarray, np.ndarray]:
    """The figure `height` pixels tall, and which of its pixels are drawn."""
    width = int(2 * HALF_WIDTH * height)
    shape = (height + 2 * PAD, width + 2 * PAD)

    def px(u: float, v: float) -> Tuple[int, int]:
        return int(round(shape[1] / 2 + u * height)), int(round(PAD + v * height))

    mask = np.zeros(shape, np.uint8)
    outline = [px(u, v) for u, v in BODY] + [px(-u, v) for u, v in reversed(BODY)]
    cv2.fillPoly(mask, [np.array(outline, np.int32)], 255)
    cv2.ellipse(mask, px(0, HEAD[0]), (int(HEAD[1] * height), int(HEAD[2] * height)),
                0, 0, 360, 255, -1)

    img = np.zeros(shape + (3,), np.uint8)
    img[mask > 0] = FILL
    rings = BEST - LEAST
    for k in range(1, rings + 1):
        cv2.ellipse(img, px(0, CENTRE_V), (int(RING[0] * k * height), int(RING[1] * k * height)),
                    0, 0, 360, overlay.GREY, 1, cv2.LINE_AA)
    # The numbers run across the body, one in each ring on either side.
    digit = (RING[0] * height * 0.7, RING[0] * height * 0.55)
    thick = 2 if height >= 500 else 1
    overlay.text_fit(img, str(BEST), px(0, CENTRE_V), digit[0] * 2, digit[1],
                     overlay.WHITE, thick)
    for k in range(1, rings):
        for side in (-1, 1):
            overlay.text_fit(img, str(BEST - k), px(side * (k + 0.5) * RING[0], CENTRE_V),
                             digit[0], digit[1], overlay.WHITE, thick)
    for v in (HEAD[0], 0.95):
        overlay.text_fit(img, str(LEAST), px(0, v), digit[0], digit[1], overlay.GREY, thick)
    img[mask == 0] = 0                  # the outer ring spills past the shoulders

    edges, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, edges, -1, overlay.GREEN, 2, cv2.LINE_AA)
    return img, (mask > 0) | (img.max(axis=2) > 0)


@dataclass
class Figure(Target):
    """A silhouette; (x, y) is the middle of its rings."""
    height: float = 0.0
    points: Optional[int] = None                 # what the shot at it scored
    hole: Optional[Tuple[float, float]] = None   # where it landed, in figure units

    def unit(self, x: float, y: float) -> Tuple[float, float]:
        return (x - self.x) / self.height, (y - self.y) / self.height + CENTRE_V

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return zone(*self.unit(x, y)) > 0


class Silhouette(Round):
    """Ten figures, one shot each, every one a little further away than the
    last; the final few also walk along the line."""

    DWELL_TIME = 0.5         # an aimed shot: the beam has to rest, not just pass
    HINT = "one shot at each figure - aim for the 10"
    SHOTS = 10
    SHOW = 1.1               # a shot figure stays up this long, to show the hit
    GAP = 0.5                # ...and the line stays empty this long after it
    GROUND = 0.84            # the target line, as a share of the screen height
    scores_name = "highscores-silhouette"

    def __init__(self, screen_size: Tuple[int, int], shots: int = SHOTS,
                 seed: Optional[int] = None) -> None:
        super().__init__(screen_size, seed)
        self.n_shots = shots
        self.results: List[int] = []         # points per figure, in order
        # A dwell only builds while the beam stays within this many pixels.
        self.steady = 0.03 * min(screen_size)
        self._next_at = 0.0
        self._hold: Optional[Tuple[float, float]] = None
        self._last_x: Optional[float] = None
        self._sprites: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        self.results = []
        self._next_at = self.started
        self._hold = None
        self._last_x = None

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        self._next_at += delta
        for t in self.targets:
            if t.dying is not None:
                t.dying += delta

    def _finished(self) -> bool:
        return len(self.results) >= self.n_shots and not self.targets

    # -- figures ------------------------------------------------------------
    def _edge(self, height: float) -> float:
        """How close to the side of the screen a figure's centre may come."""
        return HALF_WIDTH * height + 0.04 * self.screen_size[0]

    def spawn(self, now: float) -> Figure:
        w, h = self.screen_size
        k = len(self.results) / max(1, self.n_shots - 1)    # 0 first, 1 last
        height = float(int(h * (0.60 - 0.25 * k)))
        edge = self._edge(height)
        x = self.rng.uniform(edge, w - edge)
        for _ in range(8):                  # not where the last one stood
            if self._last_x is None or abs(x - self._last_x) >= 0.15 * w:
                break
            x = self.rng.uniform(edge, w - edge)
        self._last_x = x
        speed = 0.08 * w * max(0.0, k - 0.6) / 0.4
        t = Figure(
            x=x, y=h * self.GROUND - (1.05 - CENTRE_V) * height,
            radius=height / 2, born=now, lifetime=6.0 - 2.0 * k,
            vx=speed * self.rng.choice((-1, 1)), height=height,
        )
        self.targets.append(t)
        return t

    def _move(self, t: Figure, dt: float) -> None:
        edge = self._edge(t.height)
        t.x += t.vx * dt
        if t.x < edge or t.x > self.screen_size[0] - edge:
            t.vx = -t.vx
            t.x = min(max(t.x, edge), self.screen_size[0] - edge)

    # -- the shot -----------------------------------------------------------
    def _aim(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        lit = self._beam(now, point)
        t = next((t for t in self.targets if t.dying is None), None)
        if point is None:
            if self._beam_on:
                return                # a dropped frame: the hold carries on
        elif t is not None:
            if lit:
                self._shoot(t, point, now)
                return
            if t.contains(*point):
                # Unlike the gallery's dwell, sweeping across the figure must
                # not fire: the shot would land wherever the beam happened to
                # be. It has to come to rest first.
                if self._hold is None or math.hypot(point[0] - self._hold[0],
                                                    point[1] - self._hold[1]) > self.steady:
                    self._hold = point
                    t.dwell = 0.0
                else:
                    t.dwell += dt
                    if t.dwell >= self.DWELL_TIME:
                        self._shoot(t, point, now)
                return
        self._hold = None
        if t is not None:
            t.dwell = 0.0

    def _shoot(self, t: Figure, point: Tuple[float, float], now: float) -> None:
        u, v = t.unit(*point)
        points = zone(u, v)
        if points:
            t.hole = (u, v)
            self.stats.hits += 1
            self.stats.reactions.append(t.age(now))
        else:
            self.stats.misses += 1
            self.shots.append(Shot(point[0], point[1], now, 0, "miss", False))
        self._record(t, now, points, str(points))

    def _record(self, t: Figure, now: float, points: int, label: str) -> None:
        t.dying = now
        t.points = points
        t.label = label
        t.dwell = 0.0
        self._hold = None
        self.results.append(points)
        self.stats.score += points

    # -- main update --------------------------------------------------------
    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        for t in list(self.targets):
            if t.dying is not None:
                if now - t.dying > self.SHOW:
                    self.targets.remove(t)
                    self._next_at = now + self.GAP
            elif t.age(now) >= t.lifetime:
                self.stats.escaped += 1
                self._record(t, now, 0, "too slow")
            else:
                self._move(t, dt)
        if not self.targets and len(self.results) < self.n_shots and now >= self._next_at:
            self.spawn(now)
        self._aim(now, dt, point)

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        if self.state != "over":
            y = int(self.screen_size[1] * self.GROUND)
            cv2.line(canvas, (0, y), (self.screen_size[0], y), overlay.DIM, 2, cv2.LINE_AA)
        super().draw(canvas)

    def _draw_target(self, canvas, t: Figure, now: float) -> None:
        height = int(t.height)
        if height not in self._sprites:
            self._sprites[height] = render(height)
        img, solid = self._sprites[height]
        x0 = int(round(t.x - img.shape[1] / 2))
        y0 = int(round(t.y - CENTRE_V * height)) - PAD
        ground = int(self.screen_size[1] * self.GROUND)

        for side in (-1, 1):                # the posts it stands on
            x = int(t.x + side * 0.12 * height)
            cv2.line(canvas, (x, y0 + PAD + height), (x, ground), overlay.GREY, 3, cv2.LINE_AA)

        # Paste the part of the sprite that is on screen.
        h, w = canvas.shape[:2]
        a0, b0 = max(0, x0), max(0, y0)
        a1, b1 = min(w, x0 + img.shape[1]), min(h, y0 + img.shape[0])
        if a0 < a1 and b0 < b1:
            part = (slice(b0 - y0, b1 - y0), slice(a0 - x0, a1 - x0))
            canvas[b0:b1, a0:a1][solid[part]] = img[part][solid[part]]

        if t.dying is None:
            # The time left to shoot, as a bar on the ground under the figure.
            left = t.remaining(now)
            half = int(HALF_WIDTH * height * left)
            colour = overlay.CYAN if left > 0.25 else overlay.YELLOW
            cv2.line(canvas, (int(t.x) - half, ground + 14), (int(t.x) + half, ground + 14),
                     colour, 4, cv2.LINE_AA)
            if t.dwell > 0 and self._hold is not None:
                # Outside the reticle the app draws on the pointer.
                cv2.ellipse(canvas, (int(self._hold[0]), int(self._hold[1])), (68, 68), -90,
                            0, 360 * min(1.0, t.dwell / self.DWELL_TIME),
                            overlay.WHITE, 3, cv2.LINE_AA)
            return

        if t.hole is not None:
            centre = (int(t.x + t.hole[0] * height),
                      int(t.y + (t.hole[1] - CENTRE_V) * height))
            r = max(5, int(height * 0.014))
            cv2.circle(canvas, centre, r, overlay.WHITE, -1, cv2.LINE_AA)
            cv2.circle(canvas, centre, r + 5, overlay.CYAN, 2, cv2.LINE_AA)
        colour = overlay.GREEN if (t.points or 0) >= BEST - 1 else \
            overlay.WHITE if t.points else overlay.YELLOW
        size = self.screen_size[1] * 0.07
        overlay.text_fit(canvas, t.label, (int(t.x), int(y0 - size)),
                         size * 6, size, colour, 3)

    def _draw_hud(self, canvas) -> None:
        overlay.text(canvas, f"SCORE {self.stats.score}", (40, 60), 1.1, overlay.CYAN, 2)
        # The series so far: a number per figure shot, a dot per figure to come.
        step = 48
        x0 = self.screen_size[0] / 2 - step * (self.n_shots - 1) / 2
        for i in range(self.n_shots):
            centre = (int(x0 + i * step), 50)
            if i < len(self.results):
                points = self.results[i]
                colour = overlay.GREEN if points >= BEST - 1 else \
                    overlay.WHITE if points else overlay.GREY
                overlay.text_fit(canvas, str(points), centre, 38, 20, colour, 2)
            else:
                now_up = i == len(self.results)
                cv2.circle(canvas, centre, 6, overlay.CYAN if now_up else overlay.DIM,
                           -1 if now_up else 1, cv2.LINE_AA)

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        s = self.stats
        lines = [
            ("SERIES OVER", 1.8, overlay.CYAN, 3),
            (f"{s.score} of {BEST * self.n_shots}", 1.3, overlay.WHITE, 2),
            ("  ".join(str(p) for p in self.results), 0.9, overlay.GREEN, 2),
            (f"{self.results.count(BEST)} tens   {s.misses} off the figure   "
             f"{s.escaped} too slow   reaction {s.reaction:.2f}s", 0.75, overlay.GREY, 1),
        ]
        if rank:
            lines.append((f"NEW HIGH SCORE - #{rank}", 0.9, overlay.GREEN, 2))
        lines.append(("G play again    ESC back to tracking", 0.7, overlay.YELLOW, 1))
        overlay.draw_panel(canvas, lines, self.screen_size[1] * 0.42)

        if scores is not None and scores.entries:
            y = int(self.screen_size[1] * 0.78)
            overlay.text_centered(canvas, "BEST", y, 0.7, overlay.CYAN)
            for i, e in enumerate(scores.entries):
                y += 30
                overlay.text_centered(
                    canvas,
                    f"{i + 1}.  {e.get('score', 0):>4}   "
                    f"{e.get('hits', 0)} on the figure   {e.get('date', '')}",
                    y, 0.6, overlay.WHITE if i + 1 == rank else overlay.GREY)
