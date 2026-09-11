"""Shooting gallery driven by the laser pointer.

Two ways to shoot, both read from the same stream of detections:

* **Flash** - the laser turns on while pointing at a target. This is the real
  trigger: aim with the beam off, flash it on the target. An instant hit.
* **Dwell** - hold the beam on a target for a moment. Needed because plenty of
  pointers are simply on all the time, and it is the only option once the beam
  is already lit.

All timing is passed in rather than read from the clock, so a whole round can
be played out in a test without waiting for it.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import overlay

NORMAL, BONUS, DECOY = "normal", "bonus", "decoy"

# Target colours avoid red entirely: the camera is looking at this screen, and
# a red target would read as a second laser dot. Magenta and yellow are safe -
# both drive the red channel, but never more than green or blue, so they carry
# no "redness" for the detector to lock onto.
COLOURS = {
    NORMAL: (255, 220, 60),     # cyan
    BONUS: (60, 240, 255),      # yellow
    DECOY: (255, 80, 255),      # magenta
}

READY, PLAYING, PAUSED, OVER = "ready", "playing", "paused", "over"


@dataclass
class Target:
    x: float
    y: float
    radius: float
    born: float
    lifetime: float
    kind: str = NORMAL
    vx: float = 0.0
    vy: float = 0.0
    dwell: float = 0.0          # how long the beam has rested on it
    dying: Optional[float] = None   # time it was hit, for the burst animation

    def age(self, now: float) -> float:
        return now - self.born

    def remaining(self, now: float) -> float:
        return max(0.0, 1.0 - self.age(now) / self.lifetime)

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return math.hypot(x - self.x, y - self.y) <= self.radius + slack


@dataclass
class Shot:
    """A hit or a miss, kept briefly so it can be drawn."""
    x: float
    y: float
    at: float
    points: int = 0
    text: str = ""
    good: bool = True


@dataclass
class Stats:
    score: int = 0
    hits: int = 0
    misses: int = 0        # shots that hit nothing
    escaped: int = 0       # targets that timed out
    decoys: int = 0
    best_combo: int = 0
    reactions: List[float] = field(default_factory=list)

    @property
    def shots(self) -> int:
        return self.hits + self.misses

    @property
    def accuracy(self) -> float:
        return self.hits / self.shots if self.shots else 0.0

    @property
    def reaction(self) -> float:
        return float(np.mean(self.reactions)) if self.reactions else 0.0


class HighScores:
    """Top scores, kept in a small JSON file next to the calibration."""

    def __init__(self, path: Path, limit: int = 5) -> None:
        self.path = path
        self.limit = limit
        self.entries: List[dict] = []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                self.entries = [e for e in data if isinstance(e, dict)][:limit]
        except Exception:
            pass                      # no scores yet, or a file we cannot read

    def rank_of(self, score: int) -> Optional[int]:
        better = sum(1 for e in self.entries if e.get("score", 0) >= score)
        return better + 1 if better < self.limit else None

    def add(self, stats: Stats, when: str) -> Optional[int]:
        rank = self.rank_of(stats.score)
        if rank is None or stats.score <= 0:
            return None
        self.entries.append({
            "score": int(stats.score),
            "hits": int(stats.hits),
            "accuracy": round(stats.accuracy, 3),
            "combo": int(stats.best_combo),
            "date": when,
        })
        self.entries.sort(key=lambda e: e.get("score", 0), reverse=True)
        del self.entries[self.limit:]
        try:
            self.path.write_text(json.dumps(self.entries, indent=2))
        except Exception:
            pass
        return rank


class Game:
    """Round state: spawning, shooting, scoring, and the difficulty ramp."""

    DURATION = 60.0
    LIVES = 3
    COUNTDOWN = 3.0
    DWELL_TIME = 0.35        # seconds on target for a dwell hit
    FLASH_GAP = 0.15         # beam off at least this long before a flash counts
    SHOT_VISIBLE = 0.7
    DECOY_AFTER = 15.0       # seconds into the round before decoys appear

    def __init__(self, screen_size: Tuple[int, int], duration: float = DURATION,
                 lives: int = LIVES, seed: Optional[int] = None,
                 margin: float = 0.10) -> None:
        self.screen_size = screen_size
        self.duration = duration
        self.lives_max = lives
        self.rng = random.Random(seed)
        self.margin = margin

        self.state = READY
        self.started = 0.0
        self.now = 0.0
        self.lives = lives
        self.combo = 0
        self.stats = Stats()
        self.targets: List[Target] = []
        self.shots: List[Shot] = []
        self.next_spawn = 0.0
        self.level = 1
        self._paused_at = 0.0
        self._beam_off_since: Optional[float] = None
        self._beam_on = False
        self._dwelt: Optional[Target] = None

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        self.state = PLAYING
        self.started = now + self.COUNTDOWN
        self.now = now
        self.lives = self.lives_max
        self.combo = 0
        self.stats = Stats()
        self.targets.clear()
        self.shots.clear()
        self.next_spawn = self.started
        self.level = 1

    def toggle_pause(self, now: float) -> None:
        self.now = max(self.now, now)
        if self.state == PLAYING:
            self.state = PAUSED
            self._paused_at = now
        elif self.state == PAUSED:
            # Shift every deadline by the time spent paused.
            delta = now - self._paused_at
            self.started += delta
            self.next_spawn += delta
            for t in self.targets:
                t.born += delta
            self.state = PLAYING

    @property
    def counting_down(self) -> bool:
        return self.state == PLAYING and self.now < self.started

    @property
    def elapsed(self) -> float:
        return max(0.0, self.now - self.started)

    @property
    def time_left(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    # -- difficulty ---------------------------------------------------------
    def _ramp(self) -> float:
        """0 at the start of the round, 1 at the end."""
        return min(1.0, self.elapsed / self.duration) if self.duration else 0.0

    def _spawn_gap(self) -> float:
        return 1.25 - 0.75 * self._ramp()

    def _radius(self) -> float:
        big = min(self.screen_size) * 0.075
        small = min(self.screen_size) * 0.030
        return big - (big - small) * self._ramp()

    def _lifetime(self) -> float:
        return 3.2 - 1.6 * self._ramp()

    # -- targets ------------------------------------------------------------
    def spawn(self, now: float) -> Target:
        w, h = self.screen_size
        radius = self._radius() * self.rng.uniform(0.85, 1.15)
        mx, my = w * self.margin + radius, h * self.margin + radius
        kind = NORMAL
        roll = self.rng.random()
        if self.elapsed > self.DECOY_AFTER and roll < 0.18:
            kind = DECOY
        elif roll > 0.88:
            kind = BONUS
        if kind == BONUS:
            radius *= 0.65
        speed = 60.0 * self._ramp() * (1.6 if kind == BONUS else 1.0)
        angle = self.rng.uniform(0, 2 * math.pi)
        t = Target(
            x=self.rng.uniform(mx, w - mx), y=self.rng.uniform(my, h - my),
            radius=radius, born=now,
            lifetime=self._lifetime() * (0.7 if kind == BONUS else 1.0),
            kind=kind,
            vx=math.cos(angle) * speed, vy=math.sin(angle) * speed,
        )
        self.targets.append(t)
        return t

    def _move(self, dt: float) -> None:
        w, h = self.screen_size
        for t in self.targets:
            if t.dying is not None or (t.vx == 0 and t.vy == 0):
                continue
            t.x += t.vx * dt
            t.y += t.vy * dt
            if t.x < t.radius or t.x > w - t.radius:
                t.vx = -t.vx
                t.x = min(max(t.x, t.radius), w - t.radius)
            if t.y < t.radius or t.y > h - t.radius:
                t.vy = -t.vy
                t.y = min(max(t.y, t.radius), h - t.radius)

    # -- scoring ------------------------------------------------------------
    def _points(self, target: Target, now: float) -> int:
        base = 10 + 40 * (1.0 - target.radius / (min(self.screen_size) * 0.09))
        base *= 1.0 + 0.5 * target.remaining(now)         # reward quick shots
        if target.kind == BONUS:
            base *= 3.0
        return int(round(base * self.multiplier))

    @property
    def multiplier(self) -> float:
        return 1.0 + 0.1 * min(self.combo, 10)

    # -- the shot -----------------------------------------------------------
    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        live = [t for t in self.targets if t.dying is None and t.contains(x, y, slack)]
        if not live:
            return None
        return min(live, key=lambda t: math.hypot(t.x - x, t.y - y))

    def _hit(self, target: Target, now: float, how: str) -> None:
        target.dying = now
        if target.kind == DECOY:
            self.combo = 0
            self.lives -= 1
            self.stats.decoys += 1
            self.stats.score = max(0, self.stats.score - 25)
            self.shots.append(Shot(target.x, target.y, now, -25, "DECOY -25", False))
            return
        points = self._points(target, now)
        self.combo += 1
        self.stats.best_combo = max(self.stats.best_combo, self.combo)
        self.stats.hits += 1
        self.stats.score += points
        self.stats.reactions.append(target.age(now))
        label = f"+{points}"
        if self.combo > 1:
            label += f"  x{self.multiplier:.1f}"
        if how == "flash":
            label += "  SNAP"
        self.shots.append(Shot(target.x, target.y, now, points, label, True))

    def _miss(self, x: float, y: float, now: float) -> None:
        self.combo = 0
        self.stats.misses += 1
        self.shots.append(Shot(x, y, now, 0, "miss", False))

    # -- main update --------------------------------------------------------
    def update(self, now: float, point: Optional[Tuple[float, float]]) -> None:
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        if self.state != PLAYING:
            self._beam_on = point is not None
            return

        if self.counting_down:
            self._beam_on = point is not None
            return

        self._move(dt)

        # Spawning and expiry.
        while now >= self.next_spawn and self.time_left > 0:
            self.spawn(now)
            self.next_spawn += self._spawn_gap()
        for t in list(self.targets):
            if t.dying is not None:
                if now - t.dying > 0.35:
                    self.targets.remove(t)
                continue
            if t.age(now) >= t.lifetime:
                self.targets.remove(t)
                if t.kind != DECOY:
                    self.stats.escaped += 1
                    self.combo = 0
                    self.lives -= 1

        self.level = 1 + int(self._ramp() * 4)

        # Shooting.
        was_on = self._beam_on
        self._beam_on = point is not None
        if point is None:
            if was_on:
                self._beam_off_since = now
            self._dwelt = None
            for t in self.targets:
                t.dwell = 0.0
        else:
            x, y = point
            gap = now - self._beam_off_since if self._beam_off_since is not None else None
            if not was_on and (gap is None or gap >= self.FLASH_GAP):
                # The beam just appeared: that is a trigger pull.
                target = self._target_at(x, y, slack=6.0)
                if target is not None:
                    self._hit(target, now, "flash")
                else:
                    self._miss(x, y, now)
                self._dwelt = None
            else:
                under = self._target_at(x, y)
                if under is None:
                    self._dwelt = None
                    for t in self.targets:
                        t.dwell = 0.0
                else:
                    if self._dwelt is not under:
                        self._dwelt = under
                        under.dwell = 0.0
                    under.dwell += dt
                    if under.dwell >= self.DWELL_TIME:
                        self._hit(under, now, "dwell")
                        self._dwelt = None

        self.shots = [s for s in self.shots if now - s.at <= self.SHOT_VISIBLE]

        if self.lives <= 0 or self.time_left <= 0:
            self.state = OVER
            self.targets.clear()      # nothing left to draw under the summary
            self.shots.clear()

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        now = self.now
        for t in self.targets:
            self._draw_target(canvas, t, now)
        for s in self.shots:
            self._draw_shot(canvas, s, now)
        if self.state != OVER:
            self._draw_hud(canvas)
        if self.counting_down:
            left = self.started - now
            overlay.text_centered(canvas, str(max(1, int(math.ceil(left)))),
                                  self.screen_size[1] // 2, 5.0, overlay.CYAN, 6)
            overlay.text_centered(canvas, "point the laser at the targets",
                                  self.screen_size[1] // 2 + 90, 0.9, overlay.WHITE)
        if self.state == PAUSED:
            overlay.draw_panel(canvas, [("PAUSED", 1.6, overlay.CYAN, 3),
                                        ("P to resume    ESC to quit", 0.7, overlay.GREY, 1)],
                               self.screen_size[1] / 2)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        colour = COLOURS[t.kind]
        x, y, r = int(t.x), int(t.y), int(t.radius)
        if t.dying is not None:
            # Burst: a ring expanding and fading out.
            k = (now - t.dying) / 0.35
            fade = max(0.0, 1.0 - k)
            cv2.circle(canvas, (x, y), int(r * (1 + 1.4 * k)),
                       tuple(int(c * fade) for c in colour), 3, cv2.LINE_AA)
            return

        left = t.remaining(now)
        # Rings shrink with the target's remaining life, so an old target
        # visibly runs out of time.
        cv2.circle(canvas, (x, y), r, colour, 2, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), max(2, int(r * 0.62)), colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), max(1, int(r * 0.12)), colour, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x, y), (r + 8, r + 8), -90, 0, 360 * left,
                    tuple(int(c * 0.8) for c in colour), 2, cv2.LINE_AA)
        if t.kind == DECOY:
            d = int(r * 0.5)
            cv2.line(canvas, (x - d, y - d), (x + d, y + d), colour, 3, cv2.LINE_AA)
            cv2.line(canvas, (x - d, y + d), (x + d, y - d), colour, 3, cv2.LINE_AA)
        elif t.kind == BONUS:
            overlay.text(canvas, "x3", (x + r + 10, y - r), 0.6, colour)
        if t.dwell > 0:
            cv2.ellipse(canvas, (x, y), (r - 6, r - 6), -90, 0,
                        360 * min(1.0, t.dwell / Game.DWELL_TIME),
                        overlay.WHITE, 3, cv2.LINE_AA)

    def _draw_shot(self, canvas, s: Shot, now: float) -> None:
        k = (now - s.at) / self.SHOT_VISIBLE
        fade = max(0.0, 1.0 - k)
        colour = overlay.GREEN if s.good else overlay.YELLOW
        overlay.text(canvas, s.text, (int(s.x) + 18, int(s.y) - 18 - int(30 * k)),
                     0.7, tuple(int(c * fade) for c in colour), 2)
        if not s.good:
            d = 10
            c = tuple(int(v * fade) for v in overlay.YELLOW)
            cv2.line(canvas, (int(s.x) - d, int(s.y) - d), (int(s.x) + d, int(s.y) + d), c, 2)
            cv2.line(canvas, (int(s.x) - d, int(s.y) + d), (int(s.x) + d, int(s.y) - d), c, 2)

    def _draw_hud(self, canvas) -> None:
        w = self.screen_size[0]
        overlay.text(canvas, f"SCORE {self.stats.score}", (40, 60), 1.1, overlay.CYAN, 2)
        if self.combo > 1:
            overlay.text(canvas, f"x{self.multiplier:.1f}  ({self.combo} in a row)",
                         (40, 100), 0.7, overlay.GREEN)

        left = self.time_left
        bar_w = 360
        x0 = (w - bar_w) // 2
        cv2.rectangle(canvas, (x0, 34), (x0 + bar_w, 52), overlay.DIM, 1)
        fill = int(bar_w * (left / self.duration if self.duration else 0))
        colour = overlay.CYAN if left > 10 else overlay.YELLOW
        cv2.rectangle(canvas, (x0, 34), (x0 + fill, 52), colour, -1)
        overlay.text_centered(canvas, f"{left:4.1f}s   level {self.level}", 84, 0.7, colour)

        for i in range(self.lives_max):
            cx = w - 60 - i * 44
            colour = overlay.GREEN if i < self.lives else overlay.DIM
            cv2.circle(canvas, (cx, 50), 12, colour, -1 if i < self.lives else 1, cv2.LINE_AA)

    def draw_over(self, canvas, scores: HighScores, rank: Optional[int]) -> None:
        s = self.stats
        lines = [
            ("GAME OVER" if self.lives <= 0 else "TIME", 1.8, overlay.CYAN, 3),
            (f"score {s.score}", 1.3, overlay.WHITE, 2),
            (f"{s.hits} hits   {s.misses} missed shots   {s.escaped} got away",
             0.75, overlay.GREY, 1),
            (f"accuracy {s.accuracy:.0%}   best streak {s.best_combo}   "
             f"reaction {s.reaction:.2f}s", 0.75, overlay.GREY, 1),
        ]
        if rank:
            lines.append((f"NEW HIGH SCORE - #{rank}", 0.9, overlay.GREEN, 2))
        lines.append(("G play again    ESC back to tracking", 0.7, overlay.YELLOW, 1))
        overlay.draw_panel(canvas, lines, self.screen_size[1] * 0.42)

        if scores.entries:
            y = int(self.screen_size[1] * 0.78)
            overlay.text_centered(canvas, "BEST", y, 0.7, overlay.CYAN)
            for i, e in enumerate(scores.entries):
                y += 30
                overlay.text_centered(
                    canvas,
                    f"{i + 1}.  {e.get('score', 0):>6}   "
                    f"{e.get('accuracy', 0):.0%} accurate   x{e.get('combo', 0)}   "
                    f"{e.get('date', '')}",
                    y, 0.6, overlay.WHITE if i + 1 == rank else overlay.GREY)
