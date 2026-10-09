"""Games driven by the laser pointer: the shared round engine and the
shooting gallery built on it (the maths games live in mathgames.py).

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
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import overlay

NORMAL, BONUS, DECOY = "normal", "bonus", "decoy"

# Target colours avoid red entirely: the camera is looking at this screen, and
# a red target would read as a second laser dot. That rules out magenta and
# plain yellow as well - through the camera both carry enough red to be taken
# for the laser (see overlay.py).
COLOURS = {
    NORMAL: (255, 220, 60),     # cyan
    BONUS: overlay.YELLOW,
    DECOY: (255, 130, 90),      # blue
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
    label: str = ""             # what is written on it, for the maths games
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
            data = json.loads(path.read_text(encoding="utf-8"))
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
        self._save()
        return rank

    def set_name(self, rank: int, name: str) -> None:
        """Put a name to the score at `rank`, once the player has typed it."""
        if name and 0 < rank <= len(self.entries):
            self.entries[rank - 1]["name"] = name
            self._save()

    @staticmethod
    def who(entry: dict) -> str:
        """The name on a row of the table, if it has one, with room after it."""
        name = entry.get("name", "")
        return f"{name}   " if name else ""

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        except Exception:
            pass


Box = Tuple[int, int, int, int]


class Hold:
    """Buttons pressed by resting the pointer on one, as in the game chooser."""

    DWELL = 0.9
    GRACE = 0.35             # the hold survives losing the dot for this long

    def __init__(self) -> None:
        self.over: Optional[int] = None
        self.dwell = 0.0
        self._away = 0.0

    def reset(self) -> None:
        self.over, self.dwell, self._away = None, 0.0, 0.0

    def update(self, dt: float, point: Optional[Tuple[float, float]],
               boxes: Sequence[Box]) -> Optional[int]:
        """The button whose hold has just completed, if any."""
        at = None
        if point is not None:
            at = next((i for i, (x0, y0, x1, y1) in enumerate(boxes)
                       if x0 <= point[0] <= x1 and y0 <= point[1] <= y1), None)
        if at is None:
            self._away += dt
            if self._away > self.GRACE:
                self.reset()
            return None
        if at != self.over:
            self.over, self.dwell = at, 0.0
        self._away = 0.0
        self.dwell += dt
        if self.dwell >= self.DWELL:
            self.reset()
            return at
        return None

    def fill(self, i: int) -> float:
        return min(1.0, self.dwell / self.DWELL) if i == self.over else 0.0


class Round:
    """What every game shares: the countdown, pausing, and turning the stream
    of pointer positions into shots at whatever is in `targets`."""

    COUNTDOWN = 3.0
    DWELL_TIME = 0.35        # seconds on target for a dwell hit
    FLASH = True             # False: dwell is the only trigger
    FLASH_GAP = 0.3          # the dot gone this long means the beam is off; less
                             # is the detector missing frames, and changes nothing
    SHOT_VISIBLE = 0.7
    HINT = "наводь лазер на мішені"
    PAUSE_TEXT = ("ПАУЗА", "P — продовжити    ESC — вийти")
    scores_name: Optional[str] = None    # high-score file stem; None = no table
    needs_sight = True                   # False while a game has no use for the
                                         # pointer, and need not stop without it
    players = 1                          # 2: update() is also given every dot on
                                         # the screen, not just the strongest
    home: Optional[Path] = None          # where a game may keep files of its own;
                                         # the app sets it before start()

    def __init__(self, screen_size: Tuple[int, int], seed: Optional[int] = None) -> None:
        self.screen_size = screen_size
        self.rng = random.Random(seed)

        self.state = READY
        self.started = 0.0
        self.now = 0.0
        self.stats = Stats()
        self.targets: List[Target] = []
        self.shots: List[Shot] = []
        self._paused_at = 0.0
        self._beam_off_since: Optional[float] = None    # when the dot was lost
        self._beam_on = False
        self._dwelt: Optional[Target] = None

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        self.state = PLAYING
        self.started = now + self.COUNTDOWN
        self.now = now
        self.stats = Stats()
        self.targets.clear()
        self.shots.clear()

    def toggle_pause(self, now: float) -> None:
        self.now = max(self.now, now)
        if self.state == PLAYING:
            self.state = PAUSED
            self._paused_at = now
        elif self.state == PAUSED:
            self._resume(now - self._paused_at)
            self.state = PLAYING

    def _resume(self, delta: float) -> None:
        """Shift every deadline by the time spent paused."""
        self.started += delta
        for t in self.targets:
            t.born += delta

    @property
    def counting_down(self) -> bool:
        return self.state == PLAYING and self.now < self.started

    @property
    def elapsed(self) -> float:
        return max(0.0, self.now - self.started)

    # -- what a game fills in -----------------------------------------------
    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        raise NotImplementedError

    def _finished(self) -> bool:
        raise NotImplementedError

    def _hit(self, target: Target, now: float, how: str) -> None:
        raise NotImplementedError

    def _miss(self, x: float, y: float, now: float) -> None:
        pass

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        raise NotImplementedError

    def _draw_hud(self, canvas) -> None:
        pass

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        raise NotImplementedError

    def key(self, key: int) -> bool:
        """A key pressed during the game. True if the game took it, and the app
        should leave it alone."""
        return False

    def close(self) -> None:
        """The game is being left: stop whatever it has running."""

    # -- the shot -----------------------------------------------------------
    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        live = [t for t in self.targets if t.dying is None and t.contains(x, y, slack)]
        if not live:
            return None
        return min(live, key=lambda t: math.hypot(t.x - x, t.y - y))

    def _beam(self, now: float, point: Optional[Tuple[float, float]]) -> bool:
        """Follow the beam on and off. True when it has just been lit."""
        if point is None:
            if self._beam_off_since is None:
                self._beam_off_since = now
            if now - self._beam_off_since >= self.FLASH_GAP:
                self._beam_on = False
            return False
        if self._beam_off_since is not None and now - self._beam_off_since >= self.FLASH_GAP:
            self._beam_on = False
        lit = not self._beam_on
        self._beam_on, self._beam_off_since = True, None
        return lit

    def _aim(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        lit = self._beam(now, point)
        if point is None:
            if not self._beam_on:
                self._dwelt = None
                for t in self.targets:
                    t.dwell = 0.0
            return                    # a dropped frame: the hold carries on
        x, y = point
        if self.FLASH and lit:
            # The beam just appeared: that is a trigger pull.
            target = self._target_at(x, y, slack=6.0)
            if target is not None:
                self._hit(target, now, "flash")
            else:
                self._miss(x, y, now)
            self._dwelt = None
            return
        under = self._target_at(x, y)
        if under is None:
            self._dwelt = None
            for t in self.targets:
                t.dwell = 0.0
            return
        if self._dwelt is not under:
            self._dwelt = under
            under.dwell = 0.0
        under.dwell += dt
        if under.dwell >= self.DWELL_TIME:
            self._hit(under, now, "dwell")
            self._dwelt = None

    # -- main update --------------------------------------------------------
    def update(self, now: float, point: Optional[Tuple[float, float]]) -> None:
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        if self.state != PLAYING or self.counting_down:
            self._beam(now, point)
            return

        self._step(now, dt, point)
        self.shots = [s for s in self.shots if now - s.at <= self.SHOT_VISIBLE]

        if self._finished():
            self.state = OVER
            self.targets.clear()      # nothing left to draw under the summary
            self.shots.clear()

    # -- drawing ------------------------------------------------------------
    def draw_pointer(self, canvas, x: float, y: float, t: float,
                     crosshair: bool = False) -> None:
        """Where a laser is pointing. A game with a look of its own draws its own."""
        overlay.draw_target(canvas, x, y, t, 1.0, crosshair=crosshair)

    KICK = 0.2               # seconds a pointer shows that a shot went off

    def _kick(self, shots: Optional[List[Shot]] = None) -> float:
        """1 just as the newest shot went off, falling to 0 over KICK."""
        shots = self.shots if shots is None else shots
        if not shots:
            return 0.0
        return max(0.0, 1.0 - (self.now - max(s.at for s in shots)) / self.KICK)

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
            overlay.text_centered(canvas, self.HINT,
                                  self.screen_size[1] // 2 + 90, 0.9, overlay.WHITE)
        if self.state == PAUSED:
            overlay.draw_panel(canvas, [(self.PAUSE_TEXT[0], 1.6, overlay.CYAN, 3),
                                        (self.PAUSE_TEXT[1], 0.7, overlay.GREY, 1)],
                               self.screen_size[1] / 2)

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


class Game(Round):
    """The shooting gallery: spawning, scoring, and the difficulty ramp."""

    DURATION = 60.0
    LIVES = 3
    DECOY_AFTER = 15.0       # seconds into the round before decoys appear
    SPACING = 24.0           # clear pixels kept between two targets' rims
    scores_name = "highscores"

    def __init__(self, screen_size: Tuple[int, int], duration: float = DURATION,
                 lives: int = LIVES, seed: Optional[int] = None,
                 margin: float = 0.10) -> None:
        super().__init__(screen_size, seed)
        self.duration = duration
        self.lives_max = lives
        self.margin = margin
        self.lives = lives
        self.combo = 0
        self.next_spawn = 0.0
        self.level = 1

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        self.lives = self.lives_max
        self.combo = 0
        self.next_spawn = self.started
        self.level = 1

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        self.next_spawn += delta

    @property
    def time_left(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    def _finished(self) -> bool:
        return self.lives <= 0 or self.time_left <= 0

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
        kind = self._pick_kind()
        if kind == BONUS:
            radius *= 0.65
        speed = 60.0 * self._ramp() * (1.6 if kind == BONUS else 1.0)
        angle = self.rng.uniform(0, 2 * math.pi)
        x, y = self._free_spot(radius, mx, my)
        t = Target(
            x=x, y=y,
            radius=radius, born=now,
            lifetime=self._lifetime() * (0.7 if kind == BONUS else 1.0),
            kind=kind,
            vx=math.cos(angle) * speed, vy=math.sin(angle) * speed,
        )
        self.targets.append(t)
        return t

    def _free_spot(self, radius: float, mx: float, my: float) -> Tuple[float, float]:
        """Somewhere clear of the targets already up - or, on a screen too
        crowded for that, the clearest of the spots tried."""
        w, h = self.screen_size
        best, best_gap = (w / 2.0, h / 2.0), -math.inf
        for _ in range(40):
            x, y = self.rng.uniform(mx, w - mx), self.rng.uniform(my, h - my)
            gap = min((math.hypot(x - t.x, y - t.y) - t.radius - radius
                       for t in self.targets), default=math.inf)
            if gap >= self.SPACING:
                return x, y
            if gap > best_gap:
                best, best_gap = (x, y), gap
        return best

    def _pick_kind(self) -> str:
        roll = self.rng.random()
        if self.elapsed > self.DECOY_AFTER and roll < 0.18:
            return DECOY
        if roll > 0.88:
            return BONUS
        return NORMAL

    def _move(self, dt: float) -> None:
        w, h = self.screen_size
        # Targets drifting into each other bounce apart rather than overlap.
        live = [t for t in self.targets if t.dying is None]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                dist = math.hypot(dx, dy)
                if dist == 0 or dist >= a.radius + b.radius + self.SPACING:
                    continue
                nx, ny = dx / dist, dy / dist
                closing = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
                if closing > 0:
                    a.vx -= closing * nx
                    a.vy -= closing * ny
                    b.vx += closing * nx
                    b.vy += closing * ny
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
    def _hit(self, target: Target, now: float, how: str) -> None:
        target.dying = now
        if target.kind == DECOY:
            self.combo = 0
            self.lives -= 1
            self.stats.decoys += 1
            self.stats.score = max(0, self.stats.score - 25)
            self.shots.append(Shot(target.x, target.y, now, -25,
                                   self._decoy_text(target), False))
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
            label += "  МИТТЮ"
        self.shots.append(Shot(target.x, target.y, now, points, label, True))

    def _decoy_text(self, target: Target) -> str:
        return "ПАСТКА -25"

    def review(self) -> list:
        """Lines for the end panel about what went wrong; the gallery has none."""
        return []

    def draw_pointer(self, canvas, x: float, y: float, t: float,
                     crosshair: bool = False) -> None:
        overlay.draw_scope(canvas, x, y, overlay.CYAN, self.screen_size[1] / 1080.0,
                           self._kick(), crosshair)

    def _escape(self, target: Target, now: float) -> None:
        """A target ran out of time unshot."""
        if target.kind != DECOY:
            self.stats.escaped += 1
            self.combo = 0
            self.lives -= 1

    def _miss(self, x: float, y: float, now: float) -> None:
        self.combo = 0
        self.stats.misses += 1
        self.shots.append(Shot(x, y, now, 0, "мимо", False))

    # -- main update --------------------------------------------------------
    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
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
                self._escape(t, now)

        self.level = 1 + int(self._ramp() * 4)
        self._aim(now, dt, point)

    # -- drawing ------------------------------------------------------------
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
                        360 * min(1.0, t.dwell / self.DWELL_TIME),
                        overlay.WHITE, 3, cv2.LINE_AA)

    def _draw_hud(self, canvas) -> None:
        w = self.screen_size[0]
        overlay.text(canvas, f"РАХУНОК {self.stats.score}", (40, 60), 1.1, overlay.CYAN, 2)
        if self.combo > 1:
            overlay.text(canvas, f"x{self.multiplier:.1f}  ({self.combo} поспіль)",
                         (40, 100), 0.7, overlay.GREEN)

        left = self.time_left
        bar_w = 360
        x0 = (w - bar_w) // 2
        cv2.rectangle(canvas, (x0, 34), (x0 + bar_w, 52), overlay.DIM, 1)
        fill = int(bar_w * (left / self.duration if self.duration else 0))
        colour = overlay.CYAN if left > 10 else overlay.YELLOW
        cv2.rectangle(canvas, (x0, 34), (x0 + fill, 52), colour, -1)
        overlay.text_centered(canvas, f"{left:4.1f} с   рівень {self.level}", 84, 0.7, colour)

        for i in range(self.lives_max):
            cx = w - 60 - i * 44
            colour = overlay.GREEN if i < self.lives else overlay.DIM
            cv2.circle(canvas, (cx, 50), 12, colour, -1 if i < self.lives else 1, cv2.LINE_AA)

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        s = self.stats
        lines = [
            ("ГРУ ЗАКІНЧЕНО" if self.lives <= 0 else "ЧАС ВИЙШОВ", 1.8, overlay.CYAN, 3),
            (f"рахунок {s.score}", 1.3, overlay.WHITE, 2),
            (f"влучань: {s.hits}   промахів: {s.misses}   утекло мішеней: {s.escaped}",
             0.75, overlay.GREY, 1),
            (f"влучність {s.accuracy:.0%}   найдовша серія {s.best_combo}   "
             f"реакція {s.reaction:.2f} с", 0.75, overlay.GREY, 1),
        ]
        lines += self.review()
        if rank:
            lines.append((f"НОВИЙ РЕКОРД — №{rank}", 0.9, overlay.GREEN, 2))
        lines.append(("G — грати ще    ESC — вихід", 0.7, overlay.YELLOW, 1))
        overlay.draw_panel(canvas, lines, self.screen_size[1] * 0.42)

        if scores is not None and scores.entries:
            y = int(self.screen_size[1] * 0.78)
            overlay.text_centered(canvas, "РЕКОРДИ", y, 0.7, overlay.CYAN)
            for i, e in enumerate(scores.entries):
                y += 30
                overlay.text_centered(
                    canvas,
                    f"{i + 1}.  {HighScores.who(e)}{e.get('score', 0):>6}   "
                    f"влучність {e.get('accuracy', 0):.0%}   x{e.get('combo', 0)}   "
                    f"{e.get('date', '')}",
                    y, 0.6, overlay.WHITE if i + 1 == rank else overlay.GREY)
