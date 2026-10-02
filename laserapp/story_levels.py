"""The five chapters of the story game, each a small game of its own on the
round engine in game.py. story.py strings them together and tells the tale
around them.

Each chapter asks something different of the pointer:

1. Sparks       - quick shots at things blowing past
2. Siege        - guarding the middle of the screen, and telling friend from foe
3. Ships        - holding the beam on something that moves
4. Constellations - remembering an order
5. Storm        - all of it at once, against one opponent

A chapter is won or lost rather than timed, and reports what happens in it
through `events`, which story.py turns into sound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import overlay
from . import story_art as art
from . import story_script as script
from .game import Round, Shot, Target

GLOAMLING, BRUTE, MOTH, KNOT, EYE, GOLD = "gloamling", "brute", "moth", "knot", "eye", "gold"


class Chapter(Round):
    """What the chapters share: lives, a streak, an ending, and a HUD."""

    NUMBER = 0
    SCENE = "dark"           # the backdrop, see story_art.Backdrop
    LIVES = 3
    BURST = 0.35             # seconds a struck target takes to fade
    END_PAUSE = 1.3          # the last moment stays on screen this long

    def __init__(self, screen_size: Tuple[int, int], seed: Optional[int] = None,
                 language: Optional[script.Language] = None) -> None:
        super().__init__(screen_size, seed)
        self.words = language or script.ENGLISH
        self.title = self.words.chapters[self.NUMBER - 1].title
        self.HINT = self.words.say(f"hint{self.NUMBER}")
        self.PAUSE_TEXT = (self.words.say("paused"), self.words.say("resume"))
        self.unit = float(min(screen_size))
        self.lives = self.LIVES
        self.combo = 0
        self.won = False
        self.events: List[str] = []
        self.backdrop: Optional[art.Backdrop] = None    # story.py lends its own
        self._ending: Optional[float] = None            # on the round's clock
        self._bark: Tuple[str, float] = ("", 0.0)

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        self.lives = self.LIVES
        self.combo = 0
        self.won = False
        self.events = []
        self._ending = None
        self._bark = ("", 0.0)

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        for t in self.targets:
            if t.dying is not None:
                t.dying += delta
        for s in self.shots:
            s.at += delta

    @property
    def ending(self) -> bool:
        return self._ending is not None

    def _end(self, won: bool) -> None:
        if self._ending is None:
            self.won = won
            self._ending = self.elapsed + self.END_PAUSE
            self.events.append("win" if won else "fail")

    def _finished(self) -> bool:
        return self._ending is not None and self.elapsed >= self._ending

    @property
    def stars(self) -> int:
        """One to three, for a chapter that was won."""
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    @property
    def multiplier(self) -> float:
        return 1.0 + 0.1 * min(self.combo, 10)

    def _award(self, x: float, y: float, now: float, base: float, note: str = "",
               sound: str = "hit") -> int:
        points = int(round(base * self.multiplier))
        self.combo += 1
        self.stats.best_combo = max(self.stats.best_combo, self.combo)
        self.stats.hits += 1
        self.stats.score += points
        label = f"+{points}" + (f"  {note}" if note else "")
        self.shots.append(Shot(x, y, now, points, label, True))
        self.events.append(sound)
        return points

    def _lose_life(self, x: float, y: float, now: float, text: str, sound: str = "lost") -> None:
        self.combo = 0
        self.lives -= 1
        self.shots.append(Shot(x, y, now, 0, text, False))
        self.events.append(sound)
        if self.lives <= 0:
            self._end(False)

    def _miss(self, x: float, y: float, now: float) -> None:
        self.combo = 0
        self.stats.misses += 1
        self.shots.append(Shot(x, y, now, 0, self.words.say("miss"), False))

    def _reap(self, now: float) -> None:
        self.targets = [t for t in self.targets
                        if t.dying is None or now - t.dying <= self.BURST]

    def say(self, text: str, seconds: float = 2.6) -> None:
        """A word from Pip, shown under the score."""
        self._bark = (text, self.elapsed + seconds)

    def _burst(self, canvas, t: Target, now: float, colour) -> None:
        k = (now - t.dying) / self.BURST
        cv2.circle(canvas, (int(t.x), int(t.y)), int(t.radius * (1 + 1.4 * k)),
                   art.dim(colour, max(0.0, 1.0 - k)), 3, cv2.LINE_AA)

    def _dwell_ring(self, canvas, t: Target, radius: float) -> None:
        if t.dwell > 0:
            cv2.ellipse(canvas, (int(t.x), int(t.y)), (int(radius), int(radius)), -90, 0,
                        360 * min(1.0, t.dwell / self.DWELL_TIME), overlay.WHITE, 3, cv2.LINE_AA)

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        if self.backdrop is not None:
            self.backdrop.draw(canvas, self.SCENE, self.now)
        self._draw_scene(canvas)
        super().draw(canvas)
        if self.counting_down:
            h = self.screen_size[1]
            overlay.text_centered(
                canvas, self.words.say("chapter", n=self.NUMBER, title=self.title),
                int(h * 0.26), 1.3, overlay.YELLOW, 2)
            overlay.text_centered(canvas, self._goal(), h // 2 + 140, 0.8, overlay.GREY)

    def _goal(self) -> str:
        """What has to be done, in a line under the countdown."""
        return self.words.say(f"goal{self.NUMBER}")

    def _draw_scene(self, canvas) -> None:
        """Whatever lies under the targets."""

    def _progress(self) -> str:
        return ""

    def _draw_hud(self, canvas) -> None:
        w = self.screen_size[0]
        say = self.words.say
        overlay.text(canvas, say("hud_score", score=self.stats.score), (40, 60), 1.1,
                     overlay.CYAN, 2)
        if self.combo > 1:
            overlay.text(canvas, say("streak", x=f"{self.multiplier:.1f}", n=self.combo),
                         (40, 100), 0.7, overlay.GREEN)
        overlay.text_centered(canvas, self._progress(), 60, 0.9, overlay.WHITE, 2)
        for i in range(self.LIVES):
            c = (w - 60 - i * 44, 50)
            if i < self.lives:
                art.spark(canvas, c[0], c[1], 16, self.now + i)
            else:
                cv2.circle(canvas, c, 12, overlay.DIM, 1, cv2.LINE_AA)
        text, until = self._bark
        if text and self.elapsed < until:
            art.pip(canvas, 62, 160, 26, self.now)
            overlay.text(canvas, text, (104, 168), 0.75, overlay.YELLOW, 2)


# ---------------------------------------------------------------------------
# 1 - Sparks on the Wind
# ---------------------------------------------------------------------------
@dataclass
class Spark(Target):
    phase: float = 0.0
    sway: float = 0.0


class Sparks(Chapter):
    """The lamp's light is blowing away across the sky: catch enough of it."""

    NUMBER, SCENE = 1, "gale"
    LIVES = 8
    NEED = 20
    DWELL_TIME = 0.25        # they do not hold still for longer

    def start(self, now: float) -> None:
        super().start(now)
        self.caught = 0
        self.escaped = 0
        self._next = 0.0

    @property
    def stars(self) -> int:
        return 3 if self.escaped <= 1 else 2 if self.escaped <= 4 else 1

    def _goal(self) -> str:
        return self.words.say("goal1", need=self.NEED, lives=self.LIVES)

    def _ramp(self) -> float:
        return min(1.0, self.caught / self.NEED)

    def spawn(self, now: float) -> Spark:
        w, h = self.screen_size
        k = self._ramp()
        gold = self.rng.random() < 0.12
        radius = self.unit * (0.052 - 0.016 * k) * (0.7 if gold else 1.0)
        speed = w * (0.10 + 0.09 * k) * self.rng.uniform(0.85, 1.2) * (1.35 if gold else 1.0)
        t = Spark(
            x=-radius, y=self.rng.uniform(0.16, 0.80) * h, radius=radius, born=now,
            lifetime=math.inf, kind=GOLD if gold else "spark",
            vx=speed, vy=self.rng.uniform(-0.03, 0.03) * h,
            phase=self.rng.uniform(0, 2 * math.pi), sway=self.rng.uniform(0.03, 0.08) * h,
        )
        self.targets.append(t)
        return t

    def _hit(self, target: Target, now: float, how: str) -> None:
        target.dying = now
        gold = target.kind == GOLD
        base = (20 + 30 * target.vx / (0.25 * self.screen_size[0])) * (3 if gold else 1)
        self._award(target.x, target.y, now, base, "x3" if gold else "",
                    "gold" if gold else "hit")
        self.stats.reactions.append(target.age(now))
        self.caught += 1
        if self.caught >= self.NEED:
            self._end(True)

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        w, h = self.screen_size
        if not self.ending:
            while self.elapsed >= self._next:
                self.spawn(now)
                self._next += 1.9 - 0.7 * self._ramp()
        for t in list(self.targets):
            if t.dying is not None:
                continue
            t.x += t.vx * dt
            t.y += (t.vy + math.cos(t.age(now) * 2.2 + t.phase) * t.sway) * dt
            t.y = min(max(t.y, h * 0.12), h * 0.88)
            if t.x > w + t.radius:
                self.targets.remove(t)
                if not self.ending:
                    self.escaped += 1
                    self.stats.escaped += 1
                    self._lose_life(w - 170, t.y, now, self.words.say("blown"))
        self._reap(now)
        if not self.ending:
            self._aim(now, dt, point)

    def _progress(self) -> str:
        return self.words.say("sparks", a=self.caught, b=self.NEED)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        colour = overlay.WHITE if t.kind == GOLD else art.SPARK
        if t.dying is not None:
            self._burst(canvas, t, now, colour)
            return
        art.spark(canvas, t.x, t.y, t.radius, now + t.phase, colour, (1.0, 0.0))
        if t.kind == GOLD:
            overlay.text(canvas, "x3", (int(t.x + t.radius + 8), int(t.y - t.radius)), 0.6, colour)
        self._dwell_ring(canvas, t, t.radius + 8)


# ---------------------------------------------------------------------------
# 2 - What the Dark Wants
# ---------------------------------------------------------------------------
@dataclass
class Creeper(Target):
    phase: float = 0.0
    speed: float = 0.0
    goal: Tuple[float, float] = (0.0, 0.0)
    drift: float = 0.0       # seconds it is still being thrown sideways


class Siege(Chapter):
    """Gloamlings creep in from every side towards the cage in the middle,
    in three waves. Each one that gets there puts out a spark."""

    NUMBER, SCENE = 2, "room"
    LIVES = 5
    # per wave: how many, seconds between them, speed, share that are brutes
    WAVES = ((6, 1.9, 0.070, 0.0), (7, 1.8, 0.080, 0.15), (9, 1.6, 0.090, 0.25))
    BREATHER = 2.4           # seconds between waves
    MOTH_EVERY = 5.5

    def start(self, now: float) -> None:
        super().start(now)
        self.wave = 0
        self._left = self.WAVES[0][0]        # still to come in this wave
        self._next = 0.0
        self._banner = self.BREATHER         # the wave is announced until then
        self._next += self._banner
        self._moth = self._banner + 2.0

    @property
    def centre(self) -> Tuple[float, float]:
        return self.screen_size[0] / 2.0, self.screen_size[1] / 2.0

    @property
    def cage_radius(self) -> float:
        return self.unit * 0.13

    @property
    def stars(self) -> int:
        return 3 if self.lives >= self.LIVES else 2 if self.lives >= 3 else 1

    def _edge_point(self) -> Tuple[float, float]:
        w, h = self.screen_size
        side = self.rng.choice("lrtb" if self.rng.random() < 0.8 else "lr")
        if side in "lr":
            return (0.0 if side == "l" else w), self.rng.uniform(0.1, 0.9) * h
        # From above or below the cage is close: those come in at the corners.
        x = self.rng.uniform(0.04, 0.28) * w
        return (x if self.rng.random() < 0.5 else w - x), (0.0 if side == "t" else h * 0.94)

    def spawn(self, now: float, kind: str = GLOAMLING,
              at: Optional[Tuple[float, float]] = None, scale: float = 1.0) -> Creeper:
        _, _, speed, _ = self.WAVES[self.wave]
        radius = self.unit * 0.05 * scale * (1.5 if kind == BRUTE else 1.0)
        t = Creeper(
            *(at or self._edge_point()), radius=radius, born=now, lifetime=math.inf, kind=kind,
            phase=self.rng.uniform(0, 2 * math.pi), goal=self.centre,
            speed=self.unit * speed * (0.72 if kind == BRUTE else 1.0) * self.rng.uniform(0.9, 1.1),
        )
        self.targets.append(t)
        return t

    def _spawn_moth(self, now: float) -> None:
        w, h = self.screen_size
        side = self.rng.choice((-1, 1))
        y = self.rng.uniform(0.18, 0.8) * h
        t = Creeper(
            x=-40.0 if side > 0 else w + 40.0, y=y, radius=self.unit * 0.042, born=now,
            lifetime=math.inf, kind=MOTH, phase=self.rng.uniform(0, 2 * math.pi),
            goal=(w + 80.0 if side > 0 else -80.0, self.rng.uniform(0.18, 0.8) * h),
            speed=self.unit * 0.14,
        )
        self.targets.append(t)

    def _hit(self, target: Target, now: float, how: str) -> None:
        target.dying = now
        if target.kind == MOTH:
            self.stats.decoys += 1
            self.stats.score = max(0, self.stats.score - 25)
            self._lose_life(target.x, target.y, now, self.words.say("friend"), "bad")
            self.say(self.words.say("moths"))
            return
        self.stats.reactions.append(target.age(now))
        if target.kind == BRUTE:
            # A big one comes apart into two small ones, thrown sideways.
            self._award(target.x, target.y, now, 30, self.words.say("split"))
            dx, dy = target.goal[0] - target.x, target.goal[1] - target.y
            norm = math.hypot(dx, dy) or 1.0
            for side in (-1, 1):
                small = self.spawn(now, GLOAMLING, (target.x, target.y), 0.75)
                small.vx, small.vy = -dy / norm * side * small.speed * 2.2, \
                    dx / norm * side * small.speed * 2.2
                small.drift = 0.45
        else:
            far = math.hypot(target.x - target.goal[0], target.y - target.goal[1])
            self._award(target.x, target.y, now, 15 + 25 * min(1.0, far / (self.unit * 0.6)))

    def _foes(self) -> List[Target]:
        return [t for t in self.targets if t.kind != MOTH and t.dying is None]

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        w, h = self.screen_size
        clock = self.elapsed
        count, gap, _, brutes = self.WAVES[self.wave]
        if not self.ending:
            while self._left > 0 and clock >= self._next:
                self.spawn(now, BRUTE if self.rng.random() < brutes else GLOAMLING)
                self._left -= 1
                self._next += gap
            if clock >= self._moth:
                self._spawn_moth(now)
                self._moth = clock + self.MOTH_EVERY * self.rng.uniform(0.7, 1.3)
            if self._left == 0 and not self._foes():
                if self.wave + 1 >= len(self.WAVES):
                    self._end(True)
                else:
                    self.wave += 1
                    self._left = self.WAVES[self.wave][0]
                    self._banner = clock + self.BREATHER
                    self._next = self._banner
                    self._moth = max(self._moth, self._banner + 1.0)

        for t in list(self.targets):
            if t.dying is not None:
                continue
            dx, dy = t.goal[0] - t.x, t.goal[1] - t.y
            far = math.hypot(dx, dy) or 1.0
            if t.drift > 0:
                t.drift -= dt
                t.x += t.vx * dt
                t.y += t.vy * dt
                t.x, t.y = min(max(t.x, 0.0), w), min(max(t.y, 0.0), h * 0.94)
                continue
            # Straight for the goal, weaving a little on the way.
            weave = math.sin(t.age(now) * (5.0 if t.kind == MOTH else 2.0) + t.phase) \
                * (0.9 if t.kind == MOTH else 0.35)
            t.x += (dx / far - dy / far * weave) * t.speed * dt
            t.y += (dy / far + dx / far * weave) * t.speed * dt
            if t.kind == MOTH:
                if far < t.speed * 0.2:
                    self.targets.remove(t)
            elif far < self.cage_radius + t.radius * 0.4:
                t.dying = now
                if not self.ending:
                    self.stats.escaped += 1
                    self._lose_life(t.x, t.y, now, self.words.say("spark_out"))
        self._reap(now)
        if not self.ending:
            self._aim(now, dt, point)

    def _progress(self) -> str:
        return self.words.say("wave", a=self.wave + 1, b=len(self.WAVES))

    def _draw_scene(self, canvas) -> None:
        cx, cy = self.centre
        art.cage(canvas, cx, cy, self.cage_radius, max(0, self.lives) * 2, self.now)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        if t.dying is not None:
            self._burst(canvas, t, now, overlay.GREEN if t.kind == MOTH else art.FOG)
            return
        if t.kind == MOTH:
            art.moth(canvas, t.x, t.y, t.radius, now, t.phase)
        else:
            art.gloamling(canvas, t.x, t.y, t.radius, now, t.phase,
                          (t.goal[0] - t.x, t.goal[1] - t.y), eyes=3 if t.kind == BRUTE else 2)
        self._dwell_ring(canvas, t, t.radius + 10)

    def _draw_hud(self, canvas) -> None:
        super()._draw_hud(canvas)
        if self.elapsed < self._banner and not self.ending and not self.counting_down:
            overlay.text_centered(canvas, self.words.say("wave_banner", n=self.wave + 1),
                                  int(self.screen_size[1] * 0.3), 2.0, overlay.CYAN, 4)


# ---------------------------------------------------------------------------
# 3 - Ships in the Dark
# ---------------------------------------------------------------------------
@dataclass
class Boat(Target):
    phase: float = 0.0
    need: float = 1.5        # seconds of light before the skipper sees it
    guided: bool = False
    lane: float = 0.0        # the row it would sail along if nobody helped


class Ships(Chapter):
    """Boats cross the bay towards a reef they cannot see. Light held on a
    boat for long enough turns it for the channel; then on to the next."""

    NUMBER, SCENE = 3, "bay"
    LIVES = 3
    FLEET = 9
    FLASH = False            # a flash guides nobody: the light has to stay

    def start(self, now: float) -> None:
        super().start(now)
        self.sent = 0
        self.safe = 0
        self.lost = 0
        self._next = 0.0
        self._lit: Optional[Boat] = None
        self._fog = [(self.rng.random(), self.rng.uniform(0.3, 0.9), self.rng.uniform(0.6, 1.0))
                     for _ in range(3)]

    @property
    def stars(self) -> int:
        return 3 if self.lost == 0 else 2 if self.lost == 1 else 1

    @property
    def channel(self) -> float:
        return self.screen_size[1] * sum(art.CHANNEL) / 2.0

    def spawn(self, now: float) -> Boat:
        w, h = self.screen_size
        k = self.sent / max(1, self.FLEET - 1)
        radius = self.unit * 0.06
        # Never already lined up with the gap: that boat would need no keeper.
        for _ in range(60):
            y = self.rng.uniform(art.HORIZON + 0.14, 0.9) * h
            clear = not h * art.CHANNEL[0] - radius < y < h * art.CHANNEL[1] + radius
            apart = all(abs(y - t.y) > radius * 1.6 for t in self.targets if t.x < w * 0.25)
            if clear and apart:
                break
        t = Boat(x=-radius, y=y, radius=radius, born=now, lifetime=math.inf,
                 vx=w * art.REEF_X / (9.5 - 3.0 * k), phase=self.rng.uniform(0, 6.28),
                 need=1.4 + 0.5 * k, lane=y)
        self.targets.append(t)
        self.sent += 1
        return t

    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        live = [t for t in self.targets
                if t.dying is None and not t.guided and t.contains(x, y, slack)]
        if not live:
            return None
        return min(live, key=lambda t: math.hypot(t.x - x, t.y - y))

    def _aim(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        self._beam(now, point)
        if point is None:
            if not self._beam_on:
                self._lit = None
            boat = self._lit          # a dropped frame: the light is still there
        else:
            boat = self._lit = self._target_at(*point, slack=self.unit * 0.025)
        if boat is None or boat.guided or boat.dying is not None:
            return
        boat.dwell += dt
        if boat.dwell >= boat.need:
            self._guide(boat, now)

    def _guide(self, boat: Boat, now: float) -> None:
        w = self.screen_size[0]
        boat.guided = True
        reef = w * art.REEF_X
        left = max(0.0, reef - boat.x)
        # Steering takes room: the sooner it is turned, the more it is worth.
        self._award(boat.x, boat.y - boat.radius, now, 60 + 90 * left / reef,
                    self.words.say("on_course"))
        self.stats.reactions.append(boat.age(now))
        boat.vy = (self.channel - boat.y) * boat.vx / max(left, boat.radius)

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        w, h = self.screen_size
        k = self.sent / max(1, self.FLEET - 1)
        if not self.ending and self.sent < self.FLEET and self.elapsed >= self._next:
            self.spawn(now)
            self._next = self.elapsed + 5.2 - 2.4 * k
        reef = w * art.REEF_X
        for t in list(self.targets):
            if t.dying is not None:
                continue
            t.x += t.vx * dt
            if t.guided:
                t.y += t.vy * dt
                if (t.vy > 0) == (t.y > self.channel):      # arrived on the line
                    t.y, t.vy = self.channel, 0.0
                if t.x > w * art.HARBOUR_X - t.radius:
                    self.targets.remove(t)
                    self.safe += 1
            elif t.x >= reef - t.radius * 0.8:
                t.dying = now
                self.lost += 1
                self.stats.escaped += 1
                self._lose_life(t.x, t.y - t.radius, now, self.words.say("reef"))
        self._reap(now)
        if not self.ending and self.sent >= self.FLEET \
                and not any(t.dying is None for t in self.targets):
            self._end(True)
        if not self.ending:
            self._aim(now, dt, point)

    def _progress(self) -> str:
        return self.words.say("home", a=self.safe, b=self.FLEET)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        if t.dying is not None:
            self._burst(canvas, t, now, overlay.YELLOW)
            return
        colour = overlay.GREEN if t.guided else overlay.CYAN
        art.boat(canvas, t.x, t.y + t.radius * 0.3, t.radius * 0.62, colour, now, t.phase)
        if not t.guided:
            r = int(t.radius + 10)
            cv2.circle(canvas, (int(t.x), int(t.y)), r, overlay.DIM, 1, cv2.LINE_AA)
            if t.dwell > 0:
                cv2.ellipse(canvas, (int(t.x), int(t.y)), (r, r), -90, 0,
                            360 * min(1.0, t.dwell / t.need), overlay.WHITE, 3, cv2.LINE_AA)

    def draw(self, canvas) -> None:
        super().draw(canvas)
        if self.state == "over":
            return
        # The Gloam's fog, drifting over the water and whatever is on it.
        w, h = self.screen_size
        for u, v, s in self._fog:
            x = (u - self.elapsed * 0.035 * s) % 1.3 * w - w * 0.15
            art.mist(canvas, x, v * h, self.unit * 0.13 * s, self.now, u * 9)


# ---------------------------------------------------------------------------
# 4 - The Keepers' Stars
# ---------------------------------------------------------------------------
# name, the stars in the order they are traced, lines that complete the
# picture, and stars that belong to no constellation at all
SKY = (
    ("THE BOAT",
     ((0.30, 0.56), (0.40, 0.72), (0.60, 0.72), (0.70, 0.56), (0.50, 0.26)),
     ((3, 0), (4, 0)),
     ()),
    ("THE GULL",
     ((0.20, 0.52), (0.35, 0.33), (0.50, 0.50), (0.65, 0.33), (0.80, 0.52)),
     (),
     ((0.50, 0.24), (0.28, 0.74))),
    ("THE LAMP",
     ((0.39, 0.80), (0.44, 0.46), (0.21, 0.36), (0.50, 0.22), (0.79, 0.36), (0.56, 0.46),
      (0.61, 0.80)),
     ((1, 5), (0, 6)),
     ((0.26, 0.62), (0.74, 0.64), (0.50, 0.62))),
)


@dataclass
class Star(Target):
    order: int = -1          # its place in the constellation; -1 for a stray
    lit: bool = False


class Constellations(Chapter):
    """Three constellations. Each is shown star by star, then has to be
    traced back in the same order."""

    NUMBER, SCENE = 4, "sky"
    LIVES = 4
    STEP = 0.75              # seconds between two stars of the pattern
    IDLE = 14.0              # shown again, free, to a keeper who is stuck
    SHOW, PLAY, DONE = "show", "play", "done"

    def start(self, now: float) -> None:
        super().start(now)
        self.index = 0
        self.mistakes = 0
        self._set_up(now)

    @property
    def stars(self) -> int:
        return 3 if self.mistakes == 0 else 2 if self.mistakes <= 2 else 1

    @property
    def name(self) -> str:
        return self.words.say(f"sky{self.index + 1}")

    @property
    def length(self) -> int:
        return len(SKY[self.index][1])

    def _set_up(self, now: float) -> None:
        w, h = self.screen_size
        _, path, _, strays = SKY[self.index]
        self.targets = [
            Star(x=u * w, y=v * h, radius=self.unit * 0.05, born=now, lifetime=math.inf, order=i)
            for i, (u, v) in enumerate(path)
        ] + [
            Star(x=u * w, y=v * h, radius=self.unit * 0.05, born=now, lifetime=math.inf)
            for u, v in strays
        ]
        self.progress = 0
        self._idle_at = self.elapsed
        self._show(self.elapsed + 0.8)

    def _show(self, at: float) -> None:
        self.phase = self.SHOW
        self._phase_at = at
        self._dwelt = None
        for t in self.targets:
            t.dwell = 0.0

    def _shown(self) -> int:
        """How many stars of the pattern have lit up so far."""
        return int((self.elapsed - self._phase_at) / self.STEP) + 1 \
            if self.elapsed >= self._phase_at else 0

    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        live = [t for t in self.targets if not t.lit and t.contains(x, y, slack)]
        if not live:
            return None
        return min(live, key=lambda t: math.hypot(t.x - x, t.y - y))

    def _miss(self, x: float, y: float, now: float) -> None:
        pass                                    # the sky between the stars is free

    def _hit(self, target: Target, now: float, how: str) -> None:
        self._idle_at = self.elapsed
        if target.order != self.progress:
            self.mistakes += 1
            self.stats.misses += 1
            self._lose_life(target.x, target.y - target.radius, now,
                            self.words.say("not_that"), "bad")
            if not self.ending:
                self._show(self.elapsed + 0.9)
            return
        target.lit = True
        self.progress += 1
        self._award(target.x, target.y - target.radius, now, 40, sound="star")
        if self.progress >= self.length:
            self.phase = self.DONE
            self._phase_at = self.elapsed

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        clock = self.elapsed
        if self.phase == self.SHOW:
            self._beam(now, point)
            if clock >= self._phase_at + self.STEP * (self.length + 0.6):
                self.phase = self.PLAY
                self._idle_at = clock
        elif self.phase == self.PLAY:
            if not self.ending:
                self._aim(now, dt, point)
            if self.phase == self.PLAY and clock - self._idle_at > self.IDLE:
                self._show(clock)
        elif clock < self._phase_at + 2.2 or self.ending:
            self._beam(now, point)
        else:
            if self.index + 1 >= len(SKY):
                self._end(True)
            else:
                self.index += 1
                self._set_up(now)

    def _progress(self) -> str:
        return self.words.say("sky", name=self.name, a=self.index + 1, b=len(SKY))

    def _draw_scene(self, canvas) -> None:
        path = sorted((t for t in self.targets if t.order >= 0), key=lambda t: t.order)
        if not path:
            return
        if self.phase == self.SHOW:
            upto = min(self._shown(), len(path))
        else:
            upto = self.progress
        done = self.phase == self.DONE
        colour = overlay.GREEN if done else overlay.CYAN if self.phase == self.SHOW \
            else art.dim(overlay.CYAN, 0.7)
        pairs = [(i, i + 1) for i in range(upto - 1)]
        if done:
            pairs += list(SKY[self.index][2])
        for a, b in pairs:
            cv2.line(canvas, (int(path[a].x), int(path[a].y)), (int(path[b].x), int(path[b].y)),
                     colour, 2, cv2.LINE_AA)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        showing = self.phase == self.SHOW and 0 <= t.order < self._shown()
        bright = t.lit or showing or self.phase == self.DONE and t.order >= 0
        size = self.unit * (0.03 if bright else 0.022)
        twinkle = 0.8 + 0.2 * math.sin(now * 3.0 + t.x)
        if showing and t.order == self._shown() - 1:
            # The one lighting up right now, with its place in the order.
            size *= 1.5
            cv2.circle(canvas, (int(t.x), int(t.y)), int(t.radius * 1.15), overlay.CYAN, 2,
                       cv2.LINE_AA)
        colour = art.SPARK if bright else art.dim(overlay.WHITE, 0.55 * twinkle)
        art.star(canvas, t.x, t.y, size, colour)
        if showing:
            overlay.text(canvas, str(t.order + 1), (int(t.x + t.radius * 0.7),
                                                    int(t.y - t.radius * 0.6)), 0.9, overlay.CYAN, 2)
        if self.phase == self.PLAY and not t.lit:
            cv2.circle(canvas, (int(t.x), int(t.y)), int(t.radius), overlay.DIM, 1, cv2.LINE_AA)
            self._dwell_ring(canvas, t, t.radius - 5)

    def _draw_hud(self, canvas) -> None:
        super()._draw_hud(canvas)
        h = self.screen_size[1]
        if self.phase == self.SHOW:
            text, colour = self.words.say("watch", name=self.name), overlay.CYAN
        elif self.phase == self.PLAY:
            text = self.words.say("trace", a=self.progress + 1, b=self.length)
            colour = overlay.WHITE
        else:
            text, colour = self.name, overlay.GREEN
        overlay.text_centered(canvas, text, int(h * 0.915), 0.9, colour, 2)


# ---------------------------------------------------------------------------
# 5 - The Heart of the Storm
# ---------------------------------------------------------------------------
class Storm(Chapter):
    """The Gloam itself. Bright knots open in it, a moment at a time, and each
    one struck hurts it; each one left sends a gloamling at the lamp. When it
    has nothing left, its eye opens - and has to be held in the light."""

    NUMBER, SCENE = 5, "dark"
    LIVES = 5
    HEALTH = 18
    # per phase: knots open at once, seconds a knot stays open, seconds between
    # two openings, seconds between gloamlings sent unprovoked (0: none)
    PHASES = ((1, 3.2, 1.4, 0.0), (1, 2.8, 1.2, 7.0), (2, 2.8, 1.3, 6.0))
    HOLD = 2.0               # seconds of light the eye takes
    THROW_GAP = 3.0          # it never sends gloamlings faster than this, or a
                             # keeper who fell behind could never catch up
    END_PAUSE = 1.8

    def start(self, now: float) -> None:
        super().start(now)
        self.health = self.HEALTH
        self.final = False
        self._next_knot = 1.0
        self._next_volley = 0.0
        self._thrown = -9.0
        self._taunt = ("", 0.0)
        self._phase = 0
        self._flinch = -9.0
        self.__dict__.pop("DWELL_TIME", None)       # a restart: back to the class's

    @property
    def stars(self) -> int:
        return 3 if self.lives >= self.LIVES else 2 if self.lives >= 3 else 1

    @property
    def heart(self) -> Tuple[float, float]:
        return self.screen_size[0] / 2.0, self.screen_size[1] * 0.40

    @property
    def lamp(self) -> Tuple[float, float]:
        return self.screen_size[0] / 2.0, self.screen_size[1] * 0.86

    @property
    def phase(self) -> int:
        left = self.health / self.HEALTH
        return 0 if left > 2 / 3 else 1 if left > 1 / 3 else 2

    def _knot_spots(self) -> List[Tuple[float, float]]:
        cx, cy = self.heart
        r = self.unit * 0.30
        return [(cx + r * 1.25 * math.cos(a), cy + r * 0.8 * math.sin(a))
                for a in (math.pi * (0.5 + k / 3.0) + math.pi / 6 for k in range(6))]

    def _open_knot(self, now: float) -> None:
        taken = [(t.x, t.y) for t in self.targets if t.kind == KNOT]
        free = [p for p in self._knot_spots() if p not in taken]
        if not free:
            return
        _, life, _, _ = self.PHASES[self.phase]
        x, y = self.rng.choice(free)
        self.targets.append(Target(x=x, y=y, radius=self.unit * (0.052 - 0.006 * self.phase),
                                   born=now, lifetime=life, kind=KNOT))

    def _throw(self, now: float, at: Tuple[float, float]) -> None:
        if self.elapsed - self._thrown < self.THROW_GAP:
            return
        self._thrown = self.elapsed
        self.targets.append(Creeper(
            x=at[0], y=at[1], radius=self.unit * 0.045, born=now, lifetime=math.inf,
            kind=GLOAMLING, phase=self.rng.uniform(0, 6.28), goal=self.lamp,
            speed=self.unit * (0.085 + 0.012 * self.phase)))

    def _hit(self, target: Target, now: float, how: str) -> None:
        if target.kind == EYE:
            if how == "flash":
                return                      # a flash is not enough for this one
            target.dying = now
            self.stats.reactions.append(target.age(now))
            self._award(target.x, target.y, now, 500, self.words.say("lit"), "boom")
            self._end(True)
            return
        target.dying = now
        self.stats.reactions.append(target.age(now))
        if target.kind == GLOAMLING:
            self._award(target.x, target.y, now, 20)
            return
        self.health -= 1
        self._flinch = self.elapsed
        self._award(target.x, target.y, now, 40 + 40 * target.remaining(now))
        if self.health <= 0:
            self._last_stand(now)

    def _last_stand(self, now: float) -> None:
        """Everything it had is gone: only the eye is left, wide open."""
        for t in self.targets:
            if t.dying is None:
                t.dying = now
        cx, cy = self.heart
        self.final = True
        self.DWELL_TIME = self.HOLD
        self._dwelt = None
        self.targets.append(Target(x=cx, y=cy, radius=self.unit * 0.085, born=now,
                                   lifetime=math.inf, kind=EYE))
        self._taunt = (self.words.say("eye"), math.inf)
        self.events.append("boom")

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        clock = self.elapsed
        if not self.final and not self.ending:
            n_open, _, gap, volley = self.PHASES[self.phase]
            if self.phase != self._phase:
                self._phase = self.phase
                self._taunt = (self.words.say(f"taunt{self.phase}"), clock + 2.4)
                self._next_volley = clock + volley
                self.events.append("boom")
            knots = [t for t in self.targets if t.kind == KNOT and t.dying is None]
            if len(knots) < n_open and clock >= self._next_knot:
                self._open_knot(now)
                self._next_knot = clock + gap
            if volley and clock >= self._next_volley:
                cx, cy = self.heart
                self._throw(now, (cx + self.rng.uniform(-1, 1) * self.unit * 0.2,
                                  cy + self.unit * 0.12))
                self._next_volley = clock + volley

        lx, ly = self.lamp
        for t in list(self.targets):
            if t.dying is not None:
                continue
            if t.kind == KNOT:
                if t.age(now) >= t.lifetime:
                    # A knot nobody struck closes - and something comes of it.
                    self.targets.remove(t)
                    self.combo = 0
                    self._throw(now, (t.x, t.y))
            elif t.kind == GLOAMLING:
                dx, dy = lx - t.x, ly - t.y
                far = math.hypot(dx, dy) or 1.0
                weave = math.sin(t.age(now) * 2.0 + t.phase) * 0.35
                t.x += (dx / far - dy / far * weave) * t.speed * dt
                t.y += (dy / far + dx / far * weave) * t.speed * dt
                if far < self.unit * 0.07 + t.radius * 0.4:
                    t.dying = now
                    if not self.ending:
                        self.stats.escaped += 1
                        self._lose_life(t.x, t.y, now, self.words.say("spark_out"))
        self._reap(now)
        if not self.ending:
            self._aim(now, dt, point)

    def _progress(self) -> str:
        return ""

    def _draw_scene(self, canvas) -> None:
        w, h = self.screen_size
        cx, cy = self.heart
        now = self.now
        r = self.unit * 0.24
        shake = self.unit * 0.012 * max(0.0, 1.0 - (self.elapsed - self._flinch) / 0.3)
        cx += shake * math.sin(now * 90)
        fade = 1.0
        if self.ending and self.won:
            fade = max(0.0, (self._ending - self.elapsed) / self.END_PAUSE)
        # Thinner as it weakens.
        strength = (0.5 + 0.4 * self.health / self.HEALTH) * fade
        for k in range(8):
            a = now * 0.3 + k * math.pi / 4
            art.gloamling(canvas, cx + math.cos(a) * r * 1.2, cy + math.sin(a) * r * 0.7,
                          r * 0.5, now, k * 1.3, eyes=0, strength=strength * 0.7)
        art.gloamling(canvas, cx, cy, r, now * 0.5, 0.0, eyes=0, strength=strength)
        for x, y in self._knot_spots():             # where a knot may open
            cv2.circle(canvas, (int(x), int(y)), 6, art.dim(art.FOG, 0.6 * fade), 1, cv2.LINE_AA)
        if not self.final:                          # the eye, all but shut
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(r * 0.42), int(r * 0.07)), 0, 0, 360,
                        art.dim(overlay.WHITE, 0.6), 2, cv2.LINE_AA)
        lx, ly = self.lamp
        art.cage(canvas, lx, ly, self.unit * 0.07, max(0, self.lives) * 2, now)
        if self.ending and self.won:
            # The lamp catches: light pours out of it over everything.
            glow = 1.0 - fade
            layer = np.zeros_like(canvas)
            for k in range(10):
                a = now * 0.5 + k * math.pi / 5
                reach, half = self.unit * 2.2 * glow, self.unit * 0.09 * glow
                tip = (lx + math.cos(a) * reach, ly + math.sin(a) * reach)
                ray = [(lx, ly), (tip[0] - math.sin(a) * half, tip[1] + math.cos(a) * half),
                       (tip[0] + math.sin(a) * half, tip[1] - math.cos(a) * half)]
                cv2.fillPoly(layer, [np.array(ray, np.int32)], art.dim(art.SPARK, 0.3), cv2.LINE_AA)
            cv2.add(canvas, layer, dst=canvas)
            cv2.circle(canvas, (int(lx), int(ly)), int(self.unit * 0.1 * glow), art.SPARK, -1,
                       cv2.LINE_AA)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        if t.dying is not None:
            self._burst(canvas, t, now, art.FOG if t.kind == GLOAMLING else overlay.CYAN)
            return
        x, y, r = int(t.x), int(t.y), int(t.radius)
        if t.kind == GLOAMLING:
            art.gloamling(canvas, t.x, t.y, t.radius, now, t.phase, (t.goal[0] - t.x, t.goal[1] - t.y))
            self._dwell_ring(canvas, t, t.radius + 10)
        elif t.kind == KNOT:
            left = t.remaining(now)
            cv2.circle(canvas, (x, y), r, art.dim(overlay.CYAN, 0.16), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), r, overlay.CYAN, 3, cv2.LINE_AA)
            art.star(canvas, t.x, t.y, r * 0.55, overlay.WHITE)
            cv2.ellipse(canvas, (x, y), (r + 9, r + 9), -90, 0, 360 * left,
                        overlay.CYAN if left > 0.3 else overlay.YELLOW, 2, cv2.LINE_AA)
            self._dwell_ring(canvas, t, t.radius - 6)
        else:
            lid = 0.55 + 0.1 * math.sin(now * 2.0)
            cv2.ellipse(canvas, (x, y), (r, int(r * lid)), 0, 0, 360, (14, 12, 6), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (x, y), (r, int(r * lid)), 0, 0, 360, overlay.WHITE, 3, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), int(r * 0.34), overlay.CYAN, 3, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), int(r * 0.14), overlay.WHITE, -1, cv2.LINE_AA)
            self._dwell_ring(canvas, t, t.radius + 16)

    def _draw_hud(self, canvas) -> None:
        super()._draw_hud(canvas)
        w, h = self.screen_size
        bar = int(w * 0.3)
        x0 = (w - bar) // 2
        overlay.text_centered(canvas, self.words.say("boss"), 40, 0.7, art.FOG, 2)
        cv2.rectangle(canvas, (x0, 52), (x0 + bar, 70), overlay.DIM, 1)
        cv2.rectangle(canvas, (x0, 52), (x0 + int(bar * max(0, self.health) / self.HEALTH), 70),
                      art.FOG, -1)
        text, until = self._taunt
        if text and self.elapsed < until and not self.ending:
            overlay.text_centered(canvas, text, int(h * 0.70), 1.3,
                                  overlay.YELLOW if self.final else art.FOG, 3)
