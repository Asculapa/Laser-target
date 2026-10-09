"""Jar shoot: two games for two players, on the split screen of the range
duel (duel.py) - a half each, and a dot counts for the half it is in.

* Jar range - jars come up on shelves, the same in both halves at the same
  moments. They come in three sizes, and each has its own time to be shot:
  what is in it drains away as that time runs out. A small jar is hard to
  hit and pays the most; a big one is easy, and gone the soonest.
* Quick draw - both halves wait; then a jar appears in each, in the same
  place, and whoever breaks theirs first takes the point. Shooting before it
  is up gives the point away. First to five.

The jars are drawn here, under the two rules of the rest of the app's
pictures (see story_art.py): no red, so there is no jam in them; and nothing
large is bright, because the camera has to find a red dot on top of the jar.
Redness is red above the stronger of green and blue, so a dot on a bright
green or blue fill has none left. What is in a jar is dim, and its colour
shows bright only in thin lines: the outline, the surface, the number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import overlay
from .duel import Duel, Lane, Popup
from .game import Shot, Target

ASSETS = Path(__file__).parent / "assets" / "jars"
SHELVES = (0.42, 0.65, 0.88)        # where the shelves are, down the screen
COLUMNS = 3
WIDTH = 0.7                         # a jar's width, in heights
WAKE = 0.3                          # the part of its time left when a jar starts to shake

SHELF = (58, 68, 52)
GLASS = (46, 50, 40)
EDGE = (205, 205, 178)
SHINE = (235, 235, 210)
LID = (96, 100, 84)
LABEL = (64, 70, 58)
DIM = 0.42                          # how bright what is in a jar is, against its colour
LARGE = 130                         # no channel brighter than this on anything large

Point = Tuple[float, float]


@dataclass(frozen=True)
class Kind:
    height: float           # as a share of the screen's height
    points: int             # before the bonus for speed and the streak
    lifetime: float         # seconds it stays up, at the start of a round
    colour: Tuple[int, int, int]    # bright, so only for thin lines


KINDS = {
    "big": Kind(0.185, 10, 1.9, (255, 130, 90)),        # blue
    "mid": Kind(0.135, 20, 2.5, (120, 230, 110)),       # green
    "small": Kind(0.088, 40, 3.1, overlay.CYAN),
    "gold": Kind(0.088, 100, 1.3, overlay.YELLOW),      # rare, and quick to go
}
SIZES = ("big", "mid", "small")


@dataclass(frozen=True)
class JarPopup(Popup):
    scale: float = 1.0      # jars get smaller as the round goes on


@dataclass
class Shard:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    angle: float
    spin: float
    glass: bool             # a piece of the jar, or a drop of what was in it


@dataclass
class Jar(Target):
    """A jar standing on a shelf; (x, y) is the middle of it."""
    look: str = "big"
    height: int = 0
    station: int = 0
    shot: bool = False      # it went to a shot, not to the clock
    shards: List[Shard] = field(default_factory=list)

    @property
    def width(self) -> float:
        return WIDTH * self.height

    @property
    def foot(self) -> float:
        return self.y + self.height / 2

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return abs(x - self.x) <= self.width / 2 + slack \
            and abs(y - self.y) <= self.height / 2 + slack


# -- drawing ----------------------------------------------------------------
def _rounded(x0: float, y0: float, x1: float, y1: float, top: float,
             bottom: float) -> np.ndarray:
    """A rectangle whose top and bottom corners are rounded by these radii."""
    points = []
    for (cx, cy, r, a0) in ((x1 - bottom, y1 - bottom, bottom, 0),
                            (x0 + bottom, y1 - bottom, bottom, 90),
                            (x0 + top, y0 + top, top, 180),
                            (x1 - top, y0 + top, top, 270)):
        for a in np.linspace(a0, a0 + 90, 7):
            points.append((cx + r * math.cos(math.radians(a)),
                           cy + r * math.sin(math.radians(a))))
    return np.array(points, np.int32).reshape(-1, 1, 2)


def _dim(colour, fade: float):
    return tuple(int(c * fade) for c in colour)


def draw_jar(canvas, x: float, foot: float, height: float, look: str, level: float,
             fade: float = 1.0, marked: bool = True) -> None:
    """A jar with its lid on, standing at (x, foot), `level` full - and, if
    `marked`, what it is worth on its label."""
    if height < 4:
        return
    kind = KINDS[look]
    w = WIDTH * height
    x0, x1 = x - w / 2, x + w / 2
    shoulder = foot - 0.80 * height
    thick = max(1, int(round(height / 70)))

    body = _rounded(x0, shoulder, x1, foot, 0.22 * w, 0.14 * w)
    cv2.fillPoly(canvas, [body], _dim(GLASS, fade), cv2.LINE_AA)
    pad = max(2.0, 0.07 * w)
    bottom, top = foot - pad, shoulder + pad
    if level > 0:
        surface = bottom - (bottom - top) * min(1.0, level)
        liquid = _rounded(x0 + pad, surface, x1 - pad, bottom, 0, 0.1 * w)
        cv2.fillPoly(canvas, [liquid], _dim(kind.colour, DIM * fade), cv2.LINE_AA)
        cv2.line(canvas, (int(x0 + pad), int(surface)), (int(x1 - pad), int(surface)),
                 _dim(kind.colour, fade), thick, cv2.LINE_AA)
    # The label, with what the jar is worth on it.
    ly0, ly1 = foot - 0.56 * height, foot - 0.34 * height
    cv2.rectangle(canvas, (int(x0 + 1), int(ly0)), (int(x1 - 1), int(ly1)),
                  _dim(LABEL, fade), -1)
    if marked and height >= 40:
        words = "GOLD" if look == "gold" else str(kind.points)
        _label(canvas, words, (x, (ly0 + ly1) / 2), 0.8 * w, 0.62 * (ly1 - ly0),
               _dim(kind.colour, 0.7 * fade), thick)
    cv2.polylines(canvas, [body], True, _dim(kind.colour, fade), thick, cv2.LINE_AA)
    cv2.line(canvas, (int(x0 + 0.17 * w), int(shoulder + 0.12 * height)),
             (int(x0 + 0.17 * w), int(ly0 - 0.04 * height)), _dim(SHINE, fade),
             thick, cv2.LINE_AA)
    # Neck and lid.
    neck = (int(x - 0.36 * w), int(foot - 0.87 * height), int(x + 0.36 * w), int(shoulder) + 1)
    cv2.rectangle(canvas, neck[:2], neck[2:], _dim(GLASS, fade), -1)
    cv2.rectangle(canvas, neck[:2], neck[2:], _dim(EDGE, fade), thick, cv2.LINE_AA)
    lid = (int(x - 0.42 * w), int(foot - height), int(x + 0.42 * w), int(foot - 0.87 * height))
    cv2.rectangle(canvas, lid[:2], lid[2:], _dim(LID, fade), -1)
    for k in (0.33, 0.66):
        yy = int(lid[1] + k * (lid[3] - lid[1]))
        cv2.line(canvas, (lid[0], yy), (lid[2], yy), _dim(GLASS, fade), 1, cv2.LINE_AA)


def _label(canvas, words: str, centre: Point, max_w: float, max_h: float, colour,
           thick: int) -> None:
    """`words` as large as fits, without the shadow overlay.text gives text."""
    (tw, th), _ = cv2.getTextSize(words, overlay.FONT, 1.0, thick)
    scale = max(0.2, min(max_w / max(tw, 1), max_h / max(th, 1)))
    (tw, th), _ = cv2.getTextSize(words, overlay.FONT, scale, thick)
    cv2.putText(canvas, words, (int(centre[0] - tw / 2), int(centre[1] + th / 2)),
                overlay.FONT, scale, colour, thick, cv2.LINE_AA)


def draw_shelf(canvas, x0: int, x1: int, y: int, h: int) -> None:
    depth = max(4, int(h * 0.016))
    cv2.rectangle(canvas, (x0, y), (x1, y + depth), SHELF, -1)
    cv2.line(canvas, (x0, y), (x1, y), overlay.GREY, 2, cv2.LINE_AA)
    for x in (x0 + (x1 - x0) // 8, x1 - (x1 - x0) // 8):          # its brackets
        cv2.line(canvas, (x, y + depth), (x + depth * 2, y + depth * 4), SHELF,
                 max(2, depth // 2), cv2.LINE_AA)


# -- a player's half --------------------------------------------------------
class JarLane(Lane):
    """One player's half of the jar range, and that player's score."""

    RISE, DROP = 0.15, 0.6       # seconds a jar takes to come up, and to go
    MARKED = True                # what a jar is worth is written on it
    GRAVITY = 1800.0             # px/s^2, for the pieces of a jar

    def _stations(self) -> List[Tuple[float, float, int]]:
        """Where a jar can stand: its centre line, its foot, the screen height."""
        h = self.screen_size[1]
        x0, x1 = self.span
        return [(x0 + (x1 - x0) * (2 * i + 1) / (2 * COLUMNS), h * foot, h)
                for foot in SHELVES for i in range(COLUMNS)]

    def _arrive(self, p: JarPopup) -> Jar:
        return self._jar(p.look, p.station, 0.0, p.scale, self.started + p.at, p.lifetime)

    def _jar(self, look: str, station: int, shift: float, scale: float, born: float,
             lifetime: float) -> Jar:
        x, foot, h = self.stations[station]
        height = int(KINDS[look].height * h * scale)
        x += shift * (self.span[1] - self.span[0]) / COLUMNS
        return Jar(x=x, y=foot - height / 2, radius=height / 2, born=born,
                   lifetime=lifetime, look=look, height=height, station=station)

    def _expired(self, t: Jar) -> None:
        self.stats.escaped += 1
        self.combo = 0

    def _points(self, t: Jar, now: float) -> int:
        base = KINDS[t.look].points * (1.0 + t.remaining(now))     # quick shots pay double
        return int(round(base * self.multiplier / 5.0)) * 5

    def _break(self, t: Jar, now: float) -> None:
        t.dying, t.shot = now, True
        w = t.width
        for i in range(16):
            glass = i < 10
            a = self.rng.uniform(-math.pi, 0.0)                        # upwards, mostly
            speed = self.rng.uniform(150, 520) * (0.6 + t.height / 300)
            t.shards.append(Shard(
                x=t.x + self.rng.uniform(-0.4, 0.4) * w,
                y=t.y + self.rng.uniform(-0.3, 0.4) * t.height,
                vx=math.cos(a) * speed, vy=math.sin(a) * speed,
                size=self.rng.uniform(0.08, 0.2) * w if glass else self.rng.uniform(0.03, 0.07) * w,
                angle=self.rng.uniform(0, 2 * math.pi), spin=self.rng.uniform(-12, 12),
                glass=glass))

    def _hit(self, target: Jar, now: float, how: str) -> None:
        self._break(target, now)
        points = self._points(target, now)
        self.combo += 1
        self.stats.best_combo = max(self.stats.best_combo, self.combo)
        self.stats.hits += 1
        self.stats.score += points
        self.stats.reactions.append(target.age(now))
        label = f"+{points}"
        if self.combo > 1:
            label += f"  x{self.multiplier:.1f}"
        self.shots.append(Shot(self._at[0], self._at[1], now, points, label, True))
        self.events.append("gold" if target.look == "gold" else "hit")

    # -- drawing ------------------------------------------------------------
    def draw_field(self, canvas) -> None:
        h = self.screen_size[1]
        x0, x1 = self.span
        for foot in SHELVES:
            draw_shelf(canvas, x0 + 12, x1 - 12, int(h * foot), h)
        for t in self.targets:
            self._draw_target(canvas, t, self.now)
        for s in self.shots:
            self._draw_shot(canvas, s, self.now)
        if self._dwelt is not None and self._dwelt.dwell > 0 and self._at is not None:
            cv2.ellipse(canvas, (int(self._at[0]), int(self._at[1])), (68, 68), -90, 0,
                        360 * min(1.0, self._dwelt.dwell / self.DWELL_TIME),
                        overlay.WHITE, 3, cv2.LINE_AA)

    def _draw_target(self, canvas, t: Jar, now: float) -> None:
        if t.shot:
            self._draw_shards(canvas, t, now - t.dying)
            return
        left = t.remaining(now)
        if t.dying is not None:                   # time ran out: it fades away, empty
            fade = max(0.0, 1.0 - (now - t.dying) / self.DROP)
            draw_jar(canvas, t.x, t.foot, t.height, t.look, 0.0, 0.5 * fade, self.MARKED)
            return
        up = min(1.0, t.age(now) / self.RISE)
        x = t.x
        if left < WAKE:                           # nearly out of time: it shakes
            x += 0.04 * t.width * math.sin(now * 55.0) * (1.0 - left / WAKE)
        draw_jar(canvas, x, t.foot, t.height * up, t.look, left, marked=self.MARKED)

    def _draw_shards(self, canvas, t: Jar, age: float) -> None:
        fade = max(0.0, 1.0 - age / self.DROP)
        if fade <= 0:
            return
        fill = KINDS[t.look].colour                # drops: small, so they may be bright
        for s in t.shards:
            x = s.x + s.vx * age
            y = s.y + s.vy * age + 0.5 * self.GRAVITY * age * age
            if s.glass:
                a = s.angle + s.spin * age
                pts = np.array([(x + s.size * math.cos(a + k * 2.2),
                                 y + s.size * math.sin(a + k * 2.2) * (0.6 if k else 1.0))
                                for k in range(3)], np.int32)
                cv2.fillPoly(canvas, [pts], _dim(GLASS, fade), cv2.LINE_AA)
                cv2.polylines(canvas, [pts], True, _dim(EDGE, fade), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (int(x), int(y)), max(1, int(s.size)), _dim(fill, fade),
                           -1, cv2.LINE_AA)



# -- jar range --------------------------------------------------------------
class JarRange(Duel):
    """A minute at the jars for two: the same jars in both halves, and
    whoever has the higher score when time is called has won."""

    HINT = "гравець 1 ліворуч, гравець 2 праворуч — що менша банка, то більше очок"
    LANE = JarLane
    SOUNDS = ASSETS
    TUNE = "music"

    def _plan(self) -> List[Popup]:
        """Every jar of the round: when it comes up, where, which and for how
        long. They come quicker, smaller and for less time as the round goes on."""
        plan: List[Popup] = []
        busy = [0.0] * (len(SHELVES) * COLUMNS)     # when each place is free again
        at = 0.3
        while at < self.duration - 1.0:
            k = at / self.duration
            for _ in range(2 if self.rng.random() < 0.4 * k else 1):
                free = [i for i, until in enumerate(busy) if until <= at]
                if not free:
                    break
                station = self.rng.choice(free)
                look = "gold" if at > 10 and self.rng.random() < 0.07 else \
                    self.rng.choice(SIZES)
                lifetime = KINDS[look].lifetime * (1.0 - 0.35 * k)
                plan.append(JarPopup(at, station, look, lifetime, 1.0 - 0.25 * k))
                busy[station] = at + lifetime + JarLane.DROP + 0.25
            at += (1.05 - 0.5 * k) * self.rng.uniform(0.8, 1.2)
        return plan

    def _tally(self, lane: Lane) -> str:
        return f"банок: {lane.stats.hits}   пропущено: {lane.stats.escaped}"

    def _summary(self, lane: Lane) -> str:
        s = lane.stats
        return (f"{lane.name}    банок: {s.hits}    пропущено: {s.escaped}    "
                f"промахів: {s.misses}    влучність {lane.accuracy:.0%}    "
                f"реакція {s.reaction:.2f} с")


# -- quick draw -------------------------------------------------------------
WAIT, DRAW, SHOWN = "wait", "draw", "shown"


class DrawLane(JarLane):
    """A player's half in the quick draw: one jar at a time, put up by the
    game, and a shot before it is up is a foul."""

    MARKED = False               # a point is a point, whatever the jar

    def start(self, now: float) -> None:
        super().start(now)
        self.mode = WAIT
        self.fouled = False
        self.reaction: Optional[float] = None    # how long this player took, this time
        self.verdict = ""                        # what it came to, for this player

    def put(self, look: str, station: int, shift: float, now: float, lifetime: float) -> None:
        self.targets.append(self._jar(look, station, shift, 1.0, now, lifetime))

    def _expired(self, t: Jar) -> None:
        pass                                     # the game says who was too slow

    def _hit(self, target: Jar, now: float, how: str) -> None:
        self._break(target, now)
        self.reaction = target.age(now)
        self.stats.hits += 1
        self.stats.reactions.append(self.reaction)
        self.events.append("hit")

    def _miss(self, x: float, y: float, now: float) -> None:
        if self.mode == WAIT and not self.fouled:
            self.fouled = True
            self.stats.decoys += 1
            self.shots.append(Shot(x, y, now, 0, "ЗАРАНО", False))
            self.events.append("bad")
        elif self.mode == DRAW:
            super()._miss(x, y, now)


class QuickDraw(Duel):
    """Wait, then break the jar before the other player does. First to
    TARGET points has the match."""

    TARGET = 5
    LIMIT = 3.0              # seconds a jar waits to be shot
    AFTER = 0.8              # after the first shot, the other player has this
                             # long to get a time of their own
    SHOW = 2.2               # seconds the outcome of a draw stays up
    WAIT_FOR = (1.5, 4.0)    # how long the wait before a jar can be
    HINT = "чекай на банку і розбий свою раніше за суперника — вистрілиш зарано, і очко дістанеться йому"
    LANE = DrawLane
    SOUNDS = ASSETS
    TUNE = None              # just the wait
    WINS = "виграно матчів"

    def _plan(self) -> List[Popup]:
        return []                                # the jars are put up as it goes

    def start(self, now: float) -> None:
        super().start(now)
        self.played = 0
        self.point_to: Optional[int] = None      # who took the last point
        self._drawn_at = 0.0
        self._first: Optional[float] = None
        self._wait(self.started)

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        self._until += delta
        self._drawn_at += delta

    def _finished(self) -> bool:
        return self.phase == SHOWN and self.now >= self._until \
            and max(lane.score for lane in self.lanes) >= self.TARGET

    # -- the draw -----------------------------------------------------------
    def _wait(self, now: float) -> None:
        self.phase = WAIT
        self._until = now + self.rng.uniform(*self.WAIT_FOR)
        for lane in self.lanes:
            lane.mode, lane.fouled, lane.reaction, lane.verdict = WAIT, False, None, ""
            lane.targets.clear()

    def _draw(self, now: float) -> None:
        """Up goes a jar in each half, in the same place: smaller as the match goes on."""
        self.phase = DRAW
        n = self.played
        look = self.rng.choice(("big", "mid") if n < 2 else ("mid", "small") if n < 5
                               else ("small",))
        station = self.rng.randrange(len(SHELVES) * COLUMNS)
        shift = self.rng.uniform(-0.25, 0.25)
        for lane in self.lanes:
            lane.mode = DRAW
            lane.put(look, station, shift, now, self.LIMIT)
        self._drawn_at, self._first = now, None
        self._until = now + self.LIMIT

    def _show(self, now: float, winner: Optional[int], early: bool) -> None:
        self.phase = SHOWN
        self.played += 1
        self.point_to = winner
        self._until = now + self.SHOW
        for lane in self.lanes:
            lane.mode = SHOWN
            for t in lane.targets:
                if t.dying is None:
                    t.dying = now
            lane.verdict = "ЗАРАНО" if lane.fouled else \
                f"{lane.reaction:.3f} с" if lane.reaction is not None else \
                "" if early else "запізно"
        if winner is not None:
            self.lanes[winner].stats.score += 1
            self.sound.blip("point")

    def _step(self, now: float, dt: float, point: Optional[Point]) -> None:
        super()._step(now, dt, point)            # the sounds of the shots
        if self.phase == WAIT:
            fouled = [lane.fouled for lane in self.lanes]
            if any(fouled):
                # The other player takes the point - unless both were too quick.
                self._show(now, None if all(fouled) else fouled.index(False), True)
            elif now >= self._until:
                self._draw(now)
        elif self.phase == DRAW:
            times = [lane.reaction for lane in self.lanes]
            done = [t for t in times if t is not None]
            if done and self._first is None:
                self._first = now
                self._until = min(self._until, now + self.AFTER)
            if len(done) == len(times) or now >= self._until:
                best = min(done, default=None)
                winner = times.index(best) if done and times.count(best) == 1 else None
                self._show(now, winner, False)
        elif now >= self._until and max(lane.score for lane in self.lanes) < self.TARGET:
            self._wait(now)

    # -- drawing ------------------------------------------------------------
    def _draw_hud(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        box = (w // 2 - int(110 * k), int(16 * k), w // 2 + int(110 * k), int(86 * k))
        cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 0), -1)
        cv2.rectangle(canvas, box[:2], box[2:], overlay.GREY, 1)
        overlay.text_fit(canvas, f"до {self.TARGET} очок", (w // 2, (box[1] + box[3]) // 2),
                         190 * k, 30 * k, overlay.WHITE, 2)

        status_y = int(h * 0.19)
        for i, lane in enumerate(self.lanes):
            (tw, _), _ = cv2.getTextSize(lane.name, overlay.FONT, 0.75 * k, 2)
            edge = lane.span[0] + int(40 * k) if i == 0 else lane.span[1] - int(40 * k)
            overlay.text(canvas, lane.name, (edge if i == 0 else edge - tw, int(40 * k)),
                         0.75 * k, lane.colour, 2)
            # A pip for each point, filled as they are won.
            r, step = int(15 * k), int(42 * k)
            for n in range(self.TARGET):
                cx = edge + r + n * step if i == 0 else edge - r - n * step
                won = n < lane.score
                cv2.circle(canvas, (cx, int(84 * k)), r, lane.colour if won else overlay.DIM,
                           -1 if won else 2, cv2.LINE_AA)

            if self.counting_down:
                continue
            centre = ((lane.span[0] + lane.span[1]) // 2, status_y)
            if self.phase == WAIT:
                words, colour = "чекай...", overlay.GREY
            elif self.phase == DRAW:
                if lane.reaction is None:
                    words, colour = f"{self.now - self._drawn_at:.2f}", overlay.WHITE
                else:
                    words, colour = f"{lane.reaction:.3f} с", overlay.GREEN
            else:
                words = lane.verdict
                colour = overlay.GREEN if self.point_to == i else \
                    overlay.YELLOW if lane.fouled else overlay.GREY
                if self.point_to == i:
                    words = f"ОЧКО   {words}" if words else "ОЧКО"
            if words:
                overlay.text_fit(canvas, words, centre, (lane.span[1] - lane.span[0]) * 0.7,
                                 52 * k, colour, 3)
        if self.phase == SHOWN and self.point_to is None \
                and all(lane.reaction is not None for lane in self.lanes):
            overlay.text_fit(canvas, "НІЧИЯ", (w // 2, int(h * 0.30)), w * 0.3, 60 * k,
                             overlay.WHITE, 3)

    def _summary(self, lane: Lane) -> str:
        s = lane.stats
        times = (f"найкраща {min(s.reactions):.3f} с    середня {s.reaction:.3f} с"
                 if s.reactions else "жодної розбитої банки")
        return (f"{lane.name}    очок: {s.score}    {times}    "
                f"зарано: {s.decoys}    промахів: {s.misses}")
