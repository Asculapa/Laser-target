"""Wild West: a shoot-out in the manner of the arcade light-gun games, for
one player or two together.

Bandits come up in the windows, doors and street of a Western town, gun in
hand. Each one fires after a moment - the ring closing on him is how long
there is - and a bandit who fires costs a life. Townsfolk come up too, empty-
handed: shooting one costs a life as well. When a street's bandits are all
down, their leader walks out for a showdown: he waits, someone shouts DRAW,
and there is less than a second to get him. Six streets, six showdowns, and
the town is safe. Each street brings something new: bandits who duck behind
cover and peek out again, bandits running across the street, and - for two -
big ones that only a shot from each player at once will bring down.

A six-shooter holds six: then it is empty until it is reloaded, by shooting
the RELOAD strip along the bottom of the screen. A star floating over the
town is a life back; a gold coin doubles the points for a while.

Two players share the lives and the town but keep scores of their own.
Nothing tells one laser from the other but where it is, so a laser is the
player's on whose side of the screen it comes on, and stays theirs while the
beam stays lit: to help the other player, keep the beam on and slide it
across - a bandit brought down on the partner's side is a save. Hits taken
in turn by the two make a crossfire, which multiplies the points. The
showdowns are two bosses, one each; between the streets the two meet at
High Noon - first to break the bottle on their side - and at the end each is
given an award or two.

Each laser is a gun of its own (Gun below), followed from frame to frame by
where it is, so that each has its own trigger. The town and its people are
CC0 pictures from OpenGameArt.org, recoloured for this screen (west_art.py).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import jars, overlay, sound
from . import west_art as art
from .game import OVER, PAUSED, PLAYING, HighScores, Round, Shot, Stats, Target
from .story import Hold

SOUNDS = art.ASSETS / "sound"
SETUP, STREET, SHOWDOWN, CLEAR, NOON = "setup", "street", "showdown", "clear", "noon"
POP, PEEK, RUN, TEAM = "pop", "peek", "run", "team"     # kinds of bandit

Point = Tuple[float, float]


@dataclass(frozen=True)
class Stage:
    bandits: int
    fuse: Tuple[float, float]       # seconds a bandit waits before he fires, first to last
    crowd: int                      # how many can be up at once
    draw: float                     # seconds to beat the leader at the showdown
    kinds: Tuple[str, ...]          # what sort of bandits this street has


STAGES = (
    Stage(10, (2.6, 2.1), 2, 1.0, (POP,)),
    Stage(12, (2.4, 1.9), 2, 0.95, (POP, PEEK)),
    Stage(14, (2.3, 1.8), 2, 0.9, (POP, PEEK, RUN)),
    Stage(15, (2.2, 1.7), 3, 0.85, (POP, PEEK, RUN, TEAM)),
    Stage(16, (2.1, 1.6), 3, 0.8, (POP, PEEK, RUN, TEAM)),
    Stage(18, (2.0, 1.5), 3, 0.75, (POP, PEEK, RUN, TEAM)),
)
NOON_AFTER = (1, 3)                 # the streets after which two meet at High Noon


@dataclass(frozen=True)
class Level:
    title: str
    lives: int
    fuse: float                     # times the stage's fuses
    crowd: int                      # added to the stage's crowd
    draw: float                     # times the showdown's time
    ammo: int                       # the six-shooter; 0: never needs reloading
    life_back: bool                 # a star back for each street cleared


LEVELS = {
    "deputy": Level("ПОМІЧНИК ШЕРИФА", 7, 1.35, -1, 1.3, 0, True),
    "sheriff": Level("ШЕРИФ", 5, 1.1, 0, 1.0, 6, True),
    "marshal": Level("МАРШАЛ", 4, 0.8, 1, 0.85, 6, False),
}
EASIER = 1.35                       # a player who chose "easier": longer fuses on their side...
EASY_REACH = 0.12                   # ...and this much of a figure's height more to hit
STRIP = 0.94                        # where the RELOAD bar starts, down the screen


@dataclass(eq=False)
class Figure(Target):
    """Someone in the town. (x, y) is the middle of them; `foot` is where they stand."""
    look: str = ""
    spot: int = -1
    foot: float = 0.0
    height: int = 0
    clip: Optional[float] = None
    flip: bool = False
    end: str = ""               # how they went: "hit", "fired" or "left"
    pose: str = "stand"
    kind: str = POP
    side: int = 0               # whose side of the screen they are on
    easy: bool = False
    hidden: bool = False        # a peeker, ducked down: nothing to shoot
    first: Optional[Tuple[int, float]] = None   # a team target: who hit it first, and when

    @property
    def armed(self) -> bool:
        return art.LOOKS[self.look].armed

    def sprite(self, now: float) -> art.Sprite:
        n = int(now * 4) if self.dying is None else 0       # four frames a second
        n %= len(art.frames(self.look, self.pose))
        return art.sprite(self.look, self.pose, n, self.height, self.flip)

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        if self.hidden:
            return False
        if self.clip is not None and y > self.clip:
            return False
        s = art.sprite(self.look, "stand" if self.pose == "walk" else self.pose, 0,
                       self.height, self.flip)
        col = int(x - self.x + s.fx)
        row = int(y - self.foot + s.fy)
        if self.easy:
            # A player who chose "easier" may land a little wide of the figure.
            x0, y0, x1, y1 = s.box
            m = EASY_REACH * self.height
            return x0 - m <= col <= x1 + m and y0 - m <= row <= y1 + m
        return 0 <= row < s.near.shape[0] and 0 <= col < s.near.shape[1] \
            and bool(s.near[row, col])


@dataclass(eq=False)
class Pickup(Target):
    """A star (a life back) or a gold coin (double points), floating over the town."""

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return math.hypot(x - self.x, y - self.y) <= self.radius * 1.4 + slack


@dataclass(eq=False)
class Strip(Target):
    """The RELOAD strip along the bottom of the screen."""

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return y >= self.y


@dataclass(eq=False)
class Bottle(Target):
    """A bottle on a post at High Noon: (x, y) is its middle."""
    side: int = 0
    height: int = 0

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return abs(x - self.x) <= 0.5 * jars.WIDTH * self.height + 12 + slack \
            and abs(y - self.y) <= 0.5 * self.height + 12 + slack


@dataclass
class Player:
    name: str
    colour: Tuple[int, int, int]
    stats: Stats = field(default_factory=Stats)
    saves: int = 0
    draws: List[float] = field(default_factory=list)    # showdowns won, how quickly
    noon: int = 0                                         # High Noon matches won
    easy: bool = False
    ammo: int = 0

    @property
    def accuracy(self) -> float:
        shots = self.stats.shots + self.stats.decoys
        return self.stats.hits / shots if shots else 0.0


class Gun(Round):
    """One laser's trigger. The beam coming on is a shot; a beam held still
    on someone is one too. It aims at the game's targets and leaves the
    consequences to the game. Its owner is the player on whose side the beam
    came on, for as long as it stays lit."""

    def __init__(self, game: "West") -> None:
        super().__init__(game.screen_size)
        self.game = game
        self.targets = game.targets          # the same list, not a copy
        self.last: Optional[Point] = None    # where its dot was last seen
        self.owner = 0
        self.dwell = 0.0                     # its own hold: the figures are shared,
                                             # so the hold cannot be kept on them

    def aim(self, now: float, dt: float, point: Optional[Point]) -> None:
        self.now = now
        if point is not None:
            self.last = point
        self._aim(now, dt, point)

    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        # Anything in the town before the strip under it.
        live = [t for t in self.targets if t.dying is None and t.contains(x, y, slack)]
        if not live:
            return None
        live.sort(key=lambda t: (isinstance(t, Strip), math.hypot(t.x - x, t.y - y)))
        return live[0]

    def _aim(self, now: float, dt: float, point: Optional[Point]) -> None:
        lit = self._beam(now, point)
        if point is None:
            if not self._beam_on:
                self._dwelt, self.dwell = None, 0.0
            return                    # a dropped frame: the hold carries on
        if lit:
            self.owner = self.game.side_of(point)
        if self.FLASH and lit:
            target = self._target_at(point[0], point[1], slack=6.0)
            if target is not None:
                self._hit(target, now, "flash")
            else:
                self._miss(point[0], point[1], now)
            self._dwelt, self.dwell = None, 0.0
            return
        under = self._target_at(point[0], point[1])
        if under is not self._dwelt:
            self._dwelt, self.dwell = under, 0.0
        if under is None:
            return
        self.dwell += dt
        # A runner will not stand still for a beam held on him: it takes less.
        need = self.DWELL_TIME * (0.7 if isinstance(under, Figure) and under.kind == RUN else 1.0)
        if self.dwell >= need:
            self._hit(under, now, "dwell")
            self._dwelt, self.dwell = None, 0.0

    def _hit(self, target: Target, now: float, how: str) -> None:
        self.game._shot(self, target, now)

    def _miss(self, x: float, y: float, now: float) -> None:
        self.game._missed(self, x, y, now)


class West(Round):
    RISE, FIRE, FALL = 0.18, 0.45, 0.6   # seconds to come up, to fire, to fall
    STAY = 2.3               # seconds the townsfolk stay
    CYCLE, SHOW = 1.7, 1.0   # a peeker is up SHOW seconds of every CYCLE
    TEAM_WINDOW = 0.6        # seconds between the two shots of a team shot
    CROSSFIRE = 1.5          # seconds within which hits in turn make a crossfire
    GOLD = 10.0              # seconds a gold coin doubles the points
    BANNER = 2.6             # seconds a stage's title or its end stays up
    ABSENT = 25.0            # seconds one side may go without a dot before the game waits
    HINT = "стріляй у тих, хто вихопив зброю, поки вони не вистрілили — мирних не чіпай"
    players = 2              # every dot on the screen is a gun (also with one player)

    def __init__(self, screen_size: Tuple[int, int], players: int = 1,
                 level: str = "sheriff", seed: Optional[int] = None) -> None:
        super().__init__(screen_size, seed)
        self.sound = sound.Player(SOUNDS)
        self.two = players == 2
        self.level = LEVELS[level]
        self.scores_name = f"highscores-west-{players}p-{level}"
        if self.level.ammo:
            self.HINT = ("стріляй у тих, хто вихопив зброю, а мирних не чіпай; "
                         "щоб перезарядити, стріляй у смугу внизу")
        self.guns = [Gun(self), Gun(self)]
        self.team: List[Player] = []
        self.stage = 0
        self.lives = self.level.lives
        self.combo = 0
        self.crossfire = 0
        self.won = False
        self.phase = SETUP if self.two else STREET
        self.bosses: List[Figure] = []
        self._dots: List[Point] = []
        self._holds = [Hold(), Hold()]       # the setup buttons, a set for each side
        self._ready = [False, False]
        self._absent: Optional[int] = None
        self._reset_clocks(0.0)

    # -- round flow ---------------------------------------------------------
    def _reset_clocks(self, now: float) -> None:
        self._next = now
        self._shake = -1.0
        self._banner: Tuple[str, float] = ("", -1.0)
        self._message: Tuple[str, float] = ("", -1.0)
        self._hints = [-1.0, -1.0]           # until when "RELOAD!" shows, each side
        self._reloaded = [-1.0, -1.0]        # until when "RELOADED!" shows, each side
        self._taught = [False, False]        # told how to reload, the first time it ran dry
        self._fired = [-1.0, -1.0]           # when each side's gun last went off
        self._holes: List[Tuple[float, float, float]] = []   # misses in the walls: x, y, when
        self._sub: Tuple[str, float] = ("", -1.0)   # a line under the banner or the message
        self._gold = -1.0
        self._drops: List[Tuple[float, str]] = []
        self._drawn = False
        self._drawn_at = 0.0
        self._last_hit: Optional[Tuple[int, float]] = None
        self._seen = [now, now]

    def start(self, now: float) -> None:
        names = (("ГРАВЕЦЬ 1", overlay.CYAN), ("ГРАВЕЦЬ 2", overlay.YELLOW))
        old = self.team
        self.team = [Player(n, c) for n, c in names[:2 if self.two else 1]]
        for p, before in zip(self.team, old):
            p.easy = before.easy                # a rematch keeps what was chosen
        self.sound.stop()
        if self.two and not any(self._ready):
            # Two players choose first; the countdown waits for them.
            super().start(now)
            self.phase = SETUP
            self.now = now
            self._holds = [Hold(), Hold()]
            return
        self._begin(now)

    def _begin(self, now: float) -> None:
        super().start(now)
        self.lives = self.level.lives
        self.combo = self.crossfire = 0
        self.won = False
        for p in self.team:
            p.ammo = self.level.ammo
        self._reset_clocks(self.started)
        self._noon_for = -1                      # the street after which High Noon was held
        self._enter(0, self.started)

    def close(self) -> None:
        self.sound.stop()

    def _enter(self, stage: int, now: float) -> None:
        """Out into the next street."""
        self.stage = stage
        self.phase = STREET
        self.targets.clear()
        self.bosses = []
        if self.level.ammo:
            w, h = self.screen_size
            self.targets.append(Strip(x=w / 2, y=h * STRIP, radius=w / 2, born=now,
                                      lifetime=math.inf))
        self._left = STAGES[stage].bandits       # still to come
        self._next = now + 1.2
        self._banner = (self.scene.title, now + 1.6)
        self._message = ("", -1.0)
        self._holes.clear()                      # a new street, with its walls whole
        if stage == 0 and self.level.ammo:
            self._sub = (f"набоїв у барабані: {self.level.ammo} — потім стріляй у смугу "
                         "ПЕРЕЗАРЯДКИ внизу", now + 5.0)
        # Something to shoot for, once or twice a street.
        self._drops = [(now + self.rng.uniform(6.0, 16.0), "coin")]
        if stage > 0:
            self._drops.append((now + self.rng.uniform(10.0, 22.0), "star"))

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        for t in self.targets + [b for b in self.bosses if b not in self.targets]:
            if t.dying is not None:
                t.dying += delta
            if isinstance(t, Figure) and t.first is not None:
                t.first = (t.first[0], t.first[1] + delta)
        for b in self.bosses:
            if b not in self.targets:
                b.born += delta
        self._next += delta
        self._shake += delta
        self._drawn_at += delta
        self._gold += delta
        self._hints = [h + delta for h in self._hints]
        self._reloaded = [h + delta for h in self._reloaded]
        self._sub = (self._sub[0], self._sub[1] + delta)
        self._drops = [(t + delta, k) for t, k in self._drops]
        self._holes = [(x, y, t + delta) for x, y, t in self._holes]
        self._banner = (self._banner[0], self._banner[1] + delta)
        self._message = (self._message[0], self._message[1] + delta)
        self._seen = [s + delta for s in self._seen]

    def toggle_pause(self, now: float) -> None:
        was = self.state
        super().toggle_pause(now)
        if self.state != was:
            if self.state == PAUSED:
                self.sound.hold()
            else:
                self.sound.release()
                self._absent = None
                self.PAUSE_TEXT = Round.PAUSE_TEXT

    def key(self, key: int) -> bool:
        if self.phase == SETUP and key in (32, 13, 10):
            self._ready = [True, True]
            self._begin(self.now)
            return True
        if self._absent is not None and key in (ord("c"), ord("C")):
            self._seen = [self.now, self.now]
            self.toggle_pause(self.now)          # carry on: the other may come back
            return True
        return False

    def _finished(self) -> bool:
        return self.lives <= 0 or self.won

    @property
    def scene(self) -> art.Scene:
        return art.SCENES[self.stage]

    @property
    def multiplier(self) -> float:
        return 1.0 + 0.1 * min(self.combo, 10) + 0.1 * min(self.crossfire, 10)

    def side_of(self, point: Point) -> int:
        return int(self.two and point[0] >= self.screen_size[0] / 2)

    # -- the guns -----------------------------------------------------------
    def update(self, now: float, point: Optional[Point],
               points: Optional[Sequence[Point]] = None) -> None:
        self._dots = list(points)[:2] if points is not None else \
            [point] if point is not None else []
        for d in self._dots:
            self._seen[self.side_of(d)] = now
        if self.phase == SETUP:
            self._setup(now)
            return
        if self.state == PAUSED and self._absent is not None \
                and any(self.side_of(d) == self._absent for d in self._dots):
            self.toggle_pause(now)               # they are back
        if self.state == PLAYING and now < self.started:
            # Follow the beams through the countdown, so that one already on
            # when it ends is not taken for a shot.
            for gun, dot in zip(self.guns, self._deal(self._dots)):
                gun.last = dot or gun.last
                gun._beam(now, dot)
        super().update(now, point)
        if self.state != PLAYING:
            return
        if self.phase == STREET:
            self.sound.music(f"music{self.scene.music}")
            if self.two and not self.counting_down:
                for side in (0, 1):
                    if now - self._seen[side] > self.ABSENT:
                        self._wait_for(side, now)
                        break

    def _wait_for(self, side: int, now: float) -> None:
        """A side has had no laser for a long time: wait for its player."""
        self.toggle_pause(now)
        self._absent = side
        self.PAUSE_TEXT = (f"ЧЕКАЄМО: {self.team[side].name}",
                           "наведи лазер на екран    C — грати без нього")

    def _deal(self, dots: Sequence[Point]) -> List[Optional[Point]]:
        """Which dot is whose: each gun gets the one nearest where its own
        was, and a dot far from both is a gun that has just come on."""
        def cost(gun: Gun, dot: Point) -> float:
            if gun.last is None or not gun._beam_on:
                return 400.0
            return math.hypot(dot[0] - gun.last[0], dot[1] - gun.last[1])
        best, given = math.inf, [None] * len(self.guns)
        for order in itertools.permutations(range(len(self.guns)), len(dots)):
            c = sum(cost(self.guns[g], d) for g, d in zip(order, dots))
            if c < best:
                best, given = c, [None] * len(self.guns)
                for g, d in zip(order, dots):
                    given[g] = d
        return given

    # -- the setup, for two -------------------------------------------------
    def setup_boxes(self, side: int) -> List[Tuple[int, int, int, int]]:
        """The NORMAL and EASIER buttons on a side."""
        w, h = self.screen_size
        x0 = side * w // 2 + int(w * 0.07)
        bw, bh = int(w * 0.16), int(h * 0.16)
        y0 = int(h * 0.50)
        return [(x0, y0, x0 + bw, y0 + bh), (x0 + bw + int(w * 0.04), y0,
                                             x0 + 2 * bw + int(w * 0.04), y0 + bh)]

    def _setup(self, now: float) -> None:
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        for side in (0, 1):
            dot = next((d for d in self._dots if self.side_of(d) == side), None)
            chose = self._holds[side].update(dt, dot, self.setup_boxes(side))
            if chose is not None:
                self.team[side].easy = chose == 1
                self._ready[side] = True
                self.sound.blip("reload")
        if all(self._ready):
            self._begin(now)

    # -- shots --------------------------------------------------------------
    def _fire(self, p: int, now: float) -> bool:
        """A bullet out of player `p`'s gun - if there is one left."""
        if not self.level.ammo or self.phase == NOON:
            self._fired[p] = now
            return True
        player = self.team[p]
        if player.ammo <= 0:
            self._hints[p] = now + 1.0
            self.sound.blip("click")
            if not self._taught[p]:
                # The first time, stop and say it in the middle of the screen.
                self._taught[p] = True
                who = f"{player.name}: " if self.two else ""
                self._message = (f"{who}НАБОЇ СКІНЧИЛИСЯ!", now + 2.5)
                self._sub = ("стріляй у смугу ПЕРЕЗАРЯДКИ внизу екрана", now + 2.5)
            # Said where the shot went, too: a click alone looks like a shot
            # that did not work.
            if self.guns[0].last is not None or self.guns[1].last is not None:
                gun = next((g for g in self.guns if g.owner == p and g.last is not None), None)
                if gun is not None:
                    self.shots.append(Shot(gun.last[0], gun.last[1], now, 0,
                                           "ПОРОЖНЬО — ПЕРЕЗАРЯДИ", False))
            return False
        player.ammo -= 1
        self._fired[p] = now
        return True

    def _shot(self, gun: Gun, t: Target, now: float) -> None:
        p = gun.owner
        if isinstance(t, Strip):
            if self.team[p].ammo < self.level.ammo:
                self.team[p].ammo = self.level.ammo
                self._reloaded[p] = now + 0.9
                self._hints[p] = -1.0
                self.sound.blip("reload")
            return
        if not self._fire(p, now):
            return
        if isinstance(t, Pickup):
            self._pickup(t, p, now)
        elif isinstance(t, Bottle):
            self._noon_hit(t, now)
        elif t in self.bosses:
            self._boss_down(t, p, now)
        elif not t.armed:
            t.dying, t.end = now, "hit"
            self.team[p].stats.decoys += 1
            self.stats.decoys += 1
            self._hurt(now, "НЕ В МИРНИХ!", "oops")
            self.shots.append(Shot(t.x, t.y, now, 0, "мирний!", False))
        elif t.kind == TEAM:
            self._team_hit(t, p, now)
        else:
            t.dying, t.end = now, "hit"
            saved = self.two and p != t.side
            points = self._credit(p, self._points(t, now), now, t.age(now))
            label = f"+{points}"
            if saved:
                self.team[p].saves += 1
                bonus = self._credit(p, 50, now, None, counts=False)
                label = f"ВРЯТОВАНО!  +{points + bonus}"
            elif self.multiplier > 1.05:
                label += f"  x{self.multiplier:.1f}"
            self.shots.append(Shot(t.x, t.y - t.height / 2, now, points, label, True))
            self.sound.blip("hit")

    def _credit(self, p: int, points: int, now: float, reaction: Optional[float],
                counts: bool = True) -> int:
        """Points to player `p` (and to the team), with the streak and the crossfire."""
        if counts:
            if self.two and self._last_hit is not None and now - self._last_hit[1] < self.CROSSFIRE:
                self.crossfire = self.crossfire + 1 if self._last_hit[0] != p else 0
            self._last_hit = (p, now)
            self.combo += 1
            for s in (self.stats, self.team[p].stats):
                s.hits += 1
                s.best_combo = max(s.best_combo, self.combo)
                if reaction is not None:
                    s.reactions.append(reaction)
        if now < self._gold:
            points *= 2
        for s in (self.stats, self.team[p].stats):
            s.score += points
        return points

    def _points(self, t: Figure, now: float) -> int:
        """More for a quick shot, and for someone small: far off, or at a window."""
        far = math.sqrt(0.3 * self.screen_size[1] / max(20.0, t.height))
        base = 100 * (1.0 + t.remaining(now)) * min(2.0, max(0.8, far))
        if t.kind in (PEEK, RUN):
            base *= 1.25
        return int(round(base * self.multiplier / 5.0)) * 5

    def _team_hit(self, t: Figure, p: int, now: float) -> None:
        """A big one: down only to a shot from each player within TEAM_WINDOW."""
        if t.first is not None and t.first[0] != p and now - t.first[1] <= self.TEAM_WINDOW:
            t.dying, t.end = now, "hit"
            points = self._points(t, now)
            for q in (1 - p, p):
                self._credit(q, points, now, t.age(now))
            self.shots.append(Shot(t.x, t.y - t.height / 2, now, 2 * points,
                                   f"РАЗОМ!  +{points} кожному", True))
            self._message = ("СПІЛЬНИЙ ПОСТРІЛ!", now + 1.0)
            self.sound.blip("hit")
        else:
            t.first = (p, now)
            self.shots.append(Shot(t.x, t.y - t.height / 2, now, 0, "РАЗОМ!", False))
            self.sound.blip("clang")

    def _pickup(self, t: Pickup, p: int, now: float) -> None:
        t.dying = now
        if t.kind == "star":
            self.lives = min(self.level.lives, self.lives + 1)
            self._message = ("+1 ЖИТТЯ", now + 1.0)
        else:
            self._gold = now + self.GOLD
            self._message = ("ЗОЛОТО!  ПОДВІЙНІ ОЧКИ", now + 1.2)
        self.sound.blip(t.kind)

    def _missed(self, gun: Gun, x: float, y: float, now: float) -> None:
        p = gun.owner
        if not self._fire(p, now):
            return
        if self.phase == SHOWDOWN and not self._drawn and self.bosses \
                and all(b.dying is None for b in self.bosses):
            self._hurt(now, "ЗАРАНО!", "hurt", p)
            self._restart_showdown(now)
            return
        if self.phase == NOON and not self._drawn and self._noon_result < now:
            self._noon_point(1 - p, now, f"{self.team[p].name}: ЗАРАНО!")
            self.sound.blip("miss")
            return
        if self.phase == NOON:
            self._hole(x, y, now)
            self.sound.blip("miss")         # no penalty at noon, but a shot is a shot
            return
        self.combo = 0
        self.crossfire = 0
        for s in (self.stats, self.team[p].stats):
            s.misses += 1
        self.shots.append(Shot(x, y, now, 0, "мимо", False))
        self._hole(x, y, now)
        self.sound.blip("miss")

    HOLE_TIME = 6.0          # seconds a bullet hole stays in the wall
    HOLES = 40               # and the most there are at once

    def _hole(self, x: float, y: float, now: float) -> None:
        self._holes.append((x, y, now))
        del self._holes[:-self.HOLES]

    def _hurt(self, now: float, words: str, noise: str, side: Optional[int] = None) -> None:
        self.lives -= 1
        self.combo = 0
        self.crossfire = 0
        self._shake = now
        if self.two and side is not None:
            words = f"Г{side + 1}: {words}"
        self._message = (words, now + 1.2)
        self.sound.blip(noise)

    # -- the street ---------------------------------------------------------
    def _step(self, now: float, dt: float, point: Optional[Point]) -> None:
        self._move(now, dt)
        for gun, dot in zip(self.guns, self._deal(self._dots)):
            gun.aim(now, dt, dot)
        if self.phase == STREET:
            self._street(now)
        elif self.phase == SHOWDOWN:
            self._showdown(now)
        elif self.phase == NOON:
            self._noon(now)
        elif now >= self._banner[1]:             # CLEAR, and its banner is done
            self._after_street(now)
        self._expire(now)

    def _after_street(self, now: float) -> None:
        if self.stage + 1 >= len(STAGES):
            self.won = True
            return
        if self._noon_for != self.stage:
            if self.level.life_back:
                self.lives = min(self.level.lives, self.lives + 1)
            if self.two and self.stage in NOON_AFTER:
                self._noon_for = self.stage
                self._start_noon(now)
                return
        self._enter(self.stage + 1, now)

    def figures(self) -> List[Figure]:
        return [t for t in self.targets if isinstance(t, Figure)]

    def _move(self, now: float, dt: float) -> None:
        w = self.screen_size[0]
        for t in self.targets:
            if isinstance(t, Figure):
                if t.kind == RUN and t.dying is None:
                    t.x += t.vx * dt
                    t.side = self.side_of((t.x, t.y))
                if t.kind == PEEK and t.dying is None:
                    # Whatever of him shows can be shot - coming up and going down too.
                    t.hidden = (t.age(now) % self.CYCLE) >= self.SHOW
            elif isinstance(t, Pickup) and t.dying is None:
                t.x += t.vx * dt
                t.y = self.screen_size[1] * 0.2 + 25 * math.sin(now * 2.0)
        for t in list(self.targets):
            if isinstance(t, (Figure, Pickup)) and t.dying is None and not -0.1 * w < t.x < 1.1 * w:
                self.targets.remove(t)           # gone off the edge

    def _expire(self, now: float) -> None:
        for t in list(self.targets):
            if isinstance(t, Pickup):
                if t.dying is not None and now - t.dying > 0.4:
                    self.targets.remove(t)
                continue
            if not isinstance(t, Figure) or t in self.bosses:
                continue
            if t.dying is None:
                if t.age(now) >= t.lifetime:
                    if t.kind == PEEK and t.hidden:
                        # Ducked: he fires when he next comes up.
                        t.lifetime = t.age(now) + (self.CYCLE - t.age(now) % self.CYCLE) + 0.25
                        continue
                    t.dying = now
                    if t.armed:
                        t.end, t.pose = "fired", "shoot"
                        for s in (self.stats, self.team[t.side].stats):
                            s.escaped += 1
                        self._hurt(now, "ТЕБЕ ПОРАНЕНО!", "enemy", t.side)
                    else:
                        t.end = "left"
            else:
                gone = {"hit": self.FALL, "fired": self.FIRE + self.RISE,
                        "left": self.RISE}[t.end]
                if now - t.dying > gone:
                    self.targets.remove(t)

    def _street(self, now: float) -> None:
        st = STAGES[self.stage]
        crowd = max(1, st.crowd + self.level.crowd)
        live = [t for t in self.figures() if t.dying is None]
        if self._left > 0 and now >= self._next and len(live) < crowd:
            if self._spawn(now):
                self._next = now + self.rng.uniform(0.5, 1.3) * (1.0 - 0.06 * self.stage)
        while self._drops and self._drops[0][0] <= now:
            _, kind = self._drops.pop(0)
            if kind == "coin" or self.lives < self.level.lives:
                self._drop(kind, now)
        if self._left == 0 and not self.figures():
            self._start_showdown(now)

    def _drop(self, kind: str, now: float) -> None:
        w, h = self.screen_size
        right = self.rng.random() < 0.5
        vx = (w / 6.0) * (-1 if right else 1)
        self.targets.append(Pickup(x=1.05 * w if right else -0.05 * w, y=h * 0.2,
                                   radius=h * 0.035, born=now, lifetime=math.inf,
                                   kind=kind, vx=vx))

    def _spawn(self, now: float) -> bool:
        st = STAGES[self.stage]
        figures = self.figures()
        taken = {t.spot for t in figures}
        free = [i for i in range(len(self.scene.spots)) if i not in taken]
        done = st.bandits - self._left
        roll = self.rng.random()
        kind = POP
        if RUN in st.kinds and roll < 0.18 and not any(t.kind == RUN for t in figures):
            kind = RUN
        elif TEAM in st.kinds and self.two and roll < 0.30 \
                and not any(t.kind == TEAM for t in figures):
            kind = TEAM                          # on the middle line: half each
        elif PEEK in st.kinds and roll < 0.55:
            covered = [i for i in free if self.scene.spots[i].clip is not None]
            if covered:
                kind, free = PEEK, covered
        if kind != RUN and not free:
            return False
        townsfolk = kind == POP and done >= 2 and self.rng.random() < 0.25
        look = self.rng.choice(art.TOWNSFOLK if townsfolk else art.BANDITS)
        if kind == RUN:
            spot = -2
            _, foot, height, clip = art.place(self.scene, self.scene.runway, self.screen_size)
            left = self.rng.random() < 0.5
            x = -0.04 * self.screen_size[0] if left else 1.04 * self.screen_size[0]
        elif kind == TEAM:
            # He stands across the middle of the screen, so that each player
            # can shoot their own half of him.
            spot = -3
            _, foot, height, clip = art.place(self.scene, self.scene.showdown, self.screen_size)
            x = self.screen_size[0] / 2
            look = self.rng.choice(art.BANDITS)
        else:
            spot = self.rng.choice(free)
            x, foot, height, clip = art.place(self.scene, self.scene.spots[spot],
                                              self.screen_size)
        flip = x > self.screen_size[0] / 2 if kind != TEAM else self.rng.random() < 0.5
        if clip is not None and not townsfolk:
            # The gun is how a bandit is told from the townsfolk: behind a sill
            # or a railing he stands tall enough for it to show over the top.
            s = art.sprite(look, "stand", 0, height, flip)
            foot = min(foot, clip + (s.fy - s.muzzle[1]) - 0.12 * height)
        side = self.side_of((min(max(x, 0), self.screen_size[0] - 1), foot))
        easy = self.two and self.team[side].easy and kind != TEAM
        k = done / max(1, st.bandits - 1)
        fuse = (st.fuse[0] + (st.fuse[1] - st.fuse[0]) * k) * self.level.fuse
        life = self.STAY if townsfolk else \
            fuse * {POP: 1.0, PEEK: 1.6, RUN: 1.0, TEAM: 1.8}[kind] * (EASIER if easy else 1.0)
        t = Figure(x=x, y=foot - height / 2, radius=height / 2, born=now, lifetime=life,
                   look=look, spot=spot, foot=foot, height=height, clip=clip, flip=flip,
                   kind=kind, side=side, easy=easy)
        if kind == RUN:
            # Across the street, slowly enough to fire before he is off it.
            t.vx = 1.1 * self.screen_size[0] / (life * 1.7) * (1 if left else -1)
            t.flip = not left
            t.pose = "walk"
        self.targets.append(t)
        if not townsfolk:
            self._left -= 1
        return True

    # -- the showdown -------------------------------------------------------
    def _start_showdown(self, now: float) -> None:
        self.phase = SHOWDOWN
        x, foot, height, _ = art.place(self.scene, self.scene.showdown, self.screen_size)
        w = self.screen_size[0]
        places = [(0.3 * w, 0), (0.7 * w, 1)] if self.two else [(x, 0)]
        draw = STAGES[self.stage].draw * self.level.draw
        self.bosses = [Figure(x=bx, y=foot - height / 2, radius=height / 2, born=now,
                              lifetime=draw * (EASIER if self.two and self.team[side].easy
                                               else 1.0),
                              look=art.BOSS, foot=foot, height=height, flip=bx > w / 2,
                              pose="idle", side=side,
                              easy=self.two and self.team[side].easy)
                       for bx, side in places]
        self._drawn = False
        self._next = now + self.rng.uniform(2.0, 4.0)    # when they draw
        self._banner = ("ДУЕЛЬ", now + 1.5)
        self.sound.stop()
        self.sound.blip("bell")

    def _restart_showdown(self, now: float) -> None:
        for b in self.bosses:
            if b in self.targets:
                self.targets.remove(b)
            b.pose, b.dying, b.end = "idle", None, ""
        self._drawn = False
        self._next = now + 1.5 + self.rng.uniform(1.5, 3.5)

    def _showdown(self, now: float) -> None:
        if not self.bosses:
            return
        if all(b.end == "hit" for b in self.bosses):
            if now - max(b.dying for b in self.bosses) >= 1.2:
                self.phase = CLEAR
                self._banner = ("ЕТАП ПРОЙДЕНО" if self.stage + 1 < len(STAGES)
                                else "МІСТО В БЕЗПЕЦІ", now + self.BANNER)
                self.sound.blip("clear")
            return
        fired = [b for b in self.bosses if b.end == "fired"]
        if fired:
            if now - fired[0].dying >= 1.0 and self.lives > 0:
                self._restart_showdown(now)
            return
        if not self._drawn:
            if now >= self._next:
                # DRAW!
                self._drawn = True
                for b in self.bosses:
                    b.pose, b.born = "stand", now
                    self.targets.append(b)
                self._message = ("СТРІЛЯЙ!", now + 0.8)
                self.sound.blip("draw")
            return
        for b in self.bosses:
            if b.dying is None and b.age(now) >= b.lifetime:
                b.dying, b.end, b.pose = now, "fired", "shoot"
                for o in self.bosses:
                    if o in self.targets:
                        self.targets.remove(o)
                for s in (self.stats, self.team[b.side].stats):
                    s.escaped += 1
                self._hurt(now, "ЗАПІЗНО!", "enemy", b.side)
                break

    def _boss_down(self, boss: Figure, p: int, now: float) -> None:
        self.targets.remove(boss)
        took = boss.age(now)
        boss.dying, boss.end = now, "hit"
        self.team[p].draws.append(took)
        points = int(round((500 + 1000 * boss.remaining(now)) / 5.0)) * 5
        points = self._credit(p, points, now, None)
        self.shots.append(Shot(boss.x, boss.y - boss.height / 2, now, points,
                               f"+{points}   {took:.2f} с", True))
        self.sound.blip("hit")

    # -- High Noon, between the streets, for two ------------------------------
    def _start_noon(self, now: float) -> None:
        self.phase = NOON
        self.targets.clear()
        self.bosses = []
        self.noon_score = [0, 0]
        w, h = self.screen_size
        height = int(h * 0.13)
        self.bottles = [Bottle(x=w * (0.3 if side == 0 else 0.7), y=h * 0.62 - height / 2,
                               radius=height / 2, born=now, lifetime=math.inf, side=side,
                               height=height) for side in (0, 1)]
        self._noon_result = -1.0
        self._banner = ("РІВНО ОПІВДНІ", now + 2.0)
        self._noon_round(now + 1.5)

    def _noon_round(self, now: float) -> None:
        for b in self.bottles:
            b.dying = None
            if b in self.targets:
                self.targets.remove(b)
        self._drawn = False
        self._next = now + self.rng.uniform(2.0, 4.0)
        self.sound.blip("bell")

    def _noon(self, now: float) -> None:
        if self._noon_result >= 0:
            if now < self._noon_result:
                return
            self._noon_result = -1.0
            if max(self.noon_score) >= 2:
                winner = self.noon_score.index(2)
                self.team[winner].noon += 1
                bonus = self._credit(winner, 500, now, None, counts=False)
                self.phase = CLEAR
                self._banner = (f"ДУЕЛЬ ОПІВДНІ ВИГРАЄ {self.team[winner].name}  +{bonus}",
                                now + self.BANNER)
                self.sound.blip("clear")
                return
            self._noon_round(now)
            return
        if not self._drawn and now >= self._next:
            self._drawn, self._drawn_at = True, now
            self.targets.extend(self.bottles)
            self._message = ("СТРІЛЯЙ!", now + 0.8)
            self.sound.blip("draw")
        elif self._drawn and now - self._drawn_at > 3.0:
            self._noon_round(now)                # nobody: again

    def _noon_hit(self, bottle: Bottle, now: float) -> None:
        bottle.dying = now
        took = now - self._drawn_at
        self._noon_point(bottle.side, now, f"{self.team[bottle.side].name}  {took:.2f} с")
        self.sound.blip("bottle")

    def _noon_point(self, winner: int, now: float, words: str) -> None:
        self.noon_score[winner] += 1
        self._message = (words, now + 1.5)
        for b in self.bottles:
            if b in self.targets and b.dying is None:
                self.targets.remove(b)
        self._noon_result = now + 1.6
        self._drawn = False

    # -- drawing ------------------------------------------------------------
    def draw_pointer(self, canvas, x: float, y: float, t: float,
                     crosshair: bool = False) -> None:
        """A gun sight, not the tracker's spinning reticle, in the colour of
        the player whose half it is in; it kicks open when the gun fires."""
        side = self.side_of((x, y))
        colour = self.team[side].colour if side < len(self.team) else overlay.WHITE
        kick = max(0.0, 1.0 - (self.now - self._fired[side]) / self.KICK)
        overlay.draw_sight(canvas, x, y, colour, self.screen_size[1] / 1080.0, kick, crosshair)

    def draw(self, canvas) -> None:
        canvas[:] = art.backdrop(self.scene, self.screen_size)
        now = self.now
        if self.phase == SETUP:
            self._draw_setup(canvas)
            return
        self._draw_holes(canvas, now)
        if self.phase == NOON:
            self._draw_noon(canvas)
        if self.state != OVER:
            for b in self.bosses:
                if b not in self.targets:
                    self._draw_target(canvas, b, now)
        super().draw(canvas)
        if self.state == OVER:
            return
        for gun in self.guns:
            if gun._dwelt is not None and gun.dwell > 0 and gun.last is not None \
                    and not isinstance(gun._dwelt, Strip):
                cv2.ellipse(canvas, (int(gun.last[0]), int(gun.last[1])), (68, 68), -90, 0,
                            360 * min(1.0, gun.dwell / gun.DWELL_TIME), overlay.WHITE, 3,
                            cv2.LINE_AA)
        k = self.screen_size[1] / 1080.0
        h = self.screen_size[1]
        words, until = self._banner
        if now < until and not self.counting_down:
            overlay.text_centered(canvas, words, int(h * 0.42), 2.4 * k, overlay.CYAN, 4)
        words, until = self._message
        if now < until:
            colour = overlay.WHITE if words == "СТРІЛЯЙ!" else overlay.YELLOW
            overlay.text_centered(canvas, words, int(h * 0.30), 2.6 * k if len(words) < 14
                                  else 1.6 * k, colour, 5 if len(words) < 14 else 3)
        words, until = self._sub
        if now < until and not self.counting_down:
            y = int(h * 0.37) if now < self._message[1] else int(h * 0.50)
            overlay.text_centered(canvas, words, y, 1.0 * k, overlay.WHITE, 2)
        waiting = (self.phase == SHOWDOWN and not self._drawn and self.bosses
                   and all(b.dying is None for b in self.bosses)) or \
            (self.phase == NOON and not self._drawn and self._noon_result < 0)
        if waiting and now >= self._banner[1]:
            overlay.text_centered(canvas, "чекай...", int(h * 0.30), 1.2 * k,
                                  overlay.GREY, 2)
        if 0 <= now - self._shake < 0.3:
            # Hit: the screen jolts.
            # From the clock, not the game's dice: drawing must not change the game.
            a = 18 * k * (1 - (now - self._shake) / 0.3)
            canvas[:] = np.roll(canvas, (int(a * math.sin(now * 97.0)),
                                         int(a * math.cos(now * 71.0))), axis=(0, 1))

    def _draw_holes(self, canvas, now: float) -> None:
        """Where the misses went: a hole with splinters round it, and for a
        moment a puff of dust. They fade before the street fills up with them."""
        k = self.screen_size[1] / 1080.0
        h, w = canvas.shape[:2]
        r = max(4, int(7 * k))
        for x, y, when in self._holes:
            age = now - when
            if not 0 <= age < self.HOLE_TIME:
                continue
            fade = min(1.0, (self.HOLE_TIME - age) / 1.5)
            box = int(r * 11)
            x0, y0 = max(0, int(x) - box), max(0, int(y) - box)
            x1, y1 = min(w, int(x) + box), min(h, int(y) + box)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = canvas[y0:y1, x0:x1]
            mark = roi.copy()
            c = (int(x) - x0, int(y) - y0)
            # The same splinters every frame: from where it hit, not from the dice.
            seed = (int(x) * 7919 + int(y) * 104729) & 0xFFFF
            for i in range(7):
                a = (seed % 100) / 16.0 + i * math.tau / 7 + ((seed >> i) % 5 - 2) * 0.12
                n = r * (1.5 + (seed >> (i + 2)) % 7 * 0.22)
                tip = (int(c[0] + math.cos(a) * n), int(c[1] + math.sin(a) * n))
                cv2.line(mark, c, tip, (150, 165, 160), max(1, int(1.5 * k)), cv2.LINE_AA)
            cv2.circle(mark, c, r + 2, (70, 80, 80), -1, cv2.LINE_AA)
            cv2.circle(mark, c, r, (12, 12, 12), -1, cv2.LINE_AA)
            if age < 0.35:
                puff = age / 0.35
                cv2.circle(mark, c, int(r * (2 + 7 * puff)), (175, 185, 180),
                           max(1, int(3 * k * (1 - puff))), cv2.LINE_AA)
            cv2.addWeighted(mark, fade, roi, 1.0 - fade, 0, roi)

    def _draw_target(self, canvas, t: Target, now: float) -> None:
        if isinstance(t, Strip):
            self._draw_strip(canvas, t)
            return
        if isinstance(t, Pickup):
            self._draw_pickup(canvas, t, now)
            return
        if isinstance(t, Bottle):
            return                               # drawn with the posts
        s = t.sprite(now)
        if t.dying is None:
            # A boss is already standing there when he draws.
            rise = 1.0 if t in self.bosses and self._drawn else min(1.0, t.age(now) / self.RISE)
            if t.kind == PEEK:
                phase = t.age(now) % self.CYCLE
                if phase >= self.SHOW:
                    return                       # down behind his cover
                rise = min(1.0, phase / self.RISE, (self.SHOW - phase) / self.RISE)
            art.blit(canvas, s, t.x, t.foot, t.clip, rise)
            if t.armed and t not in self.bosses and rise >= 1.0:
                self._draw_fuse(canvas, t, s, now)
            return
        gone = now - t.dying
        if t.end == "hit":
            art.blit_fallen(canvas, s, t.x, t.foot, min(1.0, gone / (self.FALL * 0.6)),
                            t.flip, t.clip)
        elif t.end == "fired":
            down = max(0.0, (gone - self.FIRE) / self.RISE)
            art.blit(canvas, s, t.x, t.foot, t.clip, 1.0 - min(1.0, down))
            if gone < self.FIRE * 0.6:
                mx, my = s.muzzle
                self._flash(canvas, t.x - s.fx + mx, t.foot - s.fy + my, t.height)
        else:
            art.blit(canvas, s, t.x, t.foot, t.clip, max(0.0, 1.0 - gone / self.RISE))

    def _draw_fuse(self, canvas, t: Figure, s: art.Sprite, now: float) -> None:
        """A ring closing in on the bandit: when it is gone, he fires. A big
        one that needs a shot from each player has a ring in both colours."""
        left = t.remaining(now)
        # Round what shows of him - the sprite's picture has room to spare
        # round the figure, and behind a sill his legs are not there to see.
        # The first frame, so that the ring holds still while he moves.
        b = art.sprite(t.look, "stand", 0, t.height, t.flip).box
        x0, y0 = t.x - s.fx + b[0], t.foot - s.fy + b[1]
        x1, y1 = t.x - s.fx + b[2], t.foot - s.fy + b[3]
        if t.clip is not None:
            y1 = min(y1, t.clip)
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        reach = 0.5 * math.hypot(x1 - x0, y1 - y0)       # just clear of him
        r = int(reach * (0.8 + 0.45 * left))              # ...and closing on him
        if left <= 0.35 and int(now * 10) % 2:
            return
        if t.kind == TEAM:
            for a0, colour in ((90, overlay.CYAN), (270, overlay.YELLOW)):
                cv2.ellipse(canvas, (cx, cy), (r, r), 0, a0, a0 + 180, colour, 3, cv2.LINE_AA)
            overlay.text_fit(canvas, "РАЗОМ", (cx, int(y0 - 0.12 * t.height)), 160, 30,
                             overlay.WHITE, 2)
            return
        colour = overlay.CYAN if left > 0.35 else overlay.YELLOW
        cv2.circle(canvas, (cx, cy), r, colour, 2, cv2.LINE_AA)

    def _flash(self, canvas, x: float, y: float, height: int) -> None:
        r = max(6, int(height * 0.12))
        pts = np.array([(x + (r if i % 2 == 0 else r * 0.4) * math.cos(i * math.pi / 4),
                         y + (r if i % 2 == 0 else r * 0.4) * math.sin(i * math.pi / 4))
                        for i in range(8)], np.int32)
        cv2.fillPoly(canvas, [pts], overlay.YELLOW, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), max(2, r // 3), overlay.WHITE, -1, cv2.LINE_AA)

    def _draw_pickup(self, canvas, t: Pickup, now: float) -> None:
        r = int(t.radius)
        if t.dying is not None:
            k = (now - t.dying) / 0.4
            cv2.circle(canvas, (int(t.x), int(t.y)), int(r * (1 + 2 * k)),
                       tuple(int(c * max(0.0, 1 - k)) for c in overlay.YELLOW), 3, cv2.LINE_AA)
            return
        # Dim inside and bright only at the edge, like everything big enough
        # to aim at: a dot on a bright patch has no red left to be found by.
        dim = tuple(int(c * 0.4) for c in overlay.YELLOW)
        if t.kind == "star":
            self._star(canvas, int(t.x), int(t.y), r, True, dim)
            self._star(canvas, int(t.x), int(t.y), r, False, overlay.YELLOW)
        else:
            cv2.circle(canvas, (int(t.x), int(t.y)), r, dim, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(t.x), int(t.y)), r, overlay.YELLOW, 3, cv2.LINE_AA)
            overlay.text_fit(canvas, "$", (int(t.x), int(t.y)), r, r, overlay.YELLOW, 2)

    def _draw_strip(self, canvas, t: Strip) -> None:
        """The RELOAD bar: a part for each player, which calls for them when
        their gun is empty - lit in their colour, with arrows pointing at it."""
        w, h = self.screen_size
        k = h / 1080.0
        y = int(t.y)
        now = self.now
        parts = [(0, w)] if not self.two else [(0, w // 2), (w // 2, w)]
        for i, ((x0, x1), p) in enumerate(zip(parts, self.team)):
            empty, low = p.ammo == 0, 0 < p.ammo <= 2
            pulse = 0.5 + 0.5 * math.sin(now * 7.0)
            # Lit, but dim: a dot has to be found on it too.
            fill = tuple(int(c * (0.16 + 0.12 * pulse)) for c in p.colour) if empty \
                else (34, 34, 28)
            cv2.rectangle(canvas, (x0, y), (x1, h), fill, -1)
            edge, thick = (p.colour, 3) if empty else (overlay.YELLOW, 2) if low \
                else (overlay.GREY, 1)
            cv2.rectangle(canvas, (x0 + 1, y), (x1 - 2, h - 1), edge, thick, cv2.LINE_AA)
            cx = (x0 + x1) // 2
            if now < self._reloaded[i]:
                words, colour = "ПЕРЕЗАРЯДЖЕНО!", overlay.GREEN
            elif empty:
                words, colour = "СТРІЛЯЙ СЮДИ, ЩОБ ПЕРЕЗАРЯДИТИ", p.colour
            elif low:
                words, colour = f"ПЕРЕЗАРЯДКА  —  набоїв: {p.ammo}", overlay.YELLOW
            else:
                words, colour = "ПЕРЕЗАРЯДКА  —  стріляй сюди", overlay.GREY
            overlay.text_fit(canvas, words, (cx, (y + h) // 2), (x1 - x0) * 0.5,
                             (h - y) * 0.55, colour, 2 if empty or low else 1)
            if empty:
                # Arrows bouncing down at the bar.
                bounce = int(10 * k * abs(math.sin(now * 5.0)))
                for n in (-1, 0, 1):
                    ax = cx + n * int((x1 - x0) * 0.3)
                    ay = y - int(18 * k) + bounce
                    a = int(18 * k)
                    pts = np.array([(ax - a, ay - a), (ax, ay), (ax + a, ay - a)], np.int32)
                    cv2.polylines(canvas, [pts], False, p.colour, max(3, int(5 * k)),
                                  cv2.LINE_AA)

    def _draw_noon(self, canvas) -> None:
        h = self.screen_size[1]
        for b in self.bottles:
            post = (int(b.x - 0.02 * h), int(b.y + b.height / 2), int(b.x + 0.02 * h), int(h * 0.80))
            cv2.rectangle(canvas, post[:2], post[2:], (52, 58, 46), -1)
            cv2.rectangle(canvas, post[:2], post[2:], overlay.GREY, 1)
            if b.dying is None:
                jars.draw_jar(canvas, b.x, b.y + b.height / 2, b.height,
                              "small" if b.side == 0 else "gold", 1.0, marked=False)
            p = self.team[b.side]
            (tw, _), _ = cv2.getTextSize(p.name, overlay.FONT, 0.8, 2)
            overlay.text(canvas, p.name, (int(b.x - tw / 2), int(h * 0.86)), 0.8, p.colour, 2)
            for n in range(2):
                cx = int(b.x - 30 + 60 * n)
                won = n < self.noon_score[b.side]
                cv2.circle(canvas, (cx, int(h * 0.90)), 12, p.colour if won else overlay.DIM,
                           -1 if won else 2, cv2.LINE_AA)

    def _draw_setup(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        canvas[:] = (canvas * 0.6).astype(np.uint8)
        overlay.text_centered(canvas, "ДИКИЙ ЗАХІД  —  ДВОЄ ГРАВЦІВ", int(h * 0.14), 1.8 * k,
                              overlay.CYAN, 3)
        overlay.text_centered(canvas, f"{self.level.title}:  одне місто, спільні життя, "
                                      "а рахунок у кожного свій", int(h * 0.21), 0.85 * k, overlay.WHITE, 2)
        overlay.text_centered(canvas, "гравець 1 стріляє в лівій половині екрана, гравець 2 — у правій."
                                      "  щоб допомогти напарникові, не вимикай промінь"
                                      " і переведи його на інший бік", int(h * 0.27), 0.65 * k, overlay.GREY, 1)
        cv2.line(canvas, (w // 2, int(h * 0.35)), (w // 2, int(h * 0.85)), overlay.GREY, 2)
        for side, p in enumerate(self.team):
            cx = side * w // 2 + w // 4
            overlay.text_fit(canvas, p.name, (cx, int(h * 0.42)), w * 0.3, 50 * k, p.colour, 3)
            for i, (box, words) in enumerate(zip(self.setup_boxes(side),
                                                 ("ЗВИЧАЙНО", "ЛЕГШЕ"))):
                x0, y0, x1, y1 = box
                chosen = self._ready[side] and p.easy == (i == 1)
                fill = self._holds[side].fill(i)
                hot = fill > 0 or chosen
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1),
                              p.colour if hot else overlay.GREY, 2 if hot else 1)
                overlay.text_fit(canvas, words, ((x0 + x1) // 2, (y0 + y1) // 2 - int(10 * k)),
                                 (x1 - x0) * 0.8, (y1 - y0) * 0.3, overlay.WHITE, 2)
                note = "без поблажок" if i == 0 else "більше часу, ширший приціл"
                overlay.text_fit(canvas, note, ((x0 + x1) // 2, y1 - int(22 * k)),
                                 (x1 - x0) * 0.85, (y1 - y0) * 0.13, overlay.GREY, 1)
                if fill > 0:
                    cv2.rectangle(canvas, (x0, y1 - 8), (x0 + int((x1 - x0) * fill), y1),
                                  overlay.GREEN, -1)
            status = "ГОТОВО" if self._ready[side] else "затримай лазер на одній із кнопок"
            overlay.text_fit(canvas, status, (cx, int(h * 0.78)), w * 0.35, 40 * k,
                             overlay.GREEN if self._ready[side] else overlay.GREY, 2)
        overlay.text_centered(canvas, "SPACE — почати зараз, обом без поблажок", int(h * 0.90),
                              0.7 * k, overlay.GREY, 1)

    def _draw_hud(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        now = self.now
        if self.phase == STREET:
            left = self._left + sum(1 for t in self.figures() if t.armed and t.dying is None)
            status = f"{self.scene.title}   —   бандитів лишилося: {left}"
        elif self.phase == NOON:
            status = "РІВНО ОПІВДНІ   —   до двох перемог"
        else:
            status = f"{self.scene.title}   —   дуель"
        overlay.text_centered(canvas, f"ЕТАП {self.stage + 1}/{len(STAGES)}   {status}",
                              int(42 * k), 0.75 * k, overlay.WHITE, 2)
        extra = []
        if self.multiplier > 1.05:
            extra.append(f"x{self.multiplier:.1f}" + ("  ПЕРЕХРЕСНИЙ ВОГОНЬ" if self.crossfire else ""))
        if now < self._gold:
            extra.append(f"ЗОЛОТО x2  {self._gold - now:.0f} с")
        if extra:
            overlay.text_centered(canvas, "     ".join(extra), int(118 * k), 0.75 * k,
                                  overlay.GREEN, 2)
        stars_y = int(80 * k)
        if self.two:
            for i in range(self.level.lives):
                self._star(canvas, w // 2 + int((i - (self.level.lives - 1) / 2) * 46 * k),
                           stars_y, int(17 * k), i < self.lives)
        else:
            for i in range(self.level.lives):
                self._star(canvas, w - int((50 + 46 * i) * k), int(52 * k), int(18 * k),
                           i < self.lives)
        for i, p in enumerate(self.team):
            words = f"Г{i + 1}  {p.stats.score}" if self.two else f"РАХУНОК {p.stats.score}"
            (tw, _), _ = cv2.getTextSize(words, overlay.FONT, 1.1 * k, 2)
            x = int(40 * k) if i == 0 else w - int(40 * k) - tw
            overlay.text(canvas, words, (x, int(60 * k)), 1.1 * k, p.colour, 2)
            self._draw_ammo(canvas, i, p, now)

    def _draw_ammo(self, canvas, i: int, p: Player, now: float) -> None:
        if not self.level.ammo or self.phase == NOON:
            return
        w, h = self.screen_size
        k = h / 1080.0
        y1 = int(h * STRIP) - int(12 * k)
        bw, bh, gap = int(10 * k), int(30 * k), int(8 * k)
        for n in range(self.level.ammo):
            x = int(40 * k) + n * (bw + gap) if i == 0 else w - int(40 * k) - (n + 1) * (bw + gap)
            full = n < p.ammo
            cv2.rectangle(canvas, (x, y1 - bh), (x + bw, y1), p.colour if full else overlay.DIM,
                          -1 if full else 1)

    @staticmethod
    def _star(canvas, cx: int, cy: int, r: int, full: bool, colour=None) -> None:
        """A sheriff's star: a life."""
        pts = np.array([(cx + (r if i % 2 == 0 else r * 0.45) * math.sin(i * math.pi / 5),
                         cy - (r if i % 2 == 0 else r * 0.45) * math.cos(i * math.pi / 5))
                        for i in range(10)], np.int32)
        if full:
            cv2.fillPoly(canvas, [pts], colour or overlay.YELLOW, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [pts], True, colour or overlay.DIM, 2, cv2.LINE_AA)

    # -- the end ------------------------------------------------------------
    def rank(self) -> str:
        """What the town calls the sheriff - or the two - at the end."""
        cleared = len(STAGES) if self.won else self.stage
        shots = self.stats.shots + self.stats.decoys
        accuracy = self.stats.hits / shots if shots else 0.0
        if self.won and accuracy >= 0.8 and self.stats.decoys == 0:
            return "ЛЕГЕНДА ДИКОГО ЗАХОДУ"
        # Not the levels' names - deputy, sheriff, marshal - which are shown beside it.
        if self.won:
            return "ГОРДІСТЬ МІСТА"
        return "ГРОЗА БАНДИТІВ" if cleared >= 4 else "ДОБРА ПІДМОГА" if cleared >= 2 else "ЖОВТОДЗЬОБ"

    def awards(self) -> List[List[str]]:
        """One or two awards for each of two players, so that both go home
        with something."""
        a, b = self.team
        given: List[List[str]] = [[], []]

        def best(name: str, va: float, vb: float, higher: bool = True, enough: bool = True):
            if not enough or va == vb:
                if enough and va == vb and va:
                    given[0].append(name)
                    given[1].append(name)
                return
            given[0 if (va > vb) == higher else 1].append(name)

        best("СНАЙПЕР", a.accuracy, b.accuracy,
             enough=min(a.stats.shots, b.stats.shots) >= 5)
        if a.draws or b.draws:
            best("НАЙШВИДША РУКА", min(a.draws, default=99), min(b.draws, default=99), False)
        best("АНГЕЛ-ОХОРОНЕЦЬ", a.saves, b.saves, enough=a.saves + b.saves > 0)
        best("РІВНО ОПІВДНІ", a.noon, b.noon, enough=a.noon + b.noon > 0)
        for i, p in enumerate(self.team):
            if p.stats.decoys == 0 and p.stats.hits >= 5:
                given[i].append("ТВЕРДА РУКА")
        best("ГАРЯЧИЙ КУРОК", a.stats.misses, b.stats.misses,
             enough=max(a.stats.misses, b.stats.misses) >= 5)
        return [g[:2] or ["СПРАВЖНЯ МУЖНІСТЬ"] for g in given]

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        s = self.stats
        canvas[:] = (canvas * 0.45).astype(np.uint8)   # the town stays behind it, darker
        head = "МІСТО В БЕЗПЕЦІ" if self.won else "ГРУ ЗАКІНЧЕНО"
        overlay.text_centered(canvas, head, int(h * 0.13), 1.9 * k,
                              overlay.GREEN if self.won else overlay.CYAN, 3)
        overlay.text_centered(canvas, f"{self.rank()}    —    {self.level.title.lower()}, "
                                      f"етап {min(self.stage + 1, len(STAGES))} з {len(STAGES)}",
                              int(h * 0.20), 1.0 * k, overlay.YELLOW, 2)
        overlay.text_centered(canvas, f"{'рахунок команди' if self.two else 'рахунок'} {s.score}",
                              int(h * 0.28), 1.3 * k, overlay.WHITE, 2)
        if self.two:
            awards = self.awards()
            top = max(range(2), key=lambda i: self.team[i].stats.score)
            for i, p in enumerate(self.team):
                cx = w // 4 if i == 0 else 3 * w // 4
                ps = p.stats
                rows = [(p.name + ("   НАЙКРАЩИЙ РАХУНОК" if i == top and
                                   self.team[0].stats.score != self.team[1].stats.score else ""),
                         1.0, p.colour, 2),
                        (f"рахунок {ps.score}", 1.0, overlay.WHITE, 2),
                        (f"влучань: {ps.hits}    влучність {p.accuracy:.0%}", 0.7, overlay.GREY, 1),
                        (f"врятовано: {p.saves}    влучань у мирних: {ps.decoys}", 0.7, overlay.GREY, 1),
                        ((f"найшвидший постріл {min(p.draws):.2f} с" if p.draws else "жодної виграної дуелі")
                         + (f"    рівно опівдні: {p.noon}" if p.noon else ""), 0.7, overlay.GREY, 1),
                        ("  -  ".join(awards[i]), 0.8, overlay.GREEN, 2)]
                y = int(h * 0.38)
                for words, scale, colour, thick in rows:
                    overlay.text_fit(canvas, words, (cx, y), w * 0.42, 40 * scale * k,
                                     colour, thick)
                    y += int(52 * k)
        else:
            shots = s.shots + s.decoys
            lines = [
                (f"бандитів знешкоджено: {s.hits}    влучань у мирних: {s.decoys}    "
                 f"поранень: {s.escaped}",
                 0.75, overlay.GREY, 1),
                (f"влучність {s.hits / shots if shots else 0:.0%}    найдовша серія {s.best_combo}    "
                 f"реакція {s.reaction:.2f} с"
                 + (f"    найшвидший постріл {min(self.team[0].draws):.2f} с"
                    if self.team and self.team[0].draws else ""), 0.75, overlay.GREY, 1),
            ]
            overlay.draw_panel(canvas, lines, h * 0.42)
        y = int(h * 0.74)
        if rank:
            overlay.text_centered(canvas, f"НОВИЙ РЕКОРД — №{rank}", y, 0.9 * k,
                                  overlay.GREEN, 2)
        if scores is not None and scores.entries:
            for i, e in enumerate(scores.entries[:3]):
                y += int(30 * k)
                overlay.text_centered(
                    canvas, f"{i + 1}.  {HighScores.who(e)}{e.get('score', 0):>6}   "
                            f"влучність {e.get('accuracy', 0):.0%}   {e.get('date', '')}",
                    y, 0.6 * k, overlay.WHITE if i + 1 == rank else overlay.GREY)
        overlay.text_centered(canvas, "G — грати ще    ESC — вихід", int(h * 0.93),
                              0.7 * k, overlay.YELLOW, 1)
