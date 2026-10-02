"""Two maths games on the round engine in game.py.

* **Balloon Math** (grades 2-4) - a sum at the top, a few balloons with
  numbers; hold the laser on the right answer. No clock, no lives.
* **Number Hunt** (grades 5-9) - a rule at the top ("multiples of 7"),
  numbered targets drifting about; shoot the ones that fit, leave the rest.

Everything on screen is digits and symbols: the built-in OpenCV font has no
letters beyond ASCII, and numbers read the same in any classroom language.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import overlay
from .game import (COLOURS, DECOY, NORMAL, OVER, Game, HighScores, Round, Shot,
                   Target)

BLUE = COLOURS[DECOY]


# ---------------------------------------------------------------------------
# Balloon Math
# ---------------------------------------------------------------------------
@dataclass
class Task:
    text: str               # "7 + 5"
    answer: int
    choices: List[int]      # the answer and the wrong ones, shuffled


def make_task(rng, level: int, n_choices: int) -> Task:
    """One sum for `level`, with wrong answers a child could plausibly reach:
    off by one or ten, or what the other operation would have given."""
    near = [1, -1, 2, -2] + ([10, -10] if level >= 2 else [3, -3])
    if level == 3:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        answer, text = a * b, f"{a} x {b}"
        wrong = [a * (b + 1), a * (b - 1), (a + 1) * b, (a - 1) * b, a + b]
        wrong += [answer + d for d in near[:4]]
    else:
        top = 20 if level == 1 else 100
        if rng.random() < 0.5:
            a = rng.randint(1, top - 1) if level == 1 else rng.randint(11, top - 11)
            b = rng.randint(1 if level == 1 else 2, top - a)
            answer, text, other = a + b, f"{a} + {b}", abs(a - b)
        else:
            a = rng.randint(2, top) if level == 1 else rng.randint(21, top)
            b = rng.randint(1, a - 1) if level == 1 else rng.randint(2, a - 2)
            answer, text, other = a - b, f"{a} - {b}", a + b
        wrong = [answer + d for d in near]
        if other <= top:
            wrong.append(other)
    wrong = sorted({c for c in wrong if c >= 0 and c != answer})
    picked = rng.sample(wrong, min(n_choices - 1, len(wrong)))
    extra = answer + 4
    while len(picked) < n_choices - 1:      # tiny answers leave few neighbours
        if extra not in picked:
            picked.append(extra)
        extra += 1
    choices = picked + [answer]
    rng.shuffle(choices)
    return Task(text, answer, choices)


@dataclass
class Balloon(Target):
    home_x: float = 0.0
    home_y: float = 0.0
    phase: float = 0.0
    colour: Tuple[int, int, int] = overlay.CYAN
    tried: Optional[float] = None      # when it was picked and turned out wrong


class BalloonMath(Round):
    """Ten sums, one at a time. Nothing is ever lost: a wrong balloon wobbles
    and greys out, and the sum stays until the right one is found."""

    FLASH = False            # small hands wave the beam about; only a hold counts
    DWELL_TIME = 0.5
    HINT = "hold the laser on the balloon with the right answer"
    TASKS = 10
    RISE = 1.2               # seconds for the balloons to float into place
    SOLVED_PAUSE = 1.4       # the finished sum stays up this long
    LEVELS = {
        1: "+ and -  up to 20",
        2: "+ and -  up to 100",
        3: "multiplication table",
    }
    PALETTE = (overlay.CYAN, overlay.YELLOW, BLUE, overlay.GREEN)

    def __init__(self, screen_size: Tuple[int, int], level: int = 1,
                 tasks: int = TASKS, seed: Optional[int] = None) -> None:
        super().__init__(screen_size, seed)
        self.level = level
        self.n_tasks = tasks
        self.n_choices = 3 if level == 1 else 4
        self.tasks: List[Task] = []
        self.results: List[bool] = []      # per solved sum: right first time?
        self.task: Optional[Task] = None
        self._solved: Optional[Task] = None
        self._clean = True
        self._next_at = 0.0

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        self.tasks = []
        seen = set()
        while len(self.tasks) < self.n_tasks:
            task = make_task(self.rng, self.level, self.n_choices)
            if task.text not in seen or len(seen) > 40:
                seen.add(task.text)
                self.tasks.append(task)
        self.results = []
        self.task = None
        self._solved = None
        self._next_at = self.started

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        self._next_at += delta
        for t in self.targets:
            if t.tried is not None:
                t.tried += delta
            if t.dying is not None:
                t.dying += delta

    @property
    def first_try(self) -> int:
        return sum(self.results)

    @property
    def stars(self) -> int:
        share = self.first_try / self.n_tasks if self.n_tasks else 0.0
        return 3 if share >= 0.9 else 2 if share >= 0.6 else 1

    def _finished(self) -> bool:
        return len(self.results) >= len(self.tasks) and self.now >= self._next_at

    # -- balloons -----------------------------------------------------------
    def _show(self, task: Task, now: float) -> None:
        w, h = self.screen_size
        radius = min(w, h) * 0.12
        n = len(task.choices)
        self.targets.clear()
        for i, value in enumerate(task.choices):
            x = w * (i + 1) / (n + 1)
            self.targets.append(Balloon(
                x=x, y=h + radius, radius=radius, born=now, lifetime=math.inf,
                label=str(value), home_x=x,
                home_y=h * 0.60 + self.rng.uniform(-0.06, 0.06) * h,
                phase=self.rng.uniform(0, 2 * math.pi),
                colour=self.PALETTE[i % len(self.PALETTE)],
            ))
        self.task = task
        self._solved = None
        self._clean = True

    def _float(self, now: float) -> None:
        h = self.screen_size[1]
        for t in self.targets:
            if t.dying is not None:
                continue
            k = min(1.0, t.age(now) / self.RISE)
            ease = 1.0 - (1.0 - k) ** 3
            start = h + t.radius
            bob = math.sin(t.age(now) * 1.3 + t.phase) * h * 0.015
            t.y = start + (t.home_y - start) * ease + bob * ease
            t.x = t.home_x
            if t.tried is not None:
                since = now - t.tried
                t.x += math.sin(since * 30.0) * 14.0 * max(0.0, 1.0 - since / 0.5)

    def _target_at(self, x: float, y: float, slack: float = 0.0) -> Optional[Target]:
        live = [t for t in self.targets
                if t.dying is None and t.tried is None and t.contains(x, y, slack)]
        if not live:
            return None
        return min(live, key=lambda t: math.hypot(t.x - x, t.y - y))

    def _hit(self, target: Target, now: float, how: str) -> None:
        if int(target.label) != self.task.answer:
            target.tried = now
            target.dwell = 0.0
            self.stats.misses += 1
            self._clean = False
            self.shots.append(Shot(target.x, target.y - target.radius, now, 0,
                                   "try another one", False))
            return
        self.stats.hits += 1
        self.results.append(self._clean)
        self.stats.score = self.first_try
        self.shots.append(Shot(target.x, target.y - target.radius, now, 1, "yes!", True))
        for t in self.targets:
            t.dying = now
        self._solved, self.task = self.task, None
        self._next_at = now + self.SOLVED_PAUSE

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        for t in list(self.targets):
            if t.dying is not None and now - t.dying > 0.35:
                self.targets.remove(t)
        if self.task is None and now >= self._next_at and len(self.results) < len(self.tasks):
            self._show(self.tasks[len(self.results)], now)
        self._float(now)
        self._aim(now, dt, point)

    # -- drawing ------------------------------------------------------------
    def _draw_target(self, canvas, t: Target, now: float) -> None:
        x, y, r = int(t.x), int(t.y), int(t.radius)
        colour = overlay.DIM if t.tried is not None else t.colour
        if t.dying is not None:
            k = (now - t.dying) / 0.35
            fade = max(0.0, 1.0 - k)
            right = self._solved is not None and int(t.label) == self._solved.answer
            ring = overlay.GREEN if right else overlay.DIM
            cv2.circle(canvas, (x, y), int(r * (1 + 0.8 * k)),
                       tuple(int(c * fade) for c in ring), 4, cv2.LINE_AA)
            return

        axes = (int(r * 0.9), r)
        # Only a faint fill: a laser dot on a brightly lit patch has nothing
        # left to stand out against, and the detector would lose it.
        cv2.ellipse(canvas, (x, y), axes, 0, 0, 360,
                    tuple(int(c * 0.16) for c in colour), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x, y), axes, 0, 0, 360, colour, 4, cv2.LINE_AA)
        knot = np.array([[x, y + r], [x - 10, y + r + 16], [x + 10, y + r + 16]], np.int32)
        cv2.fillPoly(canvas, [knot], colour, cv2.LINE_AA)
        string = np.array([[x + int(8 * math.sin(i * 0.9 + t.phase)), y + r + 16 + i * 12]
                           for i in range(int(r * 0.07) + 4)], np.int32)
        cv2.polylines(canvas, [string], False, tuple(int(c * 0.6) for c in colour),
                      2, cv2.LINE_AA)
        overlay.text_fit(canvas, t.label, (x, y), r * 1.25, r * 0.6,
                         overlay.GREY if t.tried is not None else overlay.WHITE, 4)
        if t.dwell > 0:
            cv2.ellipse(canvas, (x, y), (axes[0] + 12, axes[1] + 12), -90, 0,
                        360 * min(1.0, t.dwell / self.DWELL_TIME),
                        overlay.WHITE, 5, cv2.LINE_AA)

    def _draw_hud(self, canvas) -> None:
        h = self.screen_size[1]
        for i in range(len(self.tasks)):
            centre = (50 + i * 34, 50)
            if i < len(self.results):
                colour = overlay.GREEN if self.results[i] else overlay.GREY
                cv2.circle(canvas, centre, 11, colour, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, centre, 11, overlay.DIM, 1, cv2.LINE_AA)
        scale = h / 270.0
        if self.task is not None:
            overlay.text_centered(canvas, f"{self.task.text} = ?", int(h * 0.24),
                                  scale, overlay.CYAN, max(2, int(scale * 2)))
        elif self._solved is not None:
            overlay.text_centered(canvas, f"{self._solved.text} = {self._solved.answer}",
                                  int(h * 0.24), scale, overlay.GREEN, max(2, int(scale * 2)))

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        w, h = self.screen_size
        size = h * 0.09
        for i in range(3):
            centre = (w / 2 + (i - 1) * size * 2.6, h * 0.24)
            _star(canvas, centre, size, filled=i < self.stars)
        overlay.draw_panel(canvas, [
            ("PERFECT!" if self.first_try == self.n_tasks
             else ("WELL DONE!", "GOOD JOB!", "GREAT!")[self.stars - 1],
             1.8, overlay.CYAN, 3),
            (f"{self.first_try} of {self.n_tasks} right the first time",
             1.1, overlay.WHITE, 2),
            (self.LEVELS.get(self.level, ""), 0.75, overlay.GREY, 1),
            ("G play again    ESC back to tracking", 0.7, overlay.YELLOW, 1),
        ], h * 0.56)


def _star(canvas, centre, size: float, filled: bool) -> None:
    pts = []
    for i in range(10):
        r = size if i % 2 == 0 else size * 0.42
        a = -math.pi / 2 + i * math.pi / 5
        pts.append([centre[0] + r * math.cos(a), centre[1] + r * math.sin(a)])
    pts = np.array(pts, np.int32)
    if filled:
        cv2.fillPoly(canvas, [pts], overlay.YELLOW, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [pts], True, overlay.DIM, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Number Hunt
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """A rule and the values it is played with; `yes` are the labels that fit."""
    text: str
    test: Callable[[object], bool]
    pool: Sequence
    show: Callable[[object], str] = str
    yes: List[str] = field(init=False)
    no: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.yes = [self.show(v) for v in self.pool if self.test(v)]
        self.no = [self.show(v) for v in self.pool if not self.test(v)]


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _multiples(rng) -> Rule:
    k = rng.choice((3, 4, 6, 7, 8, 9))
    return Rule(f"multiples of {k}", lambda n: n % k == 0, range(k + 1, 100))


def _divisors(rng) -> Rule:
    n = rng.choice((24, 36, 48, 60))
    return Rule(f"divisors of {n}", lambda d: n % d == 0, range(2, n // 2 + 1))


def _parity(rng) -> Rule:
    even = rng.random() < 0.5
    return Rule("even numbers" if even else "odd numbers",
                lambda n: (n % 2 == 0) == even, range(11, 200))


def _primes(rng) -> Rule:
    return Rule("prime numbers", _is_prime, range(2, 60))


def _fractions(rng) -> Rule:
    p, q = rng.choice(((1, 2), (1, 3), (2, 3), (3, 4)))
    pool = [(a, b) for b in range(2, 17) for a in range(1, b)]
    return Rule(f"equal to {p}/{q}", lambda v: v[0] * q == v[1] * p, pool,
                lambda v: f"{v[0]}/{v[1]}")


def _compare(rng) -> Rule:
    c, op = rng.randint(-6, 6), rng.choice("<>")
    return Rule(f"x {op} {c}", lambda x: x > c if op == ">" else x < c, range(-12, 13))


def _linear(rng) -> Rule:
    # Built backwards from the boundary, so it always falls on a whole number.
    a, b, edge, op = rng.choice((2, 3)), rng.choice((-1, 1)) * rng.randint(1, 9), \
        rng.randint(-3, 4), rng.choice("<>")
    c = a * edge + b
    text = f"{a}x {'+' if b > 0 else '-'} {abs(b)} {op} {c}"
    return Rule(text, lambda x: a * x + b > c if op == ">" else a * x + b < c,
                range(-9, 10))


def _absolute(rng) -> Rule:
    k, op = rng.randint(3, 7), rng.choice("<>")
    return Rule(f"|x| {op} {k}", lambda x: abs(x) > k if op == ">" else abs(x) < k,
                range(-10, 11))


def _squares(rng) -> Rule:
    squares = {n * n for n in range(1, 16)}
    # The near misses are what make it a question: 48, 50, 63, 65...
    pool = sorted(squares | {s + d for s in squares for d in (-2, -1, 1, 2) if s + d > 1})
    return Rule("perfect squares", lambda n: n in squares, pool)


def _powers(rng) -> Rule:
    base = rng.choice((2, 3))
    powers = {base ** e for e in range(1, 9 if base == 2 else 6)}
    pool = sorted(powers | set(range(base, 100, base)))
    return Rule(f"powers of {base}", lambda n: n in powers, pool)


PACKS = {
    "5-6": (_multiples, _divisors, _parity, _primes, _fractions),
    "7-9": (_compare, _linear, _absolute, _squares, _powers),
}


class NumberHunt(Game):
    """The gallery with a question attached: every target carries a number,
    and only the ones that fit the rule should be shot. A wrong number costs
    what a decoy costs; a right one that gets away costs a life."""

    HINT = "shoot only the numbers that fit the rule"
    RULE_TIME = 15.0         # seconds per rule
    BANNER = 2.0             # a new rule is announced for this long, no targets
    MATCH_SHARE = 0.35       # chance a new target fits; see _pick_kind

    def __init__(self, screen_size: Tuple[int, int], pack: str = "5-6",
                 duration: float = Game.DURATION, lives: int = Game.LIVES,
                 seed: Optional[int] = None) -> None:
        super().__init__(screen_size, duration, lives, seed)
        self.pack = pack
        self.scores_name = f"highscores-hunt-{pack}"
        self.rules: List[Rule] = [PACKS[pack][0](self.rng)]
        self.rule_index = 0
        self.banner_until = 0.0
        self._dry = 0            # targets in a row that did not fit

    @property
    def rule(self) -> Rule:
        return self.rules[self.rule_index]

    def start(self, now: float) -> None:
        super().start(now)
        makers = list(PACKS[self.pack])
        self.rng.shuffle(makers)
        count = max(1, math.ceil(self.duration / self.RULE_TIME))
        self.rules = [makers[i % len(makers)](self.rng) for i in range(count)]
        self.rule_index = 0
        self.banner_until = 0.0
        self._dry = 0

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        self.banner_until += delta

    # -- difficulty: slower than the gallery, there is thinking to do --------
    def _spawn_gap(self) -> float:
        return 1.6 - 0.7 * self._ramp()

    def _radius(self) -> float:
        big = min(self.screen_size) * 0.085
        small = min(self.screen_size) * 0.055
        return big - (big - small) * self._ramp()

    def _lifetime(self) -> float:
        return 5.0 - 2.0 * self._ramp()

    # -- targets ------------------------------------------------------------
    def _pick_kind(self) -> str:
        # Left to chance alone, a long run of numbers that do not fit leaves
        # nothing to shoot; a fitting one is forced after three. Together
        # with MATCH_SHARE that makes roughly 4 targets in 10 fit.
        fits = self._dry >= 3 or self.rng.random() < self.MATCH_SHARE
        self._dry = 0 if fits else self._dry + 1
        return NORMAL if fits else DECOY

    def spawn(self, now: float) -> Target:
        t = super().spawn(now)
        pool = self.rule.yes if t.kind == NORMAL else self.rule.no
        shown = {other.label for other in self.targets}
        t.label = self.rng.choice([v for v in pool if v not in shown] or pool)
        return t

    def _decoy_text(self, target: Target) -> str:
        return f"{target.label}: no  -25"

    def _escape(self, target: Target, now: float) -> None:
        super()._escape(target, now)
        if target.kind == NORMAL:
            self.shots.append(Shot(target.x, target.y, now, 0,
                                   f"missed {target.label}", False))

    def _step(self, now: float, dt: float, point: Optional[Tuple[float, float]]) -> None:
        index = min(int(self.elapsed // self.RULE_TIME), len(self.rules) - 1)
        if index != self.rule_index:
            self.rule_index = index
            # Numbers judged by the old rule leave without costing anything.
            for t in self.targets:
                if t.dying is None:
                    t.dying = now
            self.banner_until = now + self.BANNER
            self.next_spawn = max(self.next_spawn, self.banner_until)
        super()._step(now, dt, point)

    # -- drawing ------------------------------------------------------------
    def _draw_target(self, canvas, t: Target, now: float) -> None:
        x, y, r = int(t.x), int(t.y), int(t.radius)
        if t.dying is not None:
            # Only now does the colour say whether it fitted.
            colour = overlay.GREEN if t.kind == NORMAL else BLUE
            k = (now - t.dying) / 0.35
            fade = max(0.0, 1.0 - k)
            colour = tuple(int(c * fade) for c in colour)
            cv2.circle(canvas, (x, y), int(r * (1 + 0.6 * k)), colour, 3, cv2.LINE_AA)
            overlay.text_fit(canvas, t.label, (x, y), r * 1.3, r * 0.55, colour, 2)
            return
        colour = overlay.CYAN
        cv2.circle(canvas, (x, y), r, colour, 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (x, y), (r + 8, r + 8), -90, 0, 360 * t.remaining(now),
                    tuple(int(c * 0.8) for c in colour), 2, cv2.LINE_AA)
        overlay.text_fit(canvas, t.label, (x, y), r * 1.3, r * 0.55, overlay.WHITE, 2)
        if t.dwell > 0:
            cv2.ellipse(canvas, (x, y), (r - 6, r - 6), -90, 0,
                        360 * min(1.0, t.dwell / self.DWELL_TIME),
                        overlay.WHITE, 3, cv2.LINE_AA)

    def _draw_hud(self, canvas) -> None:
        super()._draw_hud(canvas)
        overlay.text_centered(canvas, f"SHOOT:  {self.rule.text}", 150, 1.4,
                              overlay.YELLOW, 3)
        if self.state != OVER and self.now < self.banner_until:
            overlay.draw_panel(canvas, [
                ("NEW RULE", 1.0, overlay.CYAN, 2),
                (self.rule.text, 2.2, overlay.YELLOW, 4),
            ], self.screen_size[1] / 2)
