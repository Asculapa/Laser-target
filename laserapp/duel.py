"""Range duel, on the round engine in game.py: two players, a laser each, the
screen split down the middle.

Each half is a lane of a pop-up range. Figures come up from behind two walls
(duel_art.py): armed ones to shoot, and people who must not be shot. Both
lanes are given exactly the same figures at the same moments, so the only
difference between the two scores is the two players.

Nothing tells one laser from the other except where it points: a dot counts
for the half it is in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

from . import duel_art as art
from . import overlay, sound
from .game import OVER, PAUSED, PLAYING, HighScores, Round, Shot, Target
from .story import Hold

ASSETS = Path(__file__).parent / "assets" / "duel"
MUSIC = "music"
ROWS = ((0.47, 0.7), (0.90, 1.0))   # the two walls: where on the screen, and
                                    # how tall its figures are next to the near ones
COLUMNS = 3
WALL = (30, 30, 26)

Point = Tuple[float, float]


@dataclass(frozen=True)
class Popup:
    """One figure of the round, the same in both lanes."""
    at: float               # seconds into the round
    station: int
    look: str
    lifetime: float


@dataclass
class Figure(Target):
    """A figure standing at a wall; (x, y) is its middle."""
    look: str = ""
    height: int = 0
    station: int = 0
    shot: bool = False      # it went down to a shot, not to the clock

    def zone_at(self, x: float, y: float) -> int:
        """Whose pixel (x, y) is: art.FOE, art.FRIEND, or 0 for a miss."""
        _, zone = art.render(self.look, self.height)
        col = int(x - self.x + zone.shape[1] / 2)
        row = int(y - self.y + self.height / 2)
        if 0 <= row < zone.shape[0] and 0 <= col < zone.shape[1]:
            return int(zone[row, col])
        return 0

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        return self.zone_at(x, y) != 0


class Lane(Round):
    """One player's half of the range, and that player's score."""

    PENALTY = 200            # for shooting someone unarmed: more than any hit pays
    RISE, DROP = 0.14, 0.22  # seconds a figure takes to come up, and to go down

    def __init__(self, screen_size: Tuple[int, int], span: Tuple[int, int],
                 plan: Sequence[Popup], name: str, colour) -> None:
        super().__init__(screen_size)
        self.span = span
        self.plan = plan
        self.name = name
        self.colour = colour
        self.combo = 0
        self.events: List[str] = []          # sounds to play, for the duel to pick up
        self._next = 0                       # the first figure of the plan still to come
        self._at: Optional[Point] = None     # where the beam is, for the shot
        self.stations = self._stations()

    def _stations(self) -> List[Tuple[float, float, int]]:
        """Where a figure can come up: its centre line, its foot, its height."""
        h = self.screen_size[1]
        x0, x1 = self.span
        near = int(min(0.36 * h, (x1 - x0) / COLUMNS / (art.ASPECT + 0.06)))
        return [(x0 + (x1 - x0) * (2 * i + 1) / (2 * COLUMNS), h * foot, int(near * size))
                for foot, size in ROWS for i in range(COLUMNS)]

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        self.combo = 0
        self.events.clear()
        self._next = 0

    def _resume(self, delta: float) -> None:
        super()._resume(delta)
        for t in self.targets:
            if t.dying is not None:
                t.dying += delta

    def _finished(self) -> bool:
        return False                         # the duel calls time, for both lanes

    def mine(self, points: Sequence[Point]) -> Optional[Point]:
        """The dot that is this player's: the strongest one in this half."""
        return next((p for p in points if self.span[0] <= p[0] < self.span[1]), None)

    @property
    def score(self) -> int:
        return self.stats.score

    @property
    def multiplier(self) -> float:
        return 1.0 + 0.1 * min(self.combo, 10)

    @property
    def accuracy(self) -> float:
        shots = self.stats.shots + self.stats.decoys
        return self.stats.hits / shots if shots else 0.0

    # -- the shot -----------------------------------------------------------
    def _points(self, t: Figure, now: float) -> int:
        base = 60 + 40 * t.remaining(now)                   # reward quick shots
        base *= self.stations[-1][2] / float(t.height)      # ...and far ones
        if art.LOOKS[t.look].mixed:
            base *= 2.0
        return int(round(base * self.multiplier / 5.0)) * 5

    def _hit(self, target: Figure, now: float, how: str) -> None:
        target.dying = now
        target.shot = True
        if target.zone_at(*self._at) == art.FRIEND:
            self.combo = 0
            self.stats.decoys += 1
            self.stats.score -= self.PENALTY
            self.shots.append(Shot(self._at[0], self._at[1], now, -self.PENALTY,
                                   f"БЕЗЗБРОЙНИЙ -{self.PENALTY}", False))
            self.events.append("bad")
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
        self.shots.append(Shot(self._at[0], self._at[1], now, points, label, True))
        self.events.append("hit")

    def _miss(self, x: float, y: float, now: float) -> None:
        self.combo = 0
        self.stats.misses += 1
        self.shots.append(Shot(x, y, now, 0, "мимо", False))
        self.events.append("miss")

    # -- main update --------------------------------------------------------
    def _step(self, now: float, dt: float, point: Optional[Point]) -> None:
        while self._next < len(self.plan) and self.plan[self._next].at <= self.elapsed:
            p = self.plan[self._next]
            self._next += 1
            self.targets.append(self._arrive(p))
        for t in list(self.targets):
            if t.dying is not None:
                if now - t.dying > self.DROP:
                    self.targets.remove(t)
            elif t.age(now) >= t.lifetime:
                t.dying = now
                self._expired(t)
        self._at = point
        self._aim(now, dt, point)

    def _arrive(self, p: Popup) -> Target:
        x, foot, height = self.stations[p.station]
        return Figure(x=x, y=foot - height / 2, radius=height / 2,
                      born=self.started + p.at, lifetime=p.lifetime,
                      look=p.look, height=height, station=p.station)

    def _expired(self, t: Figure) -> None:
        """`t` has gone down unshot."""
        if art.LOOKS[t.look].foe:            # an armed one nobody shot
            self.stats.escaped += 1
            self.combo = 0

    # -- drawing ------------------------------------------------------------
    def draw_field(self, canvas) -> None:
        h = self.screen_size[1]
        x0, x1 = self.span
        for t in self.targets:
            self._draw_target(canvas, t, self.now)
        for foot, _ in ROWS:
            y = int(h * foot)
            cv2.rectangle(canvas, (x0, y), (x1, y + int(h * 0.028)), WALL, -1)
            cv2.line(canvas, (x0, y), (x1, y), overlay.GREY, 2, cv2.LINE_AA)
        for t in self.targets:
            if t.dying is None:
                # The time it stays up, as a bar on the wall in front of it.
                left = t.remaining(self.now)
                half = int(0.3 * t.height * left)
                y = int(t.y + t.height / 2 + h * 0.014)
                cv2.line(canvas, (int(t.x) - half, y), (int(t.x) + half, y),
                         overlay.CYAN if left > 0.25 else overlay.YELLOW, 4, cv2.LINE_AA)
        for s in self.shots:
            self._draw_shot(canvas, s, self.now)
        if self._dwelt is not None and self._dwelt.dwell > 0 and self._at is not None:
            # Outside the reticle the app draws on the pointer.
            cv2.ellipse(canvas, (int(self._at[0]), int(self._at[1])), (68, 68), -90, 0,
                        360 * min(1.0, self._dwelt.dwell / self.DWELL_TIME),
                        overlay.WHITE, 3, cv2.LINE_AA)

    def _draw_target(self, canvas, t: Figure, now: float) -> None:
        img, zone = art.render(t.look, t.height)
        # It comes up from behind the wall, and goes back down behind it.
        up = min(1.0, t.age(now) / self.RISE) if t.dying is None else \
            max(0.0, 1.0 - (now - t.dying) / self.DROP)
        rows = int(img.shape[0] * up)
        foot = int(t.y + t.height / 2)
        x0 = int(t.x - img.shape[1] / 2)
        y0 = foot - rows
        a0, a1 = max(0, x0), min(canvas.shape[1], x0 + img.shape[1])
        b0 = max(0, y0)
        if rows <= 0 or a0 >= a1 or b0 >= foot:
            return
        part = (slice(b0 - y0, rows), slice(a0 - x0, a1 - x0))
        solid = zone[part] > 0
        canvas[b0:foot, a0:a1][solid] = img[part][solid]


class Duel(Round):
    """A minute on the range for two: the same figures in both halves, and
    whoever has the higher score when time is called has won."""

    DURATION = 60.0
    HINT = "гравець 1 ліворуч, гравець 2 праворуч — стріляй в озброєних, решту не чіпай"
    players = 2
    own_buttons = True       # REMATCH, held with either laser
    PLAYERS = (("ГРАВЕЦЬ 1", overlay.CYAN), ("ГРАВЕЦЬ 2", overlay.YELLOW))
    SETTLE = 1.5             # the rematch button ignores the pointer this long:
                             # at the last second both players are still shooting
    LANE = Lane              # what a game on the same split screen changes
    SOUNDS = ASSETS
    TUNE: Optional[str] = MUSIC
    WINS = "виграно раундів"

    def __init__(self, screen_size: Tuple[int, int], duration: float = DURATION,
                 seed: Optional[int] = None) -> None:
        super().__init__(screen_size, seed)
        self.duration = duration
        self.lanes: List[Lane] = []
        self.plan: List[Popup] = []
        self.wins = [0, 0]                   # rounds won since the duel was chosen
        self.winner: Optional[int] = None    # of the round just played; None is a draw
        self.sound = sound.Player(self.SOUNDS)
        self._settled = False
        self._over_at = 0.0
        self._hold = Hold()                  # on the rematch button

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        w = self.screen_size[0]
        self.plan = self._plan()
        self.lanes = [self.LANE(self.screen_size, span, self.plan, name, colour)
                      for span, (name, colour) in zip(((0, w // 2), (w // 2, w)), self.PLAYERS)]
        for lane in self.lanes:
            lane.start(now)
        self.winner = None
        self._settled = False
        self.sound.stop()                    # a rematch: the music from the top

    def close(self) -> None:
        self.sound.stop()

    def toggle_pause(self, now: float) -> None:
        was = self.state
        super().toggle_pause(now)
        if self.state == was:
            return
        for lane in self.lanes:
            lane.toggle_pause(now)
        if self.state == PAUSED:
            self.sound.hold()
        else:
            self.sound.release()

    def key(self, key: int) -> bool:
        if key in (ord("g"), ord("G")):
            # A rematch here, rather than the app starting a new duel: this
            # one knows how many rounds each player has won.
            self.start(self.now)
            return True
        return False

    @property
    def time_left(self) -> float:
        return max(0.0, self.duration - self.elapsed)

    def _finished(self) -> bool:
        return self.time_left <= 0

    # -- the figures --------------------------------------------------------
    def _plan(self) -> List[Popup]:
        """Every figure of the round: when it comes up, where, who it is and
        for how long. They come quicker and leave sooner as the round goes on."""
        plan: List[Popup] = []
        busy = [0.0] * (len(ROWS) * COLUMNS)         # when each station is free again
        at = 0.3
        while at < self.duration - 1.0:
            k = at / self.duration
            for _ in range(2 if self.rng.random() < 0.35 * k else 1):
                free = [i for i, until in enumerate(busy) if until <= at]
                if not free:
                    break
                station = self.rng.choice(free)
                look = self._pick(k, near=station >= len(busy) - COLUMNS)
                lifetime = (2.8 - 1.3 * k) * (1.3 if art.LOOKS[look].mixed else 1.0)
                plan.append(Popup(at, station, look, lifetime))
                busy[station] = at + lifetime + Lane.DROP + 0.25
            at += (1.3 - 0.65 * k) * self.rng.uniform(0.8, 1.2)
        return plan

    def _pick(self, k: float, near: bool) -> str:
        if self.rng.random() < 0.36:
            return self.rng.choice(art.FRIENDS)
        # The one with a hostage only up close, and not in the first seconds.
        if near and k > 0.25 and self.rng.random() < 0.3:
            return self.rng.choice(art.MIXED)
        return self.rng.choice(art.FOES)

    # -- main update --------------------------------------------------------
    def update(self, now: float, point: Optional[Point],
               points: Optional[Sequence[Point]] = None) -> None:
        """`points` is every dot on the screen, strongest first; without it
        the one pointer plays for whichever half it is in."""
        seen = list(points) if points is not None else [point] if point is not None else []
        dt = max(0.0, min(0.1, now - self.now))
        if self.state == PLAYING:
            for lane in self.lanes:
                lane.update(now, lane.mine(seen))
        super().update(now, point)
        if self.state == PLAYING and self.TUNE:
            self.sound.music(self.TUNE)
        elif self.state == OVER and not self._settled:
            self._settle()
        elif self.state == OVER and now - self._over_at >= self.SETTLE:
            # Either player's laser will do.
            x0, y0, x1, y1 = self.rematch_box()
            on = next((p for p in seen if x0 <= p[0] <= x1 and y0 <= p[1] <= y1), None)
            if self._hold.update(dt, on, [self.rematch_box()]) is not None:
                self.start(now)

    def _step(self, now: float, dt: float, point: Optional[Point]) -> None:
        for lane in self.lanes:
            for event in lane.events:
                self.sound.blip(event)
            lane.events.clear()

    def _settle(self) -> None:
        self._settled = True
        self._over_at = self.now
        self._hold.reset()
        a, b = (lane.score for lane in self.lanes)
        self.winner = None if a == b else 0 if a > b else 1
        if self.winner is not None:
            self.wins[self.winner] += 1
        self.stats.score = max(a, b)
        self.sound.stop()
        self.sound.blip("win")

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        if self.state != OVER:
            w, h = self.screen_size
            for lane in self.lanes:
                lane.draw_field(canvas)
            cv2.line(canvas, (w // 2, 0), (w // 2, h), overlay.GREY, 2, cv2.LINE_AA)
        super().draw(canvas)

    def _draw_hud(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        left = self.time_left
        colour = overlay.WHITE if left > 10 else overlay.YELLOW
        box = (w // 2 - int(96 * k), int(16 * k), w // 2 + int(96 * k), int(86 * k))
        cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 0), -1)
        cv2.rectangle(canvas, box[:2], box[2:], overlay.GREY, 1)
        overlay.text_fit(canvas, f"{left:04.1f}", (w // 2, (box[1] + box[3]) // 2),
                         150 * k, 36 * k, colour, 3)

        ahead = max(lane.score for lane in self.lanes)
        tied = all(lane.score == ahead for lane in self.lanes)
        for i, lane in enumerate(self.lanes):
            rows = [(lane.name, 0.75 * k, 2, int(40 * k)),
                    (str(lane.score), 1.7 * k, 3, int(98 * k)),
                    (self._tally(lane) + (f"   x{lane.multiplier:.1f}" if lane.combo > 1 else ""),
                     0.6 * k, 1, int(130 * k))]
            for n, (words, scale, thick, y) in enumerate(rows):
                (tw, _), _ = cv2.getTextSize(words, overlay.FONT, scale, thick)
                # Player 1 reads from the left edge, player 2 from the right.
                x = lane.span[0] + int(40 * k) if i == 0 else lane.span[1] - int(40 * k) - tw
                overlay.text(canvas, words, (x, y), scale,
                             lane.colour if n < 2 else overlay.GREY, thick)
                if n == 1 and lane.score == ahead and not tied:
                    cv2.line(canvas, (x, y + int(10 * k)), (x + tw, y + int(10 * k)),
                             overlay.GREEN, max(2, int(4 * k)), cv2.LINE_AA)

    def _tally(self, lane: Lane) -> str:
        """A player's count so far, under the score."""
        return f"збито: {lane.stats.hits}   беззбройних: {lane.stats.decoys}"

    def _summary(self, lane: Lane) -> str:
        """A player's line in the panel at the end."""
        s = lane.stats
        return (f"{lane.name}    збито: {s.hits}    беззбройних: {s.decoys}    "
                f"утекли: {s.escaped}    влучність {lane.accuracy:.0%}    "
                f"реакція {s.reaction:.2f} с")

    def rematch_box(self) -> Tuple[int, int, int, int]:
        w, h = self.screen_size
        return int(w * 0.38), int(h * 0.70), int(w * 0.62), int(h * 0.82)

    def draw_pointer(self, canvas, x: float, y: float, t: float,
                     crosshair: bool = False) -> None:
        """A gun sight in the colour of the lane it is in."""
        lane = self.lanes[int(x >= self.screen_size[0] / 2)]
        overlay.draw_sight(canvas, x, y, lane.colour, self.screen_size[1] / 1080.0,
                           self._kick(lane.shots))

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        a, b = self.lanes
        if self.winner is None:
            head = ("НІЧИЯ", 1.8, overlay.WHITE, 3)
        else:
            lane = self.lanes[self.winner]
            head = (f"ПЕРЕМАГАЄ {lane.name}", 1.8, lane.colour, 3)
        lines = [head, (f"{a.score}  :  {b.score}", 1.6, overlay.WHITE, 3)]
        for lane in self.lanes:
            lines.append((self._summary(lane), 0.7, lane.colour, 1))
        lines.append((f"{self.WINS}    {self.wins[0]} : {self.wins[1]}", 0.9, overlay.GREEN, 2))
        overlay.draw_panel(canvas, lines, self.screen_size[1] * 0.40)

        x0, y0, x1, y1 = self.rematch_box()
        fill = self._hold.fill(0)
        hot = fill > 0
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), overlay.CYAN if hot else overlay.GREY,
                      2 if hot else 1)
        overlay.text_fit(canvas, "РЕВАНШ", ((x0 + x1) // 2, (y0 + y1) // 2),
                         (x1 - x0) * 0.8, (y1 - y0) * 0.36, overlay.WHITE, 2)
        if hot:
            cv2.rectangle(canvas, (x0, y1 - 8), (x0 + int((x1 - x0) * fill), y1),
                          overlay.GREEN, -1)
        overlay.text_centered(canvas, "затримай лазер на кнопці або натисни G"
                                      "      ESC — вихід",
                              int(self.screen_size[1] * 0.87), 0.7, overlay.GREY)
