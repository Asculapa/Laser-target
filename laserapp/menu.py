"""Game chooser: big tiles picked by holding the laser on one, or by number."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2

from . import overlay
from .duel import Duel
from .game import Game, Round
from .mathgames import BalloonMath, NumberHunt
from .silhouette import Silhouette
from .story import Story
from .story_script import LANGUAGES

Factory = Callable[[Tuple[int, int]], Round]


@dataclass
class Entry:
    title: str
    subtitle: str
    options: List[Tuple[str, Factory]]     # one option: starts straight away


GAMES = [
    Entry("Shooting Gallery", "hit the targets", [
        ("targets", lambda size: Game(size)),
        ("silhouette", lambda size: Silhouette(size)),
    ]),
    Entry("Balloon Math", "grades 2-4", [
        (name, lambda size, level=level: BalloonMath(size, level))
        for level, name in BalloonMath.LEVELS.items()
    ]),
    Entry("Number Hunt", "grades 5-9", [
        (f"grades {pack}", lambda size, pack=pack: NumberHunt(size, pack))
        for pack in ("5-6", "7-9")
    ]),
    Entry("Story", "The Last Light of Lantern Rock", [
        (lang.name, lambda size, code=code: Story(size, language=code))
        for code, lang in LANGUAGES.items() if lang.name.isascii() or overlay.UNICODE
    ]),
    Entry("Range Duel", "two players, grades 8-9", [
        ("duel", lambda size: Duel(size)),
    ]),
]


class GameMenu:
    DWELL = 0.9          # seconds the pointer rests on a tile to pick it
    SETTLE = 0.7         # a new page ignores the pointer this long, so the
                         # hold that opened it does not pick from it as well
    GRACE = 0.35         # the hold survives losing the dot for this long: the
                         # detector drops frames, and a hand is never quite still
    SWITCH = 0.12        # another tile takes over only after this long on it,
                         # so one stray detection does not undo a hold

    def __init__(self, screen_size: Tuple[int, int]) -> None:
        self.screen_size = screen_size
        self.page: Optional[int] = None      # None: the games; N: options of game N
        self.now = 0.0
        self._hover: Optional[int] = None
        self._dwell = 0.0
        self._since = 0.0
        self._seen = 0.0                     # when the pointer was last on _hover
        self._other: Optional[int] = None    # a different tile it has moved to
        self._other_since = 0.0

    # -- layout -------------------------------------------------------------
    def _items(self) -> List[Tuple[str, str]]:
        if self.page is None:
            return [(e.title, e.subtitle) for e in GAMES]
        return [(name, "") for name, _ in GAMES[self.page].options]

    def boxes(self) -> List[Tuple[int, int, int, int]]:
        w, h = self.screen_size
        n = len(self._items())
        gap = int(w * 0.03)
        bw = min(int(w * 0.27), (w - gap * (n + 1)) // n)
        bh = int(h * 0.36)
        x0 = (w - (bw * n + gap * (n - 1))) // 2
        y0 = int(h * 0.36)
        return [(x0 + i * (bw + gap), y0, x0 + i * (bw + gap) + bw, y0 + bh)
                for i in range(n)]

    # -- input --------------------------------------------------------------
    def choose(self, i: int) -> Optional[Factory]:
        """Pick tile `i`. Returns the game to start, if that settles it."""
        if not 0 <= i < len(self._items()):
            return None
        self._reset()
        if self.page is not None:
            return GAMES[self.page].options[i][1]
        if len(GAMES[i].options) == 1:
            return GAMES[i].options[0][1]
        self.page = i
        return None

    def back(self) -> bool:
        """Up one page. False if already at the top."""
        if self.page is None:
            return False
        self.page = None
        self._reset()
        return True

    def _reset(self) -> None:
        self._hover, self._other, self._dwell, self._since = None, None, 0.0, self.now

    def _tile_at(self, point: Tuple[float, float]) -> Optional[int]:
        boxes = self.boxes()
        if self._hover is not None and self._hover < len(boxes):
            # The tile being held reaches halfway into the gaps around it, so
            # a dot trembling on its edge stays on it.
            x0, y0, x1, y1 = boxes[self._hover]
            slack = int(self.screen_size[0] * 0.03) // 2
            if x0 - slack <= point[0] <= x1 + slack and y0 - slack <= point[1] <= y1 + slack:
                return self._hover
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
                return i
        return None

    def update(self, now: float, point: Optional[Tuple[float, float]]) -> Optional[Factory]:
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        over = None
        if point is not None and now - self._since >= self.SETTLE:
            over = self._tile_at(point)
        if self._hover is None:
            if over is not None:
                self._hover, self._other, self._dwell, self._seen = over, None, 0.0, now
            return None

        if over == self._hover:
            self._seen, self._other = now, None
        elif over is not None:
            if over != self._other:
                self._other, self._other_since = over, now
            if now - self._other_since >= self.SWITCH:
                self._hover, self._other, self._dwell, self._seen = over, None, 0.0, now
                return None
        if now - self._seen > self.GRACE:
            self._hover, self._other, self._dwell = None, None, 0.0
            return None
        # A short dropout counts towards the hold, but only a pointer that is
        # really on the tile can finish it.
        self._dwell += dt
        if over == self._hover and self._dwell >= self.DWELL:
            return self.choose(over)
        return None

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        w, h = self.screen_size
        title = "Choose a game" if self.page is None else GAMES[self.page].title
        overlay.text_centered(canvas, title, int(h * 0.20), 1.8, overlay.CYAN, 3)
        if self.page is not None:
            overlay.text_centered(canvas, GAMES[self.page].subtitle, int(h * 0.20) + 50,
                                  0.9, overlay.GREY, 2)
        for i, ((x0, y0, x1, y1), (name, sub)) in enumerate(zip(self.boxes(), self._items())):
            hot = i == self._hover
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), overlay.CYAN if hot else overlay.GREY,
                          2 if hot else 1)
            overlay.text(canvas, str(i + 1), (x0 + 18, y0 + 44), 1.1, overlay.YELLOW, 2)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            overlay.text_fit(canvas, name, (cx, cy - (18 if sub else 0)),
                             (x1 - x0) - 50, (y1 - y0) * 0.11, overlay.WHITE, 2)
            if sub:
                overlay.text_fit(canvas, sub, (cx, cy + int((y1 - y0) * 0.17)),
                                 (x1 - x0) - 80, (y1 - y0) * 0.07, overlay.GREY, 1)
            if hot and self._dwell > 0:
                fill = int((x1 - x0) * min(1.0, self._dwell / self.DWELL))
                cv2.rectangle(canvas, (x0, y1 - 10), (x0 + fill, y1), overlay.GREEN, -1)
        overlay.text_centered(
            canvas, f"hold the laser on a tile, or press 1-{len(self._items())}"
                    "      ESC back", int(h * 0.86), 0.7, overlay.GREY)
