"""Game chooser: big tiles picked by holding the laser on one, or by number."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2

from . import overlay
from .duel import Duel
from .game import Game, Round
from .jars import JarRange, QuickDraw
from .mathgames import BalloonMath, NumberHunt
from .silhouette import Silhouette
from .story import Hold, Story
from .story_script import LANGUAGES
from .teacher import TeacherSettings
from .west import West

Factory = Callable[[Tuple[int, int]], Round]
SHOOTING, LEARNING = "Стрільба", "Навчальні ігри"


@dataclass
class Option:
    name: str
    start: Factory
    note: str = ""                  # a line under the name
    row: str = ""                   # options with the same row share a line on the page,
                                    # under this heading


@dataclass
class Entry:
    title: str
    subtitle: str
    group: str                      # which row of the first page it is on
    options: List[Option]           # one option: starts straight away


WEST_LEVELS = (("deputy", "Помічник шерифа", "7 життів, без перезаряджання"),
               ("sheriff", "Шериф", "5 життів, шестизарядний револьвер"),
               ("marshal", "Маршал", "4 життя, спритні бандити"))

# In the order they are numbered: the first page reads row by row.
GAMES = [
    Entry("Тир", "влучай у мішені", SHOOTING, [
        Option("мішені", lambda size: Game(size), "хвилина стрільби"),
        Option("силует", lambda size: Silhouette(size), "десять прицільних пострілів"),
    ]),
    Entry("Дикий Захід", "1 або 2 гравці, разом", SHOOTING, [
        Option(name, lambda size, n=n, level=level: West(size, players=n, level=level),
               note, "1 гравець" if n == 1 else f"{n} гравці")
        for n in (1, 2) for level, name, note in WEST_LEVELS
    ]),
    Entry("Дуель у тирі", "двоє гравців, 8–9 класи", SHOOTING, [
        Option("дуель", lambda size: Duel(size)),
    ]),
    Entry("Стрільба по банках", "двоє гравців", SHOOTING, [
        Option("банки", lambda size: JarRange(size), "банки трьох розмірів, хвилина"),
        Option("хто швидший", lambda size: QuickDraw(size), "хто перший розіб'є банку"),
    ]),
    Entry("Математичні кульки", "2–4 класи", LEARNING, [
        Option(name, lambda size, level=level: BalloonMath(size, level))
        for level, name in BalloonMath.LEVELS.items()
    ]),
    Entry("Полювання на числа", "5–9 класи", LEARNING, [
        Option(f"{pack.replace('-', '–')} класи", lambda size, pack=pack: NumberHunt(size, pack))
        for pack in ("5-6", "7-9")
    ]),
    Entry("Історія", LANGUAGES["uk"].title, LEARNING, [
        Option(lang.name, lambda size, code=code: Story(size, language=code))
        for code, lang in LANGUAGES.items() if lang.name.isascii() or overlay.UNICODE
    ]),
    Entry("Для вчителя", "налаштування математичних ігор", LEARNING, [
        Option("налаштування", lambda size: TeacherSettings(size)),
    ]),
]


def entry(title: str) -> Entry:
    return next(e for e in GAMES if e.title == title)


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
    MOST = 4             # tiles to a row at most: more, and they get too narrow to read

    def _items(self) -> List[Tuple[str, str]]:
        if self.page is None:
            return [(e.title, e.subtitle) for e in GAMES]
        return [(o.name[:1].upper() + o.name[1:], o.note) for o in GAMES[self.page].options]

    def rows(self) -> List[Tuple[str, List[int]]]:
        """The page as rows of tiles: a heading, and the tiles under it, in
        the order they are numbered."""
        if self.page is None:
            keys = [e.group for e in GAMES]
        else:
            keys = [o.row for o in GAMES[self.page].options]
        rows: List[Tuple[str, List[int]]] = []
        for i, key in enumerate(keys):
            if rows and rows[-1][0] == key and len(rows[-1][1]) < self.MOST:
                rows[-1][1].append(i)
            else:
                rows.append((key, [i]))
        return rows

    def _row_tops(self) -> Tuple[List[int], int, int]:
        """Where each row starts, the height of a tile, and of a heading."""
        w, h = self.screen_size
        rows = self.rows()
        heading = int(h * 0.05) if any(key for key, _ in rows) else 0
        gap = int(h * 0.035)
        top, bottom = int(h * 0.22), int(h * 0.85)
        tile = min(int(h * 0.30),
                   (bottom - top - len(rows) * heading - (len(rows) - 1) * gap) // len(rows))
        used = len(rows) * (heading + tile) + (len(rows) - 1) * gap
        y = top + (bottom - top - used) // 2
        tops = []
        for _ in rows:
            tops.append(y + heading)
            y += heading + tile + gap
        return tops, tile, heading

    def boxes(self) -> List[Tuple[int, int, int, int]]:
        w, h = self.screen_size
        rows = self.rows()
        tops, tile, _ = self._row_tops()
        widest = max(len(ids) for _, ids in rows)
        gap = int(w * 0.025)
        bw = min(int(w * 0.25), (int(w * 0.9) - gap * (widest - 1)) // widest)
        boxes = [(0, 0, 0, 0)] * len(self._items())
        for (_, ids), y0 in zip(rows, tops):
            x0 = (w - (bw * len(ids) + gap * (len(ids) - 1))) // 2
            for n, i in enumerate(ids):
                x = x0 + n * (bw + gap)
                boxes[i] = (x, y0, x + bw, y0 + tile)
        return boxes

    # -- input --------------------------------------------------------------
    def choose(self, i: int) -> Optional[Factory]:
        """Pick tile `i`. Returns the game to start, if that settles it."""
        if not 0 <= i < len(self._items()):
            return None
        self._reset()
        if self.page is not None:
            return GAMES[self.page].options[i].start
        if len(GAMES[i].options) == 1:
            return GAMES[i].options[0].start
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
        k = h / 1080.0
        title = "Обери гру" if self.page is None else GAMES[self.page].title
        overlay.text_centered(canvas, title, int(h * 0.11), 1.8 * k, overlay.CYAN, 3)
        if self.page is not None:
            overlay.text_centered(canvas, GAMES[self.page].subtitle, int(h * 0.17),
                                  0.9 * k, overlay.GREY, 2)
        boxes = self.boxes()
        tops, _, heading = self._row_tops()
        for (key, ids), top in zip(self.rows(), tops):
            if key:
                # The row's heading, over its first tile, and a rule along the row.
                x0, x1 = boxes[ids[0]][0], boxes[ids[-1]][2]
                y = top - int(heading * 0.35)
                overlay.text(canvas, key.upper(), (x0, y), 0.75 * k, overlay.GREY, 2)
                (tw, _), _ = cv2.getTextSize(key.upper(), overlay.FONT, 0.75 * k, 2)
                cv2.line(canvas, (x0 + tw + int(16 * k), y - int(8 * k)),
                         (x1, y - int(8 * k)), overlay.DIM, 1, cv2.LINE_AA)
        for i, ((x0, y0, x1, y1), (name, sub)) in enumerate(zip(boxes, self._items())):
            hot = i == self._hover
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), overlay.CYAN if hot else overlay.GREY,
                          2 if hot else 1)
            th = y1 - y0
            overlay.text(canvas, str(i + 1), (x0 + int(14 * k), y0 + int(38 * k)), 1.0 * k,
                         overlay.YELLOW, 2)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            overlay.text_fit(canvas, name, (cx, cy - (int(th * 0.08) if sub else 0)),
                             (x1 - x0) * 0.82, th * 0.17, overlay.WHITE, 2)
            if sub:
                overlay.text_fit(canvas, sub, (cx, cy + int(th * 0.22)),
                                 (x1 - x0) * 0.88, th * 0.11, overlay.GREY, 1)
            if hot and self._dwell > 0:
                fill = int((x1 - x0) * min(1.0, self._dwell / self.DWELL))
                cv2.rectangle(canvas, (x0, y1 - 10), (x0 + fill, y1), overlay.GREEN, -1)
        overlay.text_centered(
            canvas, f"затримай лазер на картці або натисни 1–{len(self._items())}"
                    "      ESC — назад", int(h * 0.92), 0.7 * k, overlay.GREY)


AGAIN, OTHER = "again", "other"


class EndButtons:
    """PLAY AGAIN and OTHER GAMES at the end of a game, picked with the laser:
    held on one for a moment, like a tile of the chooser. In the bottom
    corners, clear of what the games write at the end."""

    SETTLE = 1.5         # the buttons ignore the pointer this long: when a game
                         # ends, the players are still shooting

    def __init__(self, screen_size: Tuple[int, int], now: float) -> None:
        self.screen_size = screen_size
        self.since = now
        self.now = now
        self._hold = Hold()

    def boxes(self) -> List[Tuple[int, int, int, int]]:
        w, h = self.screen_size
        bw, bh = int(w * 0.2), int(h * 0.12)
        y0 = int(h * 0.84)
        return [(int(w * 0.04), y0, int(w * 0.04) + bw, y0 + bh),
                (w - int(w * 0.04) - bw, y0, w - int(w * 0.04), y0 + bh)]

    def update(self, now: float, points) -> Optional[str]:
        """AGAIN or OTHER once a hold on one completes, else None. Any of
        the dots on the screen will do."""
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        if now - self.since < self.SETTLE:
            return None
        boxes = self.boxes()
        on = next((p for p in points
                   if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in boxes)),
                  None)
        picked = self._hold.update(dt, on, boxes)
        return None if picked is None else (AGAIN, OTHER)[picked]

    def draw(self, canvas) -> None:
        ready = self.now - self.since >= self.SETTLE
        for i, ((x0, y0, x1, y1), words) in enumerate(zip(self.boxes(),
                                                         ("ЩЕ РАЗ", "ІНШІ ІГРИ"))):
            fill = self._hold.fill(i)
            hot = fill > 0
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1),
                          overlay.CYAN if hot else overlay.GREY if ready else overlay.DIM,
                          2 if hot else 1)
            # Both lines kept well inside the frame, and the hint clear of
            # the progress bar that fills along the bottom edge.
            bh = y1 - y0
            overlay.text_fit(canvas, words, ((x0 + x1) // 2, y0 + int(bh * 0.36)),
                             (x1 - x0) * 0.72, bh * 0.30,
                             overlay.WHITE if ready else overlay.DIM, 2)
            overlay.text_fit(canvas, "затримай лазер тут", ((x0 + x1) // 2, y0 + int(bh * 0.70)),
                             (x1 - x0) * 0.66, bh * 0.13,
                             overlay.GREY if ready else overlay.DIM, 1)
            if hot:
                cv2.rectangle(canvas, (x0, y1 - 8), (x0 + int((x1 - x0) * fill), y1),
                              overlay.GREEN, -1)
