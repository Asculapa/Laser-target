"""The teacher's settings for the maths games, and the screen they are set on.

Kept in maths.json next to the calibration, and read by each maths game as it
starts: how many sums a round of Balloon Math has, how far the adding and
taking away goes, which times tables come up, and how long a hunt lasts.

The screen is a Round, so the game chooser can open it like any game: rows of
choices picked by holding the laser on them (or with the keys shown), and
ГОТОВО to save and go back. The app goes back to the chooser once `done`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from . import overlay
from .game import HighScores, Hold, Round

Box = Tuple[int, int, int, int]
FILE = "maths.json"


@dataclass
class MathSettings:
    tasks: int = 10                          # sums in a round of Balloon Math
    top1: int = 20                           # + and - for the youngest: up to this
    top2: int = 100                          # + and - for the older ones
    tables: List[int] = field(default_factory=lambda: list(range(2, 10)))
    hunt: int = 60                           # seconds of Number Hunt

    @classmethod
    def load(cls, home: Optional[Path]) -> "MathSettings":
        s = cls()
        if home is None:
            return s
        try:
            data = json.loads((home / FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return s
        for f in fields(cls):
            if f.name in data and isinstance(data[f.name], type(getattr(s, f.name))):
                setattr(s, f.name, data[f.name])
        s.tables = sorted({t for t in s.tables if isinstance(t, int) and 2 <= t <= 9}) \
            or list(range(2, 10))
        return s

    def save(self, home: Optional[Path]) -> None:
        if home is None:
            return
        try:
            (home / FILE).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        except OSError:
            pass

    def describe(self, level: int) -> str:
        """What a Balloon Math level asks for, as the end of a round says it."""
        if level == 3:
            if self.tables == list(range(2, 10)):
                return "таблиця множення"
            return "множення на " + ", ".join(map(str, self.tables))
        return f"+ і −  до {self.top1 if level == 1 else self.top2}"


# Each row: its label, the setting, and its choices as (value, shown as).
ROWS = (
    ("Кульки: задач у грі", "tasks", ((5, "5"), (10, "10"), (15, "15"), (20, "20"))),
    ("Кульки, менші числа: до", "top1", ((10, "10"), (20, "20"))),
    ("Кульки, більші числа: до", "top2", ((50, "50"), (100, "100"))),
    ("Таблиця множення на", "tables", tuple((n, str(n)) for n in range(2, 10))),
    ("Полювання на числа: хвилин", "hunt", ((60, "1"), (90, "1,5"), (120, "2"), (180, "3"))),
)
DONE, RESET = "ГОТОВО", "як було"


class TeacherSettings(Round):
    """Not a game: the settings, on the game's engine so the chooser can open it."""

    needs_sight = False
    back_to_menu = True                      # ESC goes back to the chooser, unsaved
    scores_name = None
    HINT = ""

    def __init__(self, screen_size: Tuple[int, int], seed: Optional[int] = None) -> None:
        super().__init__(screen_size, seed)
        self.settings = MathSettings()
        self.done = False
        self._hold = Hold()
        self._keys, self._boxes = self._layout()

    def start(self, now: float) -> None:
        super().start(now)
        self.settings = MathSettings.load(self.home)
        self.done = False
        self.started = now                       # nothing to count down to

    def _layout(self) -> Tuple[List[tuple], List[Box]]:
        w, h = self.screen_size
        keys, boxes = [], []
        y = int(h * 0.17)
        row_h, gap = int(h * 0.085), int(h * 0.045)
        x0 = int(w * 0.40)
        for r, (_, name, choices) in enumerate(ROWS):
            bw = min(int(w * 0.11), (int(w * 0.96) - x0) // len(choices) - int(w * 0.008))
            for i, (value, _) in enumerate(choices):
                x = x0 + i * (bw + int(w * 0.008))
                keys.append((name, value))
                boxes.append((x, y, x + bw, y + row_h))
            y += row_h + gap
        bw = int(w * 0.2)
        keys.append((RESET, None))
        boxes.append((int(w * 0.04), int(h * 0.84), int(w * 0.04) + bw, int(h * 0.94)))
        keys.append((DONE, None))
        boxes.append((w - int(w * 0.04) - bw, int(h * 0.84), w - int(w * 0.04), int(h * 0.94)))
        return keys, boxes

    def _chosen(self, name: str, value) -> bool:
        current = getattr(self.settings, name)
        return value in current if isinstance(current, list) else current == value

    def press(self, i: int) -> None:
        name, value = self._keys[i]
        if name == DONE:
            self.settings.save(self.home)
            self.done = True
        elif name == RESET:
            self.settings = MathSettings()
        elif name == "tables":
            tables = set(self.settings.tables) ^ {value}
            self.settings.tables = sorted(tables) or self.settings.tables   # never none
        else:
            setattr(self.settings, name, value)

    def key(self, key: int) -> bool:
        if key in (13, 10):
            self.press(len(self._keys) - 1)
            return True
        return False

    def update(self, now: float, point: Optional[Tuple[float, float]]) -> None:
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        picked = self._hold.update(dt, point, self._boxes)
        if picked is not None:
            self.press(picked)

    def draw(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        overlay.text_centered(canvas, "Налаштування вчителя", int(h * 0.08), 1.6 * k,
                              overlay.CYAN, 3)
        overlay.text_centered(canvas, "для математичних ігор — затримай лазер на варіанті",
                              int(h * 0.125), 0.75 * k, overlay.GREY, 1)
        first = 0
        for label, name, choices in ROWS:
            x0, y0, _, y1 = self._boxes[first]
            overlay.text(canvas, label, (int(w * 0.04), (y0 + y1) // 2 + int(12 * k)),
                         0.9 * k, overlay.WHITE, 2)
            first += len(choices)
        shown = [str(v) for _, _, choices in ROWS for _, v in choices] + [RESET, DONE]
        for i, ((name, value), (x0, y0, x1, y1), words) in enumerate(
                zip(self._keys, self._boxes, shown)):
            on = value is not None and self._chosen(name, value)
            fill = self._hold.fill(i)
            back = (60, 50, 20) if fill > 0 else (40, 70, 30) if on or name == DONE \
                else (24, 24, 24)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), back, -1)
            edge = overlay.CYAN if fill > 0 else overlay.GREEN if on or name == DONE \
                else overlay.GREY
            cv2.rectangle(canvas, (x0, y0), (x1, y1), edge, 3 if on else 1)
            overlay.text_fit(canvas, words, ((x0 + x1) // 2, (y0 + y1) // 2),
                             (x1 - x0) * 0.7, (y1 - y0) * 0.4,
                             overlay.WHITE if on or value is None else overlay.GREY, 2)
            if fill > 0:
                cv2.rectangle(canvas, (x0, y1 - 7), (x0 + int((x1 - x0) * fill), y1),
                              overlay.GREEN, -1)
        overlay.text_centered(canvas, "Enter — зберегти    ESC — вийти без збереження",
                              int(h * 0.97), 0.6 * k, overlay.GREY, 1)

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        pass
