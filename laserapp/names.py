"""A name for a new high score, typed with the laser on a Ukrainian keyboard.

The keyboard of the computer cannot help here: OpenCV hands over one byte a
key, so Cyrillic never arrives. So the letters are on the screen, and a key
is pressed by holding the dot on it, a little quicker than a menu tile -
there are more of them to press. Held on, a key goes on typing, for a
doubled letter. From the keyboard, Backspace rubs out, Enter is done, ESC
leaves the score without a name, and Latin letters and digits type as well.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import cv2

from . import overlay
from .game import Hold

Box = Tuple[int, int, int, int]

ROWS = ("АБВГҐДЕЄЖЗИ", "ІЇЙКЛМНОПРС", "ТУФХЦЧШЩЬЮЯ")
SPACE, RUB, DONE, SKIP, AGAIN = "пробіл", "стерти", "ГОТОВО", "без імені", "again"
MOST = 14                    # letters in a name

MONTHS = ("січ", "лют", "бер", "квіт", "трав", "черв",
          "лип", "серп", "вер", "жовт", "лист", "груд")


def today() -> str:
    """When a score was made, as the tables show it: "9 жовт 14:05"."""
    t = time.localtime()
    return f"{t.tm_mday} {MONTHS[t.tm_mon - 1]} {t.tm_hour:02d}:{t.tm_min:02d}"


class _KeyHold(Hold):
    DWELL = 0.65


class NameEntry:
    SETTLE = 1.2             # the keys ignore the dot this long: the game has just
                             # ended, and the players are still shooting

    def __init__(self, screen_size: Tuple[int, int], now: float, title: str,
                 last: str = "") -> None:
        self.screen_size = screen_size
        self.title = title
        self.last = last                 # the name typed last time, offered as one key
        self.name = ""
        self.since = self.now = now
        self.result: Optional[str] = None    # the name, "" for none; None while typing
        self._hold = _KeyHold()
        self._keys, self._boxes = self._layout()

    # -- layout -------------------------------------------------------------
    def _layout(self) -> Tuple[List[str], List[Box]]:
        w, h = self.screen_size
        keys: List[str] = []
        boxes: List[Box] = []
        cols = len(ROWS[0])
        gap = int(w * 0.008)
        kw = (int(w * 0.92) - gap * (cols - 1)) // cols
        kh = int(h * 0.11)
        x0 = (w - (kw * cols + gap * (cols - 1))) // 2
        y = int(h * 0.36)
        for row in ROWS:
            for i, letter in enumerate(row):
                x = x0 + i * (kw + gap)
                keys.append(letter)
                boxes.append((x, y, x + kw, y + kh))
            y += kh + gap
        # The last row: wide keys.
        wide = [("'", 1), ("-", 1), (SPACE, 2), (RUB, 2), (SKIP, 2), (DONE, 3)]
        x = x0
        for key, span in wide:
            width = kw * span + gap * (span - 1)
            keys.append(key)
            boxes.append((x, y, x + width, y + kh))
            x += width + gap
        if self.last:
            # Above the letters, beside the name: the same player again.
            keys.append(AGAIN)
            boxes.append((int(w * 0.62), int(h * 0.20), int(w * 0.96), int(h * 0.30)))
        return keys, boxes

    # -- input --------------------------------------------------------------
    def press(self, key: str) -> None:
        if key == DONE:
            self.result = self.name.strip()
        elif key == SKIP:
            self.result = ""
        elif key == RUB:
            self.name = self.name[:-1]
        elif key == AGAIN:
            self.name = self.last
        elif len(self.name) < MOST:
            letter = " " if key == SPACE else key
            if letter == " " and (not self.name or self.name.endswith(" ")):
                return
            # A capital to start each word, small letters after.
            first = not self.name or self.name[-1] in " -'"
            self.name += letter.upper() if first else letter.lower()

    def key(self, code: int) -> bool:
        """A key on the computer's keyboard. True if it was taken."""
        if code in (13, 10):
            self.press(DONE)
        elif code == 27:
            self.press(SKIP)
        elif code in (8, 127):
            self.press(RUB)
        elif code == 32:
            self.press(SPACE)
        elif 32 < code < 127 and (chr(code).isalnum() or chr(code) in "-'"):
            self.press(chr(code))
        else:
            return False
        return True

    def update(self, now: float, points: Sequence[Tuple[float, float]]) -> Optional[str]:
        """The name once it is settled, else None. Any of the dots will do."""
        dt = max(0.0, min(0.1, now - self.now))
        self.now = now
        if now - self.since >= self.SETTLE and self.result is None:
            on = next((p for p in points
                       if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1
                              for x0, y0, x1, y1 in self._boxes)), None)
            picked = self._hold.update(dt, on, self._boxes)
            if picked is not None:
                self.press(self._keys[picked])
        return self.result

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        canvas[:] = (canvas * 0.25).astype(canvas.dtype)
        overlay.text_centered(canvas, self.title, int(h * 0.09), 1.5 * k, overlay.GREEN, 3)
        overlay.text_centered(canvas, "Як тебе звати? Затримай лазер на літері",
                              int(h * 0.15), 0.8 * k, overlay.GREY, 1)
        # The name so far, with a cursor.
        x0, y0, x1, y1 = int(w * 0.04), int(h * 0.20), int(w * 0.58), int(h * 0.30)
        if not self.last:
            x1 = int(w * 0.96)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (10, 10, 10), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), overlay.CYAN, 2)
        shown = self.name + ("_" if int(self.now * 2) % 2 == 0 else " ")
        overlay.text(canvas, shown, (x0 + int(24 * k), (y0 + y1) // 2 + int(18 * k)),
                     1.5 * k, overlay.WHITE, 3)
        ready = self.now - self.since >= self.SETTLE
        for i, (key, (bx0, by0, bx1, by1)) in enumerate(zip(self._keys, self._boxes)):
            fill = self._hold.fill(i)
            hot = fill > 0
            special = key in (DONE, SKIP, RUB, SPACE, AGAIN)
            back = (60, 50, 20) if hot else (40, 60, 30) if key == DONE else (24, 24, 24)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), back, -1)
            edge = overlay.CYAN if hot else overlay.GREEN if key == DONE else \
                overlay.GREY if ready else overlay.DIM
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), edge, 2 if hot else 1)
            label = f"ще раз: {self.last}" if key == AGAIN else key
            bw, bh = bx1 - bx0, by1 - by0
            overlay.text_fit(canvas, label, ((bx0 + bx1) // 2, (by0 + by1) // 2),
                             bw * (0.75 if special else 0.6), bh * (0.32 if special else 0.45),
                             overlay.WHITE if ready else overlay.DIM, 2)
            if hot:
                cv2.rectangle(canvas, (bx0, by1 - 7), (bx0 + int(bw * fill), by1),
                              overlay.GREEN, -1)
        overlay.text_centered(canvas, "Enter — готово    Backspace — стерти    ESC — без імені",
                              int(h * 0.95), 0.65 * k, overlay.GREY, 1)
