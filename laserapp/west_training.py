"""The Wild West, taught: a practice street before the real town.

Four short lessons, one at a time, each said at the top of the screen until
it is done: shoot a bandit who has his gun out; let one of the townsfolk
walk away; reload by shooting the bar at the bottom; and beat a bandit who
does shoot back. Nothing is lost here - a mistake says what went wrong and
the lesson starts again - and no score is kept.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from . import overlay
from . import west_art as art
from .game import HighScores
from .west import STREET, Figure, West

# What each lesson says: the task, then a word of what to watch for.
LESSONS = (
    ("Бандит вихопив револьвер — стріляй у нього!",
     "наведи лазер на бандита й увімкни промінь"),
    ("Це мирний житель — не стріляй",
     "просто дочекайся, поки він піде"),
    ("Набої скінчилися — перезаряди",
     "стріляй у смугу ПЕРЕЗАРЯДКИ внизу екрана"),
    ("А цей стріляє у відповідь — випереди його!",
     "поки горить ґніт над бандитом, ще є час"),
)
WELL_DONE = ("Влучно!", "Правильно — мирних не чіпаємо", "Перезаряджено!", "Ось це швидкість!")


class WestTraining(West):
    PAUSE_BETWEEN = 1.6      # seconds between a lesson done and the next one
    FUSE = 3.0               # the last lesson's bandit: seconds before he fires

    def __init__(self, screen_size: Tuple[int, int], seed: Optional[int] = None) -> None:
        super().__init__(screen_size, players=1, level="sheriff", seed=seed)
        self.scores_name = None                  # practice: no table
        self.HINT = "навчання: чотири короткі уроки, життя не забираються"
        self.lesson = 0
        self._pupil: Optional[Figure] = None     # the figure the lesson is about
        self._active = False                     # a lesson is under way
        self._next_lesson = 0.0

    # -- the street, by lessons ---------------------------------------------
    def _enter(self, stage: int, now: float) -> None:
        super()._enter(0, now)
        self._left = 0                           # nobody comes but who the lesson brings
        self._drops = []
        self._sub = ("", -1.0)
        self._banner = ("НАВЧАННЯ", now + 1.6)
        self.lesson = 0
        self._pupil = None
        self._active = False
        self._next_lesson = now + 1.8

    def _street(self, now: float) -> None:
        if self.lesson >= len(LESSONS):
            if now >= self._next_lesson:
                self.won = True
            return
        if not self._active:
            if now >= self._next_lesson:
                self._begin_lesson(now)
            return
        if self.lesson == 2:
            if self.team[0].ammo == self.level.ammo:
                self._passed(now)
            return
        pupil = self._pupil
        if pupil is None or pupil.dying is None:
            return
        armed_lesson = self.lesson in (0, 3)
        if (pupil.end == "hit") == armed_lesson:
            self._passed(now)
        else:
            # Shot the townsman, or let the bandit fire: say so, and again.
            self._sub = ("Спробуй ще раз" if armed_lesson else
                         "Спробуй ще раз — у мирних не стріляють", now + 2.0)
            self._announce("again")
            self._pupil = None
            self._active = False
            self._next_lesson = now + self.PAUSE_BETWEEN

    def _begin_lesson(self, now: float) -> None:
        self._active = True
        self._announce(f"lesson{self.lesson + 1}")
        if self.lesson == 2:
            self.team[0].ammo = 0                # the gun is empty: reload it
            return
        self.team[0].ammo = self.level.ammo
        armed = self.lesson in (0, 3)
        life = {0: math.inf, 1: self.STAY * 1.6, 3: self.FUSE}[self.lesson]
        self._pupil = self._put(armed, life, now)

    def _put(self, armed: bool, life: float, now: float) -> Figure:
        """Someone on a spot in plain view, near the middle of the street."""
        w = self.screen_size[0]
        spots = [i for i, s in enumerate(self.scene.spots) if s.clip is None] \
            or list(range(len(self.scene.spots)))
        spot = min(spots, key=lambda i: abs(art.place(self.scene, self.scene.spots[i],
                                                      self.screen_size)[0] - w / 2))
        x, foot, height, clip = art.place(self.scene, self.scene.spots[spot], self.screen_size)
        look = self.rng.choice(art.BANDITS if armed else art.TOWNSFOLK)
        t = Figure(x=x, y=foot - height / 2, radius=height / 2, born=now, lifetime=life,
                   look=look, spot=spot, foot=foot, height=height, clip=clip,
                   flip=x > w / 2, side=0)
        self.targets.append(t)
        return t

    def _passed(self, now: float) -> None:
        self._message = (WELL_DONE[self.lesson], now + 1.4)
        self.sound.blip("clear" if self.lesson == len(LESSONS) - 1 else "coin")
        self.lesson += 1
        self._pupil = None
        self._active = False
        self._next_lesson = now + self.PAUSE_BETWEEN
        self._announce("good")
        if self.lesson >= len(LESSONS):
            self._announce("trained")
            self._banner = ("ГОТОВО! ТЕПЕР — СПРАВЖНЄ МІСТО", now + 2.4)
            self._next_lesson = now + 2.6

    def _hurt(self, now: float, words: str, noise: str, side: Optional[int] = None,
              voice: str = "") -> None:
        """As in the town, but the star stays: this is practice."""
        if voice:
            self._announce(voice)
        self._shake = now
        self._message = (words, now + 1.2)
        self.sound.blip(noise)

    def _fire(self, p: int, now: float) -> bool:
        # The reload lesson starts on an empty gun; outside it, never run dry.
        if self.lesson != 2 and self.team[p].ammo <= 1:
            self.team[p].ammo = self.level.ammo
        self._taught[p] = True                   # the lesson says it, not a pop-up
        return super()._fire(p, now)

    # -- drawing ------------------------------------------------------------
    def _draw_hud(self, canvas) -> None:
        w, h = self.screen_size
        k = h / 1080.0
        n = min(self.lesson, len(LESSONS) - 1)
        overlay.text_centered(canvas, f"НАВЧАННЯ   —   урок {n + 1} з {len(LESSONS)}",
                              int(42 * k), 0.75 * k, overlay.WHITE, 2)
        if self.lesson < len(LESSONS) and self.phase == STREET and not self.counting_down:
            task, tip = LESSONS[self.lesson]
            overlay.draw_panel(canvas, [(task, 1.2 * k, overlay.YELLOW, 3),
                                        (tip, 0.8 * k, overlay.WHITE, 1)], int(h * 0.16))
        self._draw_ammo(canvas, 0, self.team[0], self.now)

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        h = self.screen_size[1]
        k = h / 1080.0
        canvas[:] = (canvas * 0.45).astype(canvas.dtype)
        overlay.draw_panel(canvas, [
            ("НАВЧАННЯ ЗАВЕРШЕНО", 1.8 * k, overlay.GREEN, 3),
            ("Тепер обери рівень: помічник шерифа, шериф або маршал", 0.9 * k, overlay.WHITE, 2),
            ("G — пройти ще раз    ESC — вихід", 0.7 * k, overlay.YELLOW, 1),
        ], h * 0.40)
