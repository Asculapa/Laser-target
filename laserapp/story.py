"""The story game: "The Last Light of Lantern Rock".

Five chapters, each a small game of its own (story_levels.py), with the tale
told between them: pictures (story_art.py), words (story_script.py) and, where
the recordings are there, voices (sound.py). While a chapter is being played
its own piece of music runs under it; the scenes are left to the voices.

To the app this is one more Round. Inside, it is a string of stages - the
chapter list, a scene before each chapter, the chapter itself, how it went, a
scene after - and it keeps what has been reached in story.json, so the story
can be put down and picked up again.

Nothing here needs the keyboard: scenes move on by themselves as each line
ends, and everything that can be chosen can be chosen by holding the pointer
on it. SPACE and ENTER are there for whoever is sitting at it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import overlay, sound
from . import story_art as art
from . import story_script as script
from .game import OVER, PAUSED, PLAYING, HighScores, Hold, Round
from .mathgames import _star
from .story_levels import Chapter, Constellations, Ships, Siege, Sparks, Storm

ASSETS = Path(__file__).parent / "assets" / "story"
LEVELS = (Sparks, Siege, Ships, Constellations, Storm)

TITLE, INTRO, PLAY, RESULT, RETRY, OUTRO, END = \
    "title", "intro", "play", "result", "retry", "outro", "end"

Box = Tuple[int, int, int, int]

FRAME = {script.NARRATOR: overlay.GREY, script.MAREN: overlay.CYAN,
         script.PIP: art.SPARK, script.GLOAM: art.FOG}
FACE = {script.MAREN: art.maren, script.PIP: art.pip, script.GLOAM: art.gloam}


class Progress:
    """How far the story has got, and the best result in each chapter."""

    def __init__(self, path: Optional[Path], chapters: int) -> None:
        self.path = path
        self.chapters = chapters
        self.unlocked = 1                   # chapters that may be played
        self.best: Dict[str, dict] = {}     # by chapter number
        self.finished = False
        if path is None:
            return
        try:
            data = json.loads(path.read_text())
            self.unlocked = min(chapters, max(1, int(data.get("unlocked", 1))))
            self.best = {k: v for k, v in data.get("best", {}).items() if isinstance(v, dict)}
            self.finished = bool(data.get("finished"))
        except Exception:
            pass                            # no story yet, or a file we cannot read

    def record(self, chapter: int, score: int, stars: int) -> bool:
        """Chapter `chapter` (from 0) was won. True if that beat the best."""
        self.unlocked = min(self.chapters, max(self.unlocked, chapter + 2))
        old = self.best.get(str(chapter + 1), {})
        better = score > old.get("score", -1)
        self.best[str(chapter + 1)] = {"score": max(score, old.get("score", 0)),
                                       "stars": max(stars, old.get("stars", 0))}
        self.save()
        return better

    def stars(self, chapter: int) -> int:
        return int(self.best.get(str(chapter + 1), {}).get("stars", 0))

    def score(self, chapter: int) -> int:
        return int(self.best.get(str(chapter + 1), {}).get("score", 0))

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.write_text(json.dumps(
                {"unlocked": self.unlocked, "finished": self.finished, "best": self.best},
                indent=2))
        except OSError:
            pass


class Story(Round):
    COUNTDOWN = 0.0
    LINE_GAP = 0.7           # the pause after a line, before the next
    READ = 14.0              # characters a second, for a line with no recording
    SETTLE = 0.7             # a new stage ignores the pointer this long, so the
                             # hold that opened it does not press something in it
    RESULT_TIME = 9.0        # a result moves on by itself after this long
    RETRY_TIME = 7.0

    def __init__(self, screen_size: Tuple[int, int], seed: Optional[int] = None,
                 levels: Sequence[type] = LEVELS, language: str = "uk") -> None:
        super().__init__(screen_size, seed)
        self.levels = tuple(levels)
        self.words = script.LANGUAGES[language]
        self.voice = sound.Player(ASSETS / self.words.code)
        self.effects = sound.Player(ASSETS)
        self.music = sound.Player(ASSETS / "music")     # c1.wav for chapter 1, and so on
        self._voice_held = False             # the voice is paused, not cut off
        self.art = art.Backdrop(screen_size)
        self.progress = Progress(None, len(self.levels))
        self.stage = TITLE
        self.chapter = 0
        self.level: Optional[Chapter] = None
        self.clock = 0.0                     # seconds of story, pauses left out
        self.scene = "lit"
        self.lines: Tuple[script.Line, ...] = ()
        self.line_index = 0
        self._line_at = 0.0
        self._line_len = 0.0
        self._stage_at = 0.0
        self._hold = Hold()
        self._new_best = False
        self._wrapped: Dict[Tuple[str, int], List[str]] = {}

    # -- round flow ---------------------------------------------------------
    def start(self, now: float) -> None:
        super().start(now)
        path = self.home / "story.json" if self.home is not None else None
        self.progress = Progress(path, len(self.levels))
        self.level = None
        self.clock = 0.0
        self.lines = ()
        self._enter(TITLE)
        self.scene = "lit"
        self._say(self.words.title_line)

    def close(self) -> None:
        self.voice.stop()
        self.music.stop()

    @property
    def needs_sight(self) -> bool:
        # Only a chapter is lost by not seeing the laser. A scene tells itself,
        # and pausing it for a camera that stutters would only interrupt it.
        return self.stage == PLAY

    def toggle_pause(self, now: float) -> None:
        self.now = max(self.now, now)
        playing = self.stage == PLAY and self.level is not None
        if self.state == PLAYING:
            self.state = PAUSED
            # The line stops where it is, to go on from there - or, where the
            # player cannot do that, is cut off and said again afterwards.
            self._voice_held = self.voice.hold()
            if not self._voice_held:
                self.voice.stop()
            if playing:
                self.level.toggle_pause(now)
                self.music.hold()
        elif self.state == PAUSED:
            self.state = PLAYING
            if playing:
                self.level.toggle_pause(now)
                self.music.release()
            elif self._voice_held:
                self.voice.release()
            elif self.stage in (INTRO, OUTRO) and \
                    self.clock - self._line_at < self._line_len:
                self._say(self.line)
            self._voice_held = False

    def _enter(self, stage: str) -> None:
        self.stage = stage
        self._stage_at = self.clock
        self._hold.reset()

    @property
    def line(self) -> script.Line:
        return self.lines[self.line_index]

    @property
    def total(self) -> int:
        return sum(self.progress.score(i) for i in range(len(self.levels)))

    # -- telling ------------------------------------------------------------
    def _say(self, line: script.Line) -> None:
        length = self.voice.say(line.id, line.text)
        self._line_len = length if length is not None else 1.5 + len(line.text) / self.READ
        self._line_at = self.clock
        if line.scene:
            self.scene = line.scene

    def _tell(self, lines: Tuple[script.Line, ...], stage: str) -> None:
        self.lines, self.line_index = lines, 0
        self._enter(stage)
        self._say(lines[0])

    def _next_line(self) -> None:
        if self.line_index + 1 < len(self.lines):
            self.line_index += 1
            self._say(self.line)
        else:
            self._told()

    def _told(self) -> None:
        """The scene is over: on to whatever follows it."""
        self.voice.stop()
        if self.stage == INTRO:
            self._play()
        elif self.chapter + 1 < len(self.levels):
            self.open(self.chapter + 1)
        else:
            self.progress.finished = True
            self.progress.save()
            self.stats.score = self.total
            self._enter(END)
            self.scene = "dawn"
            self.state = OVER

    def open(self, chapter: int) -> None:
        """Begin chapter `chapter` (from 0), with the scene that leads into it."""
        self.chapter = chapter
        self._tell(self.words.chapters[chapter].intro, INTRO)

    def _play(self) -> None:
        self.voice.stop()
        self.level = self.levels[self.chapter](
            self.screen_size, seed=self.rng.randrange(1 << 30), language=self.words)
        self.level.backdrop = self.art
        self.level.start(self.now)
        self._enter(PLAY)

    def _played(self) -> None:
        self.music.stop()
        level = self.level
        if level.won:
            self._new_best = self.progress.record(self.chapter, level.stats.score, level.stars)
            self._enter(RESULT)
        else:
            self._enter(RETRY)
            self.lines, self.line_index = (self.words.chapters[self.chapter].fail,), 0
            self._say(self.line)

    # -- layout -------------------------------------------------------------
    def title_boxes(self) -> List[Box]:
        w, h = self.screen_size
        n = len(self.levels)
        gap = int(w * 0.02)
        bw = (w - gap * (n + 1)) // n
        return [(gap + i * (bw + gap), int(h * 0.52), gap + i * (bw + gap) + bw, int(h * 0.80))
                for i in range(n)]

    def dialogue_box(self) -> Box:
        w, h = self.screen_size
        return int(w * 0.05), int(h * 0.70), int(w * 0.95), int(h * 0.935)

    def skip_box(self) -> Box:
        w, h = self.screen_size
        return int(w * 0.86), int(h * 0.04), int(w * 0.97), int(h * 0.115)

    def button_box(self) -> Box:
        w, h = self.screen_size
        return int(w * 0.38), int(h * 0.76), int(w * 0.62), int(h * 0.87)

    # -- input --------------------------------------------------------------
    def key(self, key: int) -> bool:
        if self.state != PLAYING or self.stage == PLAY:
            return False
        ch = chr(key) if 32 <= key < 127 else ""
        if self.stage == TITLE:
            if ch.isdigit() and 1 <= int(ch) <= self.progress.unlocked:
                self.open(int(ch) - 1)
                return True
            if key in (32, 13, 10):
                self.open(self.progress.unlocked - 1)
                return True
            return False
        if key == 32:
            self._advance()
            return True
        if key in (13, 10):
            if self.stage in (INTRO, OUTRO):
                self._told()
            else:
                self._advance()
            return True
        return False

    def _advance(self) -> None:
        """What SPACE, or a held button, does at this point."""
        if self.stage in (INTRO, OUTRO):
            self._next_line()
        elif self.stage == RESULT:
            self._tell(self.words.chapters[self.chapter].outro, OUTRO)
        elif self.stage == RETRY:
            self._play()

    # -- main update --------------------------------------------------------
    def update(self, now: float, point: Optional[Tuple[float, float]]) -> None:
        dt = max(0.0, min(0.25, now - self.now))
        self.now = now
        if self.state != PLAYING:
            self._beam(now, point)
            return
        self.clock += dt

        if self.stage == PLAY:
            level = self.level
            level.update(now, point)
            for event in level.events:
                if event != "fail":
                    self.effects.blip(event)
            level.events.clear()
            if level.state == OVER:
                self._played()
            else:
                self.music.music(f"c{self.chapter + 1}")
            return

        lit = self._beam(now, point)
        since = self.clock - self._stage_at
        if since < self.SETTLE:
            point, lit = None, False
        if self.stage == TITLE:
            pick = self._hold.update(dt, point, self.title_boxes()[:self.progress.unlocked])
            if pick is not None:
                self.effects.blip("tick")
                self.open(pick)
        elif self.stage in (INTRO, OUTRO):
            if self._hold.update(dt, point, [self.skip_box()]) is not None:
                self.effects.blip("tick")
                self._told()
                return
            spoken = self.clock - self._line_at
            x0, y0, x1, y1 = self.dialogue_box()
            # The beam switched on over the words: the reader wants the next.
            flashed = lit and spoken > 0.5 and x0 <= point[0] <= x1 and y0 <= point[1] <= y1
            if flashed or spoken >= self._line_len + self.LINE_GAP:
                self._next_line()
        else:
            limit = self.RESULT_TIME if self.stage == RESULT else self._line_len + self.RETRY_TIME
            if self._hold.update(dt, point, [self.button_box()]) is not None or since >= limit:
                self.effects.blip("tick")
                self._advance()

    # -- drawing ------------------------------------------------------------
    def draw(self, canvas) -> None:
        if self.stage == PLAY:
            self.level.draw(canvas)
            return
        t = self.clock
        if self.stage == TITLE:
            self.art.draw(canvas, "lit", t)
            self._draw_title(canvas)
        elif self.stage in (INTRO, OUTRO):
            self.art.draw(canvas, self.scene, t)
            if self.stage == INTRO and self.line_index == 0:
                h = self.screen_size[1]
                overlay.text_centered(canvas, self._heading(), int(h * 0.12), 1.4 * h / 1080,
                                      art.SPARK, 3)
            self._draw_dialogue(canvas, self.line)
            self._draw_button(canvas, self.skip_box(), self.words.say("skip"),
                              self._hold.fill(0), small=True)
        elif self.stage == RESULT:
            self.art.draw(canvas, self.level.SCENE, t)
            self._draw_result(canvas)
        elif self.stage == RETRY:
            self.art.draw(canvas, "dark", t)
            self._draw_retry(canvas)
        else:
            self.art.draw(canvas, "dawn", self.now)
        if self.state == PAUSED:
            overlay.draw_panel(canvas, [(self.words.say("paused"), 1.6, overlay.CYAN, 3),
                                        (self.words.say("resume"), 0.7, overlay.GREY, 1)],
                               self.screen_size[1] / 2)

    def _heading(self) -> str:
        return self.words.say("chapter", n=self.chapter + 1,
                              title=self.words.chapters[self.chapter].title)

    def draw_pointer(self, canvas, x: float, y: float, t: float,
                     crosshair: bool = False) -> None:
        # The keeper's beam: a spot of the lamp's light, not a gun sight.
        overlay.draw_glow(canvas, x, y, art.SPARK, self.screen_size[1] / 1080.0, t)

    def _draw_button(self, canvas, box: Box, label: str, fill: float,
                     small: bool = False) -> None:
        x0, y0, x1, y1 = box
        hot = fill > 0
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (24, 24, 24), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), overlay.CYAN if hot else overlay.GREY,
                      2 if hot else 1)
        overlay.text_fit(canvas, label, ((x0 + x1) // 2, (y0 + y1) // 2),
                         (x1 - x0) * 0.7, (y1 - y0) * (0.3 if small else 0.36),
                         overlay.GREY if small and not hot else overlay.WHITE, 2)
        if hot:
            cv2.rectangle(canvas, (x0, y1 - 8), (x0 + int((x1 - x0) * fill), y1), overlay.GREEN, -1)

    def _draw_title(self, canvas) -> None:
        w, h = self.screen_size
        say = self.words.say
        overlay.text_fit(canvas, self.words.title, (int(w * 0.34), int(h * 0.20)),
                         w * 0.58, h * 0.06, art.SPARK, 3)
        overlay.text_fit(canvas, say("subtitle"),
                         (int(w * 0.34), int(h * 0.30)), w * 0.4, h * 0.03, overlay.GREY, 1)
        newest = self.progress.unlocked - 1
        for i, (x0, y0, x1, y1) in enumerate(self.title_boxes()):
            open_ = i < self.progress.unlocked
            fill = self._hold.fill(i) if open_ else 0.0
            hot = fill > 0
            edge = overlay.CYAN if hot or (i == newest and not self.progress.finished) \
                else overlay.GREY if open_ else overlay.DIM
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 50, 20) if hot else (20, 18, 12), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), edge, 2 if hot else 1)
            bw, bh = x1 - x0, y1 - y0
            overlay.text(canvas, str(i + 1), (x0 + 16, y0 + 44), 1.1,
                         overlay.YELLOW if open_ else overlay.DIM, 2)
            words = self.words.chapters[i].title.split()
            half = (len(words) + 1) // 2
            for k, part in enumerate((" ".join(words[:half]), " ".join(words[half:]))):
                overlay.text_fit(canvas, part, ((x0 + x1) // 2, int(y0 + bh * (0.36 + 0.16 * k))),
                                 bw * 0.82, bh * 0.085, overlay.WHITE if open_ else overlay.DIM, 2)
            cy = int(y0 + bh * 0.76)
            if not open_:
                overlay.text_fit(canvas, say("locked"), ((x0 + x1) // 2, cy), bw * 0.5, bh * 0.07,
                                 overlay.DIM, 1)
            elif self.progress.stars(i):
                for k in range(3):
                    _star(canvas, ((x0 + x1) / 2 + (k - 1) * bh * 0.2, cy), bh * 0.075,
                          filled=k < self.progress.stars(i))
                overlay.text_fit(canvas, str(self.progress.score(i)),
                                 ((x0 + x1) // 2, int(y0 + bh * 0.90)), bw * 0.5, bh * 0.05,
                                 overlay.GREY, 1)
            else:
                overlay.text_fit(canvas, say("begin"), ((x0 + x1) // 2, cy), bw * 0.5, bh * 0.07,
                                 overlay.CYAN, 2)
            if hot:
                cv2.rectangle(canvas, (x0, y1 - 10), (x0 + int(bw * fill), y1), overlay.GREEN, -1)
        keys = "1" if self.progress.unlocked == 1 else f"1-{self.progress.unlocked}"
        overlay.text_centered(canvas, say("choose", keys=keys), int(h * 0.88), 0.7, overlay.GREY)

    def _wrap(self, line: script.Line, width: int, scale: float) -> List[str]:
        key = (line.id, width)
        if key not in self._wrapped:
            rows, row = [], ""
            for word in line.text.split():
                trial = f"{row} {word}".strip()
                if row and cv2.getTextSize(trial, overlay.FONT, scale, 2)[0][0] > width:
                    rows.append(row)
                    row = word
                else:
                    row = trial
            self._wrapped[key] = rows + [row]
        return self._wrapped[key]

    def _draw_dialogue(self, canvas, line: script.Line, box: Optional[Box] = None,
                       footer: bool = True) -> None:
        h = self.screen_size[1]
        x0, y0, x1, y1 = box or self.dialogue_box()
        region = canvas[y0:y1, x0:x1]
        cv2.addWeighted(region, 0.18, np.full_like(region, (24, 18, 8)), 0.82, 0, region)
        colour = FRAME[line.who]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)

        k = h / 1080.0
        left = x0 + int(36 * k)
        face = FACE.get(line.who)
        if face is not None:
            r = (y1 - y0) * 0.36
            face(canvas, x0 + r * 1.35, (y0 + y1) / 2, r, self.clock)
            left = int(x0 + r * 2.7)
        top = y0 + int(46 * k)
        if self.words.names[line.who]:
            overlay.text(canvas, self.words.names[line.who], (left, top), 0.8 * k, colour, 2)
            top += int(46 * k)
        else:
            top += int(10 * k)

        scale = 0.95 * k
        rows = self._wrap(line, x1 - left - int(40 * k), scale)
        # The words appear as they are spoken.
        spoken = self.clock - self._line_at
        show = int(len(line.text) * min(1.0, spoken / max(0.3, self._line_len * 0.85)))
        text_colour = overlay.WHITE if line.who != script.NARRATOR else (225, 225, 190)
        for row in rows:
            if show <= 0:
                break
            overlay.text(canvas, row[:show], (left, top), scale, text_colour, 2)
            show -= len(row) + 1
            top += int(44 * k)
        if footer:
            hint = self.words.say("next")
            (tw, _), _ = cv2.getTextSize(hint, overlay.FONT, 0.5 * k, 1)
            overlay.text(canvas, hint, (x1 - tw - int(20 * k), y1 - int(14 * k)), 0.5 * k,
                         overlay.GREY)
            n = len(self.lines)
            for i in range(n):
                c = (x0 + int((26 + i * 18) * k), y1 - int(18 * k))
                cv2.circle(canvas, c, max(2, int(4 * k)), colour if i <= self.line_index
                           else overlay.DIM, -1 if i <= self.line_index else 1, cv2.LINE_AA)

    def _draw_result(self, canvas) -> None:
        w, h = self.screen_size
        level = self.level
        size = h * 0.075
        for i in range(3):
            _star(canvas, (w / 2 + (i - 1) * size * 2.6, h * 0.2), size, filled=i < level.stars)
        say = self.words.say
        lines = [
            (say("complete", n=self.chapter + 1), 1.6, overlay.CYAN, 3),
            (self.words.chapters[self.chapter].title, 1.0, overlay.WHITE, 2),
            (say("score", score=level.stats.score) + (
                say("new_best") if self._new_best
                else say("best", best=self.progress.score(self.chapter))),
             1.0, overlay.GREEN if self._new_best else overlay.WHITE, 2),
            (say("hits", hits=level.stats.hits, combo=level.stats.best_combo),
             0.7, overlay.GREY, 1),
        ]
        overlay.draw_panel(canvas, lines, h * 0.48)
        self._draw_button(canvas, self.button_box(), say("continue"), self._hold.fill(0))
        overlay.text_centered(canvas, say("button"), int(h * 0.91), 0.6, overlay.GREY)

    def _draw_retry(self, canvas) -> None:
        w, h = self.screen_size
        say = self.words.say
        overlay.text_centered(canvas, say("falters"), int(h * 0.24), 1.8, overlay.YELLOW, 3)
        overlay.text_centered(canvas, self._heading(), int(h * 0.32), 0.9, overlay.GREY, 2)
        # The same box as in a scene, moved up to make room for the button.
        x0, y0, x1, y1 = self.dialogue_box()
        shift = int(h * 0.29)
        self._draw_dialogue(canvas, self.line, (x0, y0 - shift, x1, y1 - shift), footer=False)
        self._draw_button(canvas, self.button_box(), say("again"), self._hold.fill(0))
        overlay.text_centered(canvas, say("button"), int(h * 0.91), 0.6, overlay.GREY)

    def draw_over(self, canvas, scores: Optional[HighScores], rank: Optional[int]) -> None:
        w, h = self.screen_size
        n = len(self.levels)
        earned = sum(self.progress.stars(i) for i in range(n))
        size = h * 0.022
        for i in range(n):
            for k in range(3):
                x = w / 2 + (i - (n - 1) / 2) * size * 9 + (k - 1) * size * 2.5
                _star(canvas, (x, h * 0.86), size, filled=k < self.progress.stars(i))
        overlay.draw_panel(canvas, [
            (self.words.say("end"), 1.8, overlay.CYAN, 3),
            (self.words.title, 1.0, overlay.WHITE, 2),
            (self.words.say("total", score=self.total, stars=earned, of=3 * n),
             1.0, overlay.GREEN, 2),
            (self.words.say("replay"), 0.7, overlay.GREY, 1),
            (self.words.say("end_keys"), 0.7, overlay.YELLOW, 1),
        ], h * 0.48)
