"""The script of the story game: who speaks when, and over which picture.

The words themselves are kept per language, in story_words_*.py, so that the
story can be edited or translated without touching the game - and so that
tools/story_audio.py can read all of it without OpenCV. Every line has an
id, which is the key of its text in each language and the name of its
recording; after changing a line's text, run build_story_audio.sh to record
it again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from . import story_words_en, story_words_uk

NARRATOR, MAREN, PIP, GLOAM = "narrator", "maren", "pip", "gloam"


@dataclass(frozen=True)
class Line:
    id: str
    who: str
    text: str
    scene: str = ""         # the picture to change to; "" keeps the one showing


@dataclass(frozen=True)
class Script:
    """One chapter's scenes."""
    title: str
    intro: Tuple[Line, ...]
    outro: Tuple[Line, ...]
    fail: Line


@dataclass(frozen=True)
class Language:
    code: str               # also the folder its recordings are in
    name: str               # as it calls itself
    title: str
    names: Dict[str, str]   # the speakers
    chapters: Tuple[Script, ...]
    title_line: Line
    ui: Dict[str, str]      # everything on screen that is not a spoken line
    stress: Dict[str, str]  # words the speech model has to be told how to stress

    def say(self, key: str, **values) -> str:
        return self.ui[key].format(**values) if values else self.ui[key]

    def all_lines(self) -> List[Line]:
        out = [self.title_line]
        for chapter in self.chapters:
            out += [*chapter.intro, *chapter.outro, chapter.fail]
        return out


def _cues(prefix: str, rows: Sequence[tuple]) -> Tuple[tuple, ...]:
    return tuple((f"{prefix}{i + 1:02d}", *row) for i, row in enumerate(rows))


# Per chapter: the scene before it, the scene after it, and who speaks when
# it is lost. A cue is (id, who) or (id, who, picture).
PLOT = (
    (_cues("c1i", [(NARRATOR, "night"), (NARRATOR,), (MAREN, "room"), (MAREN,), (MAREN,),
                   (NARRATOR, "gale")]),
     _cues("c1o", [(PIP, "room"), (NARRATOR,), (PIP,), (MAREN,), (PIP,), (NARRATOR,)]),
     ("c1f", MAREN)),
    (_cues("c2i", [(NARRATOR, "fog"), (MAREN,), (GLOAM,), (PIP,), (MAREN,)]),
     _cues("c2o", [(NARRATOR, "room"), (GLOAM,), (PIP,), (MAREN,)]),
     ("c2f", PIP)),
    (_cues("c3i", [(NARRATOR, "bay"), (MAREN,), (PIP,), (MAREN,), (GLOAM,)]),
     _cues("c3o", [(NARRATOR, "bay"), (PIP,), (MAREN,), (MAREN,)]),
     ("c3f", MAREN)),
    (_cues("c4i", [(NARRATOR, "sky"), (MAREN,), (PIP,), (MAREN,)]),
     _cues("c4o", [(NARRATOR, "sky"), (MAREN,), (PIP,), (NARRATOR, "dark")]),
     ("c4f", MAREN)),
    (_cues("c5i", [(NARRATOR, "storm"), (GLOAM,), (MAREN,), (PIP,)]),
     _cues("c5o", [(NARRATOR, "lit"), (GLOAM,), (MAREN,), (NARRATOR, "dawn"), (MAREN,),
                   (NARRATOR,), (PIP,)]),
     ("c5f", PIP)),
)
TITLE_CUE = ("title", NARRATOR, "lit")


def _language(words) -> Language:
    def line(cue: tuple) -> Line:
        return Line(cue[0], cue[1], words.LINES[cue[0]], *cue[2:])

    chapters = tuple(
        Script(title, tuple(map(line, intro)), tuple(map(line, outro)), line(fail))
        for title, (intro, outro, fail) in zip(words.CHAPTERS, PLOT))
    return Language(words.CODE, words.NAME, words.TITLE, dict(words.NAMES), chapters,
                    line(TITLE_CUE), dict(words.UI), dict(getattr(words, "STRESS", {})))


LANGUAGES: Dict[str, Language] = {
    words.CODE: _language(words) for words in (story_words_en, story_words_uk)
}
ENGLISH = LANGUAGES["en"]
