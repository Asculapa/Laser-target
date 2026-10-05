"""Sound: plays the WAV files in a folder, without adding a dependency.

Linux hands the file to whichever player is installed (PipeWire, PulseAudio
or ALSA), macOS to afplay, Windows to winsound. Nothing here ever blocks the
main loop, and a machine with no way to play sound simply stays silent.

Two channels: a *voice*, of which there is one at a time - saying a new line
cuts the old one off - and *blips*, short effects that play over it. *Music*
is a third, which starts again when it ends.

Blips go through mixer.py where it can open a stream, so any number of them
play at once (two players firing together are two shots) and without the
delay of starting a program. Where it cannot, they are played like the rest.

On Windows the voice and the blips share winsound, which plays one sound at
a time, so music goes another way there - MCI, through ctypes - and a machine
where that does not work has the voices and the effects without the music.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Union

from . import mixer

enabled = True              # --no-sound, and the self-test, turn this off

_command: Optional[List[str]] = None
_voice_until = 0.0          # when the line being spoken ends, whoever speaks it


def _player() -> List[str]:
    """The command that plays a WAV file here; empty if there is none."""
    global _command
    if _command is None:
        _command = []
        options = (["afplay"],) if sys.platform == "darwin" else \
            (["pw-play"], ["paplay"], ["aplay", "-q"])
        for cmd in options:
            if shutil.which(cmd[0]):
                _command = list(cmd)
                break
    return _command


def _mci(command: str) -> Optional[str]:
    """Windows' own media player, told something. None if it would not."""
    try:
        import ctypes
        reply = ctypes.create_unicode_buffer(64)
        if ctypes.windll.winmm.mciSendStringW(command, reply, 64, None) != 0:
            return None
        return reply.value
    except Exception:
        return None


class Player:
    BLIP_GAP = 0.08         # without the mixer, two effects closer together than this are one

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._lengths: Dict[str, Optional[float]] = {}
        self._voice: Optional[subprocess.Popen] = None
        self._last_blip = 0.0
        self._music: Optional[str] = None       # what is playing as music
        self._music_at = 0.0
        # The player it is playing in: a process, or on Windows an MCI alias.
        self._tune: Union[subprocess.Popen, str, None] = None
        # What each voice file says. A line whose text has been edited since
        # it was recorded is shown without sound rather than with the wrong one.
        try:
            self.texts = json.loads((folder / "voices.json").read_text())
        except (OSError, ValueError):
            self.texts = {}

    def length(self, name: str, text: Optional[str] = None) -> Optional[float]:
        """Seconds of sound in `name`, or None if there is no such recording
        (or it is a recording of something other than `text`)."""
        if text is not None and self.texts.get(name) != text:
            return None
        if name not in self._lengths:
            try:
                with wave.open(str(self.folder / f"{name}.wav"), "rb") as f:
                    self._lengths[name] = f.getnframes() / float(f.getframerate())
            except (OSError, EOFError, wave.Error):
                self._lengths[name] = None
        return self._lengths[name]

    def say(self, name: str, text: Optional[str] = None) -> Optional[float]:
        """Speak a line, cutting off the one before. Returns its length."""
        global _voice_until
        self.stop()
        length = self.length(name, text)
        if length is not None and enabled:
            self._voice = self._start(name, voice=True)
            _voice_until = time.monotonic() + length
        return length

    def music(self, name: str) -> None:
        """Keep `name` playing, round and round. To be called every frame:
        it starts the music, and starts it again each time it has ended."""
        length = self.length(name)
        if length is None or not enabled:
            return
        now = time.monotonic()
        # Not again before its time is up, whatever became of the player: one
        # that cannot play must not be started sixty times a second.
        if self._music == name and (now - self._music_at < length or self._tune_alive()):
            return
        self._tune_stop()
        self._music, self._music_at = name, now
        path = str(self.folder / f"{name}.wav")
        if sys.platform == "win32":
            alias = f"laser{id(self)}"
            if _mci(f'open "{path}" type waveaudio alias {alias}') is not None:
                self._tune = alias
                _mci(f"play {alias}")
        else:
            self._tune = self._start(name, voice=False)

    def _tune_alive(self) -> bool:
        if isinstance(self._tune, str):
            return _mci(f"status {self._tune} mode") in ("playing", "paused")
        return self._tune is not None and self._tune.poll() is None

    def _tune_stop(self) -> None:
        if isinstance(self._tune, str):
            _mci(f"stop {self._tune}")
            _mci(f"close {self._tune}")
        elif self._tune is not None:
            try:
                self._tune.terminate()
                os.kill(self._tune.pid, signal.SIGCONT)   # a held player cannot hear it
            except OSError:
                pass
        self._tune = None

    def hold(self) -> bool:
        """Stop the line being spoken where it is, and the music with it.
        False if the line cannot be held here (Windows), or there is none."""
        if isinstance(self._tune, str):
            _mci(f"pause {self._tune}")
        elif self._tune is not None and self._tune.poll() is None:
            try:
                os.kill(self._tune.pid, signal.SIGSTOP)
            except OSError:
                pass
        if self._voice is None or self._voice.poll() is not None \
                or not hasattr(signal, "SIGSTOP"):
            return False
        try:
            os.kill(self._voice.pid, signal.SIGSTOP)
        except OSError:
            return False
        return True

    def release(self) -> None:
        """Let a held line, and held music, go on."""
        if isinstance(self._tune, str):
            _mci(f"resume {self._tune}")
        for held in (self._voice, self._tune):
            if isinstance(held, subprocess.Popen) and held.poll() is None:
                try:
                    os.kill(held.pid, signal.SIGCONT)
                except OSError:
                    pass

    def blip(self, name: str) -> None:
        if not enabled or self.length(name) is None:
            return
        mix = mixer.get()
        if mix is not None and mix.play(self.folder / f"{name}.wav"):
            return
        now = time.monotonic()
        if now - self._last_blip < self.BLIP_GAP:
            return
        # winsound has one channel: an effect would cut the speech short.
        if sys.platform == "win32" and now < _voice_until:
            return
        self._last_blip = now
        self._start(name, voice=False)

    def stop(self) -> None:
        global _voice_until
        _voice_until = 0.0
        self._music = None
        self._tune_stop()
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self._voice is not None:
            try:
                self._voice.terminate()
                self.release()            # a held player cannot hear the request
            except OSError:
                pass
            self._voice = None

    def _start(self, name: str, voice: bool) -> Optional[subprocess.Popen]:
        path = str(self.folder / f"{name}.wav")
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC
                                   | winsound.SND_NODEFAULT)
                return None
            cmd = _player()
            if not cmd:
                return None
            return subprocess.Popen(cmd + [path], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return None               # no sound is not a reason to stop a game
