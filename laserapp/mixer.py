"""Sound effects mixed in this process, through one audio stream kept open.

Handing every effect to a player program of its own costs a process start
per bang - a delay you can hear - and lets only one effect through at a time
(on Windows, winsound has the one channel). Two players firing together
were heard as one shot. Here effects are added into a stream that is always
running, so any number overlap, and each starts within a few hundredths of
a second.

The stream is PulseAudio's simple API on Linux (which PipeWire provides
too), through ctypes, and the `sounddevice` package elsewhere (its Windows
and macOS wheels bring PortAudio along). With neither, `get()` returns None
and sound.py plays effects the old way.

Every recording in assets/ is 24 kHz mono 16-bit, so nothing is resampled.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import threading
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

RATE = 24000
CHUNK = 240                  # 10 ms written at a time
LATENCY = 0.04               # how far ahead of the speaker the stream runs


class Mixer:
    def __init__(self) -> None:
        self._playing: List[list] = []          # [samples, position]
        self._lock = threading.Lock()
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    # -- what the game calls -------------------------------------------------
    def play(self, path: Path, gain: float = 1.0) -> bool:
        samples = self._load(path)
        if samples is None:
            return False
        with self._lock:
            self._playing.append([samples * gain if gain != 1.0 else samples, 0])
        return True

    def _load(self, path: Path) -> Optional[np.ndarray]:
        key = str(path)
        if key not in self._cache:
            try:
                with wave.open(key, "rb") as f:
                    ok = (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (RATE, 1, 2)
                    raw = f.readframes(f.getnframes())
                self._cache[key] = (np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
                                    if ok else None)
            except (OSError, EOFError, wave.Error):
                self._cache[key] = None
        return self._cache[key]

    # -- what the stream calls -----------------------------------------------
    def mix(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self._lock:
            keep = []
            for item in self._playing:
                samples, at = item
                part = samples[at:at + n]
                out[:len(part)] += part
                item[1] = at + n
                if item[1] < len(samples):
                    keep.append(item)
            self._playing = keep
        # Several bangs at once would clip: round the peaks off instead.
        loud = np.abs(out) > 0.8
        if loud.any():
            out[loud] = np.sign(out[loud]) * (0.8 + 0.2 * np.tanh((np.abs(out[loud]) - 0.8) / 0.2))
        return out


class _Pulse(Mixer):
    """PulseAudio's blocking simple API, written to from a thread of its own."""

    class _Spec(ctypes.Structure):
        _fields_ = [("format", ctypes.c_int), ("rate", ctypes.c_uint32),
                    ("channels", ctypes.c_uint8)]

    class _Attr(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint32)
                    for name in ("maxlength", "tlength", "prebuf", "minreq", "fragsize")]

    def __init__(self) -> None:
        super().__init__()
        name = ctypes.util.find_library("pulse-simple") or "libpulse-simple.so.0"
        lib = ctypes.CDLL(name)
        lib.pa_simple_new.restype = ctypes.c_void_p
        lib.pa_simple_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                      ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_int)]
        lib.pa_simple_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_int)]
        spec = self._Spec(3, RATE, 1)                       # PA_SAMPLE_S16LE
        unset = 0xFFFFFFFF
        attr = self._Attr(unset, int(LATENCY * RATE) * 2, unset, unset, unset)
        error = ctypes.c_int(0)
        handle = lib.pa_simple_new(None, b"Laser Target", 1, None, b"effects",  # PLAYBACK
                                   ctypes.byref(spec), None, ctypes.byref(attr),
                                   ctypes.byref(error))
        if not handle:
            raise OSError(f"PulseAudio: error {error.value}")
        self._lib, self._handle = lib, handle
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        error = ctypes.c_int(0)
        while True:
            pcm = (self.mix(CHUNK) * 32767).astype("<i2").tobytes()
            # Blocks until there is room: this is what keeps time.
            if self._lib.pa_simple_write(self._handle, pcm, len(pcm), ctypes.byref(error)) < 0:
                return


class _Device(Mixer):
    """PortAudio, through the sounddevice package, calling back for sound."""

    def __init__(self) -> None:
        super().__init__()
        import sounddevice
        self._stream = sounddevice.OutputStream(
            samplerate=RATE, channels=1, dtype="float32", latency=LATENCY,
            callback=lambda out, frames, _time, _status: out.__setitem__(
                (slice(None), 0), self.mix(frames)))
        self._stream.start()


_mixer: Optional[Mixer] = None
_tried = False


def get() -> Optional[Mixer]:
    """The one mixer, opened on first use; None if this machine has neither."""
    global _mixer, _tried
    if not _tried:
        _tried = True
        for kind in ((_Pulse, _Device) if sys.platform.startswith("linux") else (_Device,)):
            try:
                _mixer = kind()
                break
            except Exception:
                continue
    return _mixer
