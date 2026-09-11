"""Camera enumeration and threaded capture (so rendering never blocks on grab)."""
from __future__ import annotations

import glob
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraInfo:
    index: int          # the number OpenCV wants (N in /dev/videoN, DirectShow order on Windows)
    name: str
    external: bool      # a plugged-in USB camera rather than the built-in one

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name}" + ("" if self.external else "  (built-in)")


_BUILTIN_HINTS = ("integrated", "built-in", "builtin", "internal", "facetime")


IS_WINDOWS = sys.platform.startswith("win")
# DirectShow opens quickly and its device order is the one pygrabber reports;
# Media Foundation can take several seconds per open.
DEFAULT_BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2


def list_cameras() -> List[CameraInfo]:
    """Capture devices the app can open, in the order OpenCV indexes them."""
    cams = _list_windows() if IS_WINDOWS else _list_v4l2()
    for c in cams:
        c.external = not any(h in c.name.lower() for h in _BUILTIN_HINTS)
    return cams


def _list_windows() -> List[CameraInfo]:
    """DirectShow video inputs. Names come from pygrabber when it is
    installed; otherwise indices are probed and labelled generically."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        return [CameraInfo(index=i, name=n, external=True) for i, n in enumerate(names)]
    except Exception:
        pass
    found: List[CameraInfo] = []
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        cap.release()
        if ok:
            found.append(CameraInfo(index=i, name=f"Camera {i}", external=True))
    return found


def _list_v4l2() -> List[CameraInfo]:
    """Capture-capable /dev/video* devices, in device order.

    UVC cameras expose several nodes per physical device; only the ones whose
    V4L2 capabilities include `capture` actually produce frames - the others
    carry metadata and would open as an empty stream.
    """
    found: List[CameraInfo] = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*"),
                       key=lambda p: int(re.search(r"\d+$", p).group())):
        node = os.path.basename(path)
        index = int(re.search(r"\d+$", node).group())
        try:
            with open(os.path.join(path, "name")) as fh:
                name = fh.read().strip()
        except OSError:
            continue
        if not _is_capture_device(index):
            continue
        name = name.split(":")[0].strip() or node
        found.append(CameraInfo(index=index, name=name, external=True))
    return found


def _is_capture_device(index: int) -> bool:
    """True if /dev/videoN can deliver frames (udev tags metadata nodes)."""
    try:
        import subprocess
        out = subprocess.run(["udevadm", "info", "-q", "property", "-n", f"/dev/video{index}"],
                             capture_output=True, text=True, timeout=2).stdout
        m = re.search(r"ID_V4L_CAPABILITIES=(\S*)", out)
        if m:
            return "capture" in m.group(1)
    except Exception:
        pass
    # No udev answer: try opening it.
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    ok = cap.isOpened() and cap.grab()
    cap.release()
    return ok


def resolve_camera(spec: Optional[str]) -> Tuple[int, str]:
    """Turn a --camera value into a device index.

    Accepts a number, a case-insensitive name fragment ("genius"), or "auto"
    (the default): prefer a plugged-in USB camera over the built-in one.
    """
    cams = list_cameras()
    if spec is not None and spec.strip().lstrip("-").isdigit():
        index = int(spec)
        match = next((c for c in cams if c.index == index), None)
        return index, match.name if match else f"camera {index}"
    if spec and spec.lower() != "auto":
        needle = spec.lower()
        for c in cams:
            if needle in c.name.lower():
                return c.index, c.name
        raise RuntimeError(f"No camera matching {spec!r}. Available: "
                           + (", ".join(c.label for c in cams) or "none"))
    if not cams:
        return 0, "camera 0"
    external = [c for c in cams if c.external]
    chosen = external[0] if external else cams[0]
    return chosen.index, chosen.name


class Camera:
    """Background grabber that always hands out the freshest frame."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        exposure: Optional[float] = None,
        backend: int = DEFAULT_BACKEND,
    ) -> None:
        self.index = index
        self.requested = (width, height)
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():  # fall back to whatever backend is available
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

        # MJPG keeps USB bandwidth low enough for 720p30 on most webcams.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.auto_exposure = True
        if exposure is not None:
            self.set_exposure(exposure)

        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- properties ---------------------------------------------------------
    @property
    def size(self) -> Tuple[int, int]:
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested[0],
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested[1],
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def exposure(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_EXPOSURE))

    # -- controls -----------------------------------------------------------
    def set_exposure(self, value: float) -> None:
        """Switch to manual exposure and apply `value`.

        Units are the driver's: V4L2 uses 100 us steps (e.g. 50), DirectShow
        uses log2 seconds (e.g. -6 = 1/64 s).
        """
        if IS_WINDOWS:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow: manual
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # V4L2: 1 = manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        self.auto_exposure = False

    def nudge_exposure(self, factor: float) -> float:
        """Darker for factor < 1, brighter for factor > 1."""
        current = self.exposure
        if IS_WINDOWS:
            # Log2 scale: one step doubles or halves the exposure time.
            if current >= 0:
                current = -6.0
            value = min(0.0, max(-13.0, current + (1 if factor > 1 else -1)))
        else:
            if current <= 0:
                current = 100.0
            value = max(1.0, round(current * factor))
        self.set_exposure(value)
        return value

    def set_auto_exposure(self, enabled: bool) -> None:
        if IS_WINDOWS:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enabled else 0.25)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if enabled else 1)
        self.auto_exposure = enabled

    # -- frames -------------------------------------------------------------
    def _loop(self) -> None:
        last = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
            with self._lock:
                self._frame = frame
                self._seq += 1

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Return (frame, sequence). Sequence lets callers skip re-processing."""
        with self._lock:
            if self._frame is None:
                return None, self._seq
            return self._frame, self._seq

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
