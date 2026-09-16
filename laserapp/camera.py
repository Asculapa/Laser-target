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


def jpeg_intact(data: np.ndarray) -> bool:
    """True if a JPEG frame (a flat uint8 array) arrived whole.

    USB cameras lose packets, and what arrives is then a frame cut short, two
    frames run together, or one with a stretch missing from the middle. The
    first two have no end marker where it belongs; the third shows up as a
    gap in the restart markers the camera numbers its scan with.
    """
    n = len(data)
    while n > 1 and data[n - 1] == 0:       # drivers pad the buffer with zeros
        n -= 1
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8 \
            or data[n - 2] != 0xFF or data[n - 1] != 0xD9:
        return False
    i, interval, mcus = 2, 0, 0
    while i + 4 <= n and data[i] == 0xFF:
        marker, length = int(data[i + 1]), (int(data[i + 2]) << 8) | int(data[i + 3])
        if i + 2 + length > n:
            return False
        if marker == 0xDD:                  # restart interval, in MCUs
            interval = (int(data[i + 4]) << 8) | int(data[i + 5])
        elif marker == 0xC0:                # frame header: size and subsampling
            h = (int(data[i + 5]) << 8) | int(data[i + 6])
            w = (int(data[i + 7]) << 8) | int(data[i + 8])
            parts = [int(data[i + 11 + 3 * c]) for c in range(int(data[i + 9]))]
            mcus = -(-w // (8 * max(p >> 4 for p in parts))) \
                * -(-h // (8 * max(p & 15 for p in parts)))
        i += 2 + length
        if marker == 0xDA:                  # the scan starts here
            break
    else:
        return False
    if not interval or not mcus:
        return True                         # no restart markers to count
    scan = data[i:n - 2]
    after = scan[1:][scan[:-1] == 0xFF]
    marks = after[after != 0]
    # Some cameras pad the end of the scan with junk, so only the markers the
    # picture needs are checked; whatever follows them is ignored.
    want = -(-mcus // interval) - 1
    return len(marks) >= want and bool(np.all(marks[:want] == 0xD0 + np.arange(want) % 8))


class FrameDecoder:
    """Turns the camera's compressed frames into pictures, and throws away the
    ones that arrived damaged instead of passing them on.

    One kind of damage it repairs. A camera can slip: from some frame on, the
    end of every frame turns up at the start of the next buffer instead, so no
    buffer holds a whole picture although nothing is missing. It can stay that
    way for a minute. The two parts are put back together here, which costs a
    frame of delay and keeps the picture.
    """

    def __init__(self) -> None:
        self.loss = 0.0                              # recent share of frames lost
        self._head: Optional[np.ndarray] = None      # a frame still short of its end

    def decode(self, data: np.ndarray) -> Optional[np.ndarray]:
        data = data.reshape(-1)
        whole = data if jpeg_intact(data) else self._join(data)
        frame = cv2.imdecode(whole, cv2.IMREAD_COLOR) if whole is not None else None
        self.loss = 0.95 * self.loss + 0.05 * (frame is None)
        return frame

    def _join(self, data: np.ndarray) -> Optional[np.ndarray]:
        raw = data.tobytes()
        head, self._head = self._head, None
        start = raw.rfind(b"\xff\xd8")
        if start > 0:
            self._head = data[start:].copy()         # the next buffer has its end
        if head is None or len(raw) < 6:
            return None
        # The buffer opens with its own start marker and comment; the end of
        # the previous frame follows those.
        skip = 4 + ((raw[4] << 8) | raw[5]) if raw[2:4] == b"\xff\xfe" else 2
        end = raw.find(b"\xff\xd9", skip)
        if end < 0:
            return None
        whole = np.concatenate([head, data[skip:end + 2]])
        return whole if jpeg_intact(whole) else None


class Camera:
    """Background grabber that always hands out the freshest frame.

    A stream that stops delivering (USB glitch, cable knocked, driver stall)
    is closed and reopened by the grabber itself, so the app never has to be
    restarted to get the picture back.

    Where the backend allows it, frames are taken from the camera still
    compressed and decoded here. Left to decode them itself, OpenCV answers a
    frame damaged on the way with the previous picture, or with one that is
    grey from the damage down - and says nothing, so the app would go on
    looking for the laser in a picture that is seconds old.
    """

    STALL_AFTER = 1.5   # seconds without a frame before the stream counts as dead
    OPEN_GRACE = 5.0    # a just-opened camera gets this long to deliver its first
    RETRY_EVERY = 1.0   # pause between attempts to reopen
    HANG_AFTER = 8.0    # grabber stuck inside a driver call: replace the thread

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        exposure: Optional[float] = None,
        backend: int = DEFAULT_BACKEND,
        name: str = "",
    ) -> None:
        self.index = index
        self.name = name            # lets a replugged camera be found again
        self.requested = (width, height)
        self.requested_fps = fps
        self.backend = backend
        self._raw = False           # frames arrive compressed and are decoded here
        self._decoder = FrameDecoder()
        cap = self._open(index)
        if cap is None:
            raise RuntimeError(f"Cannot open camera {index}")
        self.cap = cap
        # Held around every use of `cap` except the grabber's blocking read.
        self._cap_lock = threading.RLock()

        self.auto_exposure = True
        self._exposure: Optional[float] = None
        if exposure is not None:
            self.set_exposure(exposure)

        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._stamp = 0.0           # when the newest frame arrived
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps = 0.0
        self._gen = 0               # bumped to retire a hung grabber thread
        self._beat = time.perf_counter()
        self.reopens = 0
        self._thread = threading.Thread(target=self._loop, args=(0, cap), daemon=True)
        self._thread.start()

    def _open(self, index: int) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(index, self.backend)
        if not cap.isOpened():  # fall back to whatever backend is available
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None

        # MJPG keeps USB bandwidth low enough for 720p30 on most webcams.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested[1])
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._raw = False
        if not IS_WINDOWS and int(cap.get(cv2.CAP_PROP_FOURCC)) == cv2.VideoWriter_fourcc(*"MJPG"):
            try:
                self._raw = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
            except Exception:
                pass
        return cap

    def _reopen(self) -> Optional[cv2.VideoCapture]:
        cap = self._open(self.index)
        if cap is None and self.name:
            # A replugged camera usually comes back under a different number.
            try:
                match = next((c for c in list_cameras()
                              if c.name == self.name and c.index != self.index), None)
            except Exception:
                match = None
            if match is not None:
                cap = self._open(match.index)
                if cap is not None:
                    self.index = match.index
        return cap

    # -- properties ---------------------------------------------------------
    @property
    def size(self) -> Tuple[int, int]:
        with self._cap_lock:
            return (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested[0],
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested[1],
            )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def exposure(self) -> float:
        with self._cap_lock:
            return float(self.cap.get(cv2.CAP_PROP_EXPOSURE))

    @property
    def stalled(self) -> bool:
        """True while a camera that was delivering frames has gone quiet."""
        return (self._frame is not None
                and time.perf_counter() - self._stamp > self.STALL_AFTER)

    @property
    def age(self) -> float:
        """Seconds since the newest frame arrived."""
        return time.perf_counter() - self._stamp

    @property
    def loss(self) -> float:
        """Recent share of frames that arrived damaged and were thrown away."""
        return self._decoder.loss

    # -- controls -----------------------------------------------------------
    def set_exposure(self, value: float) -> None:
        """Switch to manual exposure and apply `value`.

        Units are the driver's: V4L2 uses 100 us steps (e.g. 50), DirectShow
        uses log2 seconds (e.g. -6 = 1/64 s).
        """
        with self._cap_lock:
            if IS_WINDOWS:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # DirectShow: manual
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # V4L2: 1 = manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        self._exposure = float(value)
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
        with self._cap_lock:
            if IS_WINDOWS:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if enabled else 0.25)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if enabled else 1)
        self.auto_exposure = enabled

    def _restore_exposure(self) -> None:
        """Put a reopened camera back on the exposure the user had chosen."""
        if self.auto_exposure:
            return
        if self._exposure is not None:
            self.set_exposure(self._exposure)
        else:
            self.set_auto_exposure(False)

    # -- frames -------------------------------------------------------------
    def _loop(self, gen: int, cap: Optional[cv2.VideoCapture]) -> None:
        last = time.perf_counter()
        alive = last                # when this capture last proved it works
        grace = self.OPEN_GRACE
        while not self._stop.is_set() and gen == self._gen:
            self._beat = time.perf_counter()
            if cap is None:
                cap = self._reopen()
                if cap is None:
                    self._stop.wait(self.RETRY_EVERY)
                    continue
                with self._cap_lock:
                    if gen != self._gen:
                        break
                    self.cap = cap
                    self._restore_exposure()
                self.reopens += 1
                print(f"[camera] reopened camera {self.index}")
                last = alive = time.perf_counter()
                grace = self.OPEN_GRACE
                continue

            ok, frame = cap.read()
            if ok and frame is not None and self._raw and frame.ndim < 3:
                frame = self._decoder.decode(frame)
            now = time.perf_counter()
            if not ok or frame is None:
                if now - alive > grace:
                    print(f"[camera] no usable frames from camera {self.index} "
                          f"for {now - alive:.1f}s - reopening")
                    with self._cap_lock:
                        cap.release()
                    cap = None
                    self._fps = 0.0
                else:
                    time.sleep(0.01)
                continue
            alive = now
            grace = self.STALL_AFTER
            dt = now - last
            last = now
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._stamp = now
        if cap is not None:
            with self._cap_lock:
                cap.release()

    def _replace_grabber(self) -> None:
        """The grabber is stuck inside the driver and cannot be interrupted:
        leave it to finish on its own and carry on with a fresh one."""
        print(f"[camera] camera {self.index} stopped responding - restarting capture")
        self._gen += 1
        self._beat = time.perf_counter()
        self._fps = 0.0
        self._thread = threading.Thread(target=self._loop, args=(self._gen, None),
                                        daemon=True)
        self._thread.start()

    def read(self) -> Tuple[Optional[np.ndarray], int]:
        """Return (frame, sequence). Sequence lets callers skip re-processing.

        The frame is None until the first one arrives, and again while a
        stalled camera is being reopened - a stale frame would look frozen.
        """
        now = time.perf_counter()
        if now - self._beat > self.HANG_AFTER and not self._stop.is_set():
            self._replace_grabber()
        with self._lock:
            if self._frame is None or now - self._stamp > self.STALL_AFTER:
                return None, self._seq
            return self._frame, self._seq

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        with self._cap_lock:
            self.cap.release()
