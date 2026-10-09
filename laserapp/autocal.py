"""Automatic calibration: the screen shows the targets, the camera finds them.

The camera is already pointed at the screen, so no laser and no printed
pattern are needed. The screen lights one bright dot at a time at a known
position; each captured frame is differenced against a reference frame of the
black screen, which leaves the dot and nothing else - ambient light, glare,
reflections and the screen's own backlight glow all cancel out.

That differencing is what makes this work where a displayed chessboard does
not: a chessboard has to be *recognised* in a low-contrast, glare-streaked
image of a screen, while a lit dot only has to be brighter than the same
screen was a moment ago.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .calibration import Calibration, solve
from .lens import LensModel

DOT_RADIUS = 18
MIN_PEAK = 35           # difference value that counts as "the dot is lit"
STABLE_DISTANCE = 2.5   # px, between two consecutive detections
STABLE_SAMPLES = 3      # agreeing readings averaged into one point
MOVE_MIN = 12.0         # px; the dot must have visibly moved since the last point
DWELL = 0.2             # s after lighting a dot before readings are taken
MAX_BLOB = 4000         # px; anything larger is not a dot
POINT_TIMEOUT = 2.5     # s before a point is given up on
MIN_POINTS = 8


def find_dot(frame: np.ndarray, reference: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Brightest thing that was not there in `reference`. Returns (x, y, peak)."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(grey, reference)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)

    # The camera keeps adjusting its exposure, which lifts or drops the whole
    # frame between the reference and now. Removing the median takes that
    # global shift out, so the peak is the dot and not the brightest wall.
    floor = int(np.median(diff[::4, ::4]))
    if floor:
        diff = cv2.subtract(diff, floor)

    _, peak, _, loc = cv2.minMaxLoc(diff)
    if peak < MIN_PEAK:
        return None

    # Weighted centroid in a window around the peak, with everything below
    # 60% of the peak treated as background.
    half = 40
    x0, y0 = max(0, loc[0] - half), max(0, loc[1] - half)
    x1, y1 = min(diff.shape[1], loc[0] + half), min(diff.shape[0], loc[1] + half)
    win = diff[y0:y1, x0:x1].astype(np.float32)
    win -= peak * 0.6
    np.clip(win, 0, None, out=win)
    total = float(win.sum())
    if total <= 0:
        return None
    if int(np.count_nonzero(win)) > MAX_BLOB:
        return None           # a large bright area is glare, not the dot
    m = cv2.moments(win, binaryImage=False)
    return x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"], float(peak)


class AutoCalibration:
    """State machine: black reference, then one dot at a time, then solve."""

    REFERENCE, POINTS, FINISHED = "reference", "points", "finished"

    def __init__(self, screen_size: Tuple[int, int], camera_size: Tuple[int, int],
                 cols: int = 5, rows: int = 4, margin: float = 0.08,
                 lens: bool = True, settle: float = 0.8,
                 dwell: float = DWELL, timeout: float = POINT_TIMEOUT) -> None:
        self.screen_size = screen_size
        self.camera_size = camera_size
        self.lens = lens
        self.settle = settle        # black-screen time before the reference
        self.dwell = dwell          # per-point settle before readings count
        self.timeout = timeout      # per-point give-up time
        sw, sh = screen_size
        xs = np.linspace(sw * margin, sw * (1 - margin), cols)
        ys = np.linspace(sh * margin, sh * (1 - margin), rows)
        self.targets: List[Tuple[float, float]] = [(float(x), float(y)) for y in ys for x in xs]

        self.state = self.REFERENCE
        self.index = 0
        self.samples: List[Tuple[float, float]] = []   # camera points, aligned with kept
        self.kept: List[Tuple[float, float]] = []      # screen points actually found
        self.skipped = 0
        self.reference: Optional[np.ndarray] = None
        self._ref_frames: List[np.ndarray] = []
        self._last: Optional[Tuple[float, float, float]] = None
        self._agreeing: List[Tuple[float, float]] = []
        self._accepted: Optional[Tuple[float, float]] = None
        self._started = time.time()
        self._seen_seq = -1
        self.result: Optional[Calibration] = None
        self.message = "підготовка..."

    # -- state --------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.state == self.FINISHED

    @property
    def progress(self) -> float:
        return self.index / float(len(self.targets))

    # -- rendering ----------------------------------------------------------
    def render(self, canvas) -> None:
        """Black screen for the reference, then the current dot."""
        if self.state == self.POINTS and self.index < len(self.targets):
            x, y = self.targets[self.index]
            cv2.circle(canvas, (int(x), int(y)), DOT_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), DOT_RADIUS + 10, (90, 90, 90), 1, cv2.LINE_AA)

    # -- input --------------------------------------------------------------
    def update(self, frame: np.ndarray, seq: int) -> None:
        if self.state == self.FINISHED or frame is None or seq == self._seen_seq:
            return
        self._seen_seq = seq
        now = time.time()

        if self.state == self.REFERENCE:
            # Let the screen go black and the camera settle before sampling it.
            if now - self._started < self.settle:
                return
            self._ref_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if len(self._ref_frames) >= 3:
                self.reference = np.median(np.stack(self._ref_frames), axis=0).astype(np.uint8)
                self._ref_frames.clear()
                self.state = self.POINTS
                self._started = now
                self.message = "шукаю екран"
            return

        if now - self._started < self.dwell:
            return

        found = find_dot(frame, self.reference)

        # The camera lags the screen by a few frames, so right after moving to
        # a new target the previous dot is often still the brightest thing in
        # view. Readings that have not moved since the last accepted point are
        # that stale dot, and accepting one would pair a camera position with
        # the wrong screen position.
        if (found is not None and self._accepted is not None
                and float(np.hypot(found[0] - self._accepted[0],
                                   found[1] - self._accepted[1])) < MOVE_MIN):
            self._last = None
            self._agreeing.clear()
            return

        if found is not None and self._last is not None:
            moved = float(np.hypot(found[0] - self._last[0], found[1] - self._last[1]))
            if moved <= STABLE_DISTANCE:
                # Consecutive frames agree: this is the dot, not a flicker of
                # the display caught mid-refresh. Average several such readings.
                if not self._agreeing:
                    self._agreeing.append(self._last[:2])
                self._agreeing.append(found[:2])
                if len(self._agreeing) >= STABLE_SAMPLES:
                    mean = np.mean(np.array(self._agreeing), axis=0)
                    self._accepted = (float(mean[0]), float(mean[1]))
                    self.samples.append(self._accepted)
                    self.kept.append(self.targets[self.index])
                    self._advance(now)
                    return
            else:
                self._agreeing.clear()
        self._last = found

        if now - self._started > self.timeout:
            self.skipped += 1
            self._advance(now)

    def _advance(self, now: float) -> None:
        self.index += 1
        self._last = None
        self._agreeing.clear()
        self._started = now
        if self.index >= len(self.targets):
            self._finish()

    def _finish(self) -> None:
        self.state = self.FINISHED
        if len(self.kept) < MIN_POINTS:
            self.message = (f"камера побачила лише {len(self.kept)} з {len(self.targets)} точок — "
                            "перевір, чи видно їй увесь екран")
            return
        try:
            self.result = solve(self.samples, self.kept, self.camera_size,
                                self.screen_size, lens=self.lens)
        except RuntimeError as exc:
            self.message = f"калібрування не вдалося: {exc}"
            return
        r = self.result
        self.message = f"Відкалібровано за екраном — похибка {r.error:.1f} пкс, точок: {len(self.kept)}"
        if r.lens is not None:
            self.message += (f" (об'єктив k1={r.lens.k1:+.2f}; "
                             f"без корекції {r.error_no_lens:.1f} пкс)")
        if self.skipped:
            self.message += f", не побачено: {self.skipped}"
