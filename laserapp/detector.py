"""Red laser dot detection.

Absolute brightness thresholds do not survive real rooms: sunlight on the
screen lifts the whole frame, and a camera pointed straight at a display picks
up rolling-shutter banding - broad bright "waves" drifting through the image.
Both are *large-scale* structures, so the dot is found by local contrast
instead: a top-hat filter subtracts whatever the neighbourhood is doing and
leaves only things smaller than the filter, which is what a laser spot is.

What is left is scored on redness (R above the stronger of G and B), which the
grey banding and white glare have none of, and thresholded relative to the
frame's own noise, so the detector re-tunes itself as the light changes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    x: float           # camera pixel coordinates, sub-pixel
    y: float
    area: int
    score: float
    radius: float


@dataclass
class DetectorSettings:
    sensitivity: float = 5.0    # peak must beat the frame's noise by this much
    min_redness: int = 20       # absolute floor on local redness
    min_brightness: int = 50    # absolute floor on the red channel
    min_area: int = 3
    max_area: int = 4000
    blur: int = 3               # odd kernel, 0 disables
    tophat: int = 25            # local-contrast window; must exceed the dot
    core_level: int = 248       # near-saturated pixels count as fully red
    max_aspect: float = 3.0     # a dot is roundish; banding is a long streak
    min_fill: float = 0.30      # blob area / bounding box area


class LaserDetector:
    def __init__(self, settings: Optional[DetectorSettings] = None) -> None:
        self.s = settings or DetectorSettings()
        self._small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._tophat_size = 0
        self._tophat_kernel = None
        self.mask: Optional[np.ndarray] = None
        self.candidates = 0
        self.threshold = 0.0        # last adaptive threshold, for the HUD
        self.saturation = 0.0       # fraction of blown-out pixels, for the HUD

    # -- tuning -------------------------------------------------------------
    def adjust(self, sensitivity: float = 0.0, redness: int = 0) -> None:
        self.s.sensitivity = float(np.clip(self.s.sensitivity + sensitivity, 1.0, 30.0))
        self.s.min_redness = int(np.clip(self.s.min_redness + redness, 0, 200))

    def as_dict(self) -> dict:
        return asdict(self.s)

    def auto_threshold(self, frames) -> Tuple[int, float]:
        """Raise the floor just above whatever the camera sees right now.

        Call with the laser pointed away: the strongest local red contrast in
        view becomes the background level.
        """
        keep = self.s.min_redness
        self.s.min_redness = 4                      # look with the floor down
        worst = 0.0
        try:
            for f in frames:
                found = self.detect(f)
                if found is not None:
                    top = self._local_redness(f)
                    x, y = int(round(found.x)), int(round(found.y))
                    half = max(3, int(found.radius) + 3)
                    patch = top[max(0, y - half):y + half, max(0, x - half):x + half]
                    worst = max(worst, float(patch.max()))
        finally:
            self.s.min_redness = keep
        # Just above anything the detector would have accepted with the floor
        # down - so whatever is in view now stops counting as a laser.
        self.s.min_redness = int(np.clip(worst * 1.15 + 6, 6, 220))
        return self.s.min_redness, self.s.sensitivity

    # -- internals ----------------------------------------------------------
    def _kernel(self):
        size = max(5, self.s.tophat | 1)
        if size != self._tophat_size:
            # A rectangle is separable, so this stays cheap at large sizes.
            self._tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            self._tophat_size = size
        return self._tophat_kernel

    def _local_redness(self, frame: np.ndarray) -> np.ndarray:
        s = self.s
        img = cv2.GaussianBlur(frame, (s.blur, s.blur), 0) if s.blur >= 3 else frame
        b, g, r = cv2.split(img)
        redness = cv2.subtract(r, cv2.max(g, b))        # uint8, saturating at 0

        self._red_channel = r
        return cv2.morphologyEx(redness, cv2.MORPH_TOPHAT, self._kernel())

    # -- detection ----------------------------------------------------------
    def detect(self, frame: np.ndarray,
               roi: Optional[np.ndarray] = None) -> Optional[Detection]:
        """Find the laser spot. `roi` limits the search to the screen area."""
        s = self.s
        top = self._local_redness(frame)
        r = self._red_channel

        # Noise level of this frame, from a subsample: the median absolute
        # deviation is unmoved by the handful of pixels a dot occupies.
        sub = top[::4, ::4].astype(np.float32)
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) or 1.0

        self.threshold = max(float(s.min_redness), med + s.sensitivity * mad)
        self.saturation = float(np.count_nonzero(r >= 250)) / r.size

        seed = cv2.compare(top, self.threshold, cv2.CMP_GE)
        seed = cv2.bitwise_and(seed, cv2.compare(r, s.min_brightness, cv2.CMP_GE))
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, self._small)

        # A spot bright enough to clip the sensor turns white in the middle and
        # loses all redness. Reclaim those pixels, but only where they touch a
        # confirmed red seed - otherwise every sunlit patch joins in.
        weight = top.astype(np.float32)
        mask = seed
        if s.core_level < 255:
            core = cv2.compare(r, s.core_level, cv2.CMP_GE)
            grown = cv2.bitwise_and(cv2.dilate(seed, self._small, iterations=2), core)
            mask = cv2.bitwise_or(seed, grown)
            weight = np.maximum(weight, (grown > 0) * np.float32(self.threshold))
        mask = cv2.dilate(mask, self._small)
        if roi is not None:
            # Everything outside the screen is someone's red jumper, a wooden
            # shelf or a standby LED - none of it can be a laser *on the
            # screen*, and excluding it stops it outscoring the real spot.
            mask = cv2.bitwise_and(mask, roi)
        self.mask = mask

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        self.candidates = max(0, n - 1)
        if n <= 1:
            return None

        best: Optional[Detection] = None
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < s.min_area or area > s.max_area:
                continue
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # Banding arrives as long thin streaks and glare as ragged patches;
            # a laser spot is compact and fills its bounding box.
            aspect = max(w / float(h), h / float(w))
            fill = area / float(w * h)
            if aspect > s.max_aspect or fill < s.min_fill:
                continue
            sub_w = weight[y0:y0 + h, x0:x0 + w].copy()
            sub_w[labels[y0:y0 + h, x0:x0 + w] != i] = 0.0
            m = cv2.moments(sub_w, binaryImage=False)
            if m["m00"] <= 0:
                continue
            cx = x0 + m["m10"] / m["m00"]
            cy = y0 + m["m01"] / m["m00"]
            score = float(m["m00"])
            if best is None or score > best.score:
                best = Detection(cx, cy, area, score, float(np.sqrt(area / np.pi)))
        return best
