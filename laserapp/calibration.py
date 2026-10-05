"""Camera -> screen mapping and the interactive calibration state machine.

The mapping is a homography: the camera sees the (planar) screen from an
angle, so a 3x3 projective transform is exactly the right model. It is solved
from the laser positions recorded while the user points at known on-screen
markers, which means it folds in camera pose, lens scaling and any projector
keystone in one step.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection, DetectorSettings
from .lens import LensModel, fit as fit_lens


@dataclass
class Calibration:
    H: np.ndarray                      # camera px -> screen px
    camera_size: Tuple[int, int]
    screen_size: Tuple[int, int]
    error: float = 0.0                 # mean reprojection error, screen px
    created: float = 0.0
    points: Optional[List[List[float]]] = None   # screen-space markers used
    lens: Optional[LensModel] = None   # radial distortion removed before H
    error_no_lens: float = 0.0         # what the error would be without it
    dropped: int = 0                   # captures discarded as laser slips
    worst: float = 0.0                 # largest residual kept, screen px

    # -- use ----------------------------------------------------------------
    def to_screen(self, x: float, y: float) -> Tuple[float, float]:
        pt = [[float(x), float(y)]]
        if self.lens is not None:
            pt = self.lens.undistort(pt)
        out = cv2.perspectiveTransform(np.array(pt, dtype=np.float64).reshape(1, -1, 2),
                                       self.H)[0][0]
        return float(out[0]), float(out[1])

    def screen_quad(self, distorted: bool = True) -> np.ndarray:
        """The screen rectangle mapped back into camera pixels.

        `distorted=True` gives raw-frame coordinates (bowed, as the camera
        actually sees it); False gives them in the straightened image.
        """
        w, h = self.screen_size
        corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
        quad = cv2.perspectiveTransform(corners, np.linalg.inv(self.H))[0]
        if distorted and self.lens is not None:
            # Sample along the edges: they are straight only once corrected.
            dense = []
            for i in range(4):
                a, b = quad[i], quad[(i + 1) % 4]
                for t in np.linspace(0, 1, 24, endpoint=False):
                    dense.append(a + (b - a) * t)
            return self.lens.distort(np.array(dense))
        return quad

    def in_bounds(self, sx: float, sy: float, margin: float = 0.05) -> bool:
        w, h = self.screen_size
        mx, my = w * margin, h * margin
        return -mx <= sx <= w + mx and -my <= sy <= h + my

    def camera_roi(self, camera_size: Tuple[int, int], margin: int = 0) -> np.ndarray:
        """Filled mask of the screen as the camera sees it, grown by `margin`.

        The default is the exact screen outline: growing it by even a dozen
        pixels lets the bezel in, and a standby LED sitting there reads as a
        small bright red spot.
        """
        quad = self.screen_quad(distorted=True)
        mask = np.zeros((camera_size[1], camera_size[0]), np.uint8)
        cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
        if margin > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1,) * 2)
            mask = cv2.dilate(mask, k)
        return mask

    # -- persistence --------------------------------------------------------
    def save(self, path: Path, detector: Optional[DetectorSettings] = None) -> None:
        # Everything is coerced to plain Python numbers: sizes and residuals
        # often arrive as numpy scalars, which json cannot serialise.
        data = {
            "homography": np.asarray(self.H, dtype=float).tolist(),
            "camera_size": [int(v) for v in self.camera_size],
            "screen_size": [int(v) for v in self.screen_size],
            "error": float(self.error),
            "created": float(self.created or time.time()),
            "points": [[float(x), float(y)] for x, y in (self.points or [])] or None,
            "lens": self.lens.to_dict() if self.lens else None,
            "error_no_lens": float(self.error_no_lens),
            "dropped": int(self.dropped),
            "worst": float(self.worst),
        }
        if detector is not None:
            data["detector"] = detector.__dict__
        path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path: Path) -> Tuple[Optional["Calibration"], Optional[DetectorSettings]]:
        if not path.exists():
            return None, None
        try:
            data = json.loads(path.read_text())
            calib = Calibration._parse(data)
        except Exception as exc:
            print(f"[warn] ignoring {path}: {exc}")
            return None, None
        return calib, Calibration._parse_detector(data.get("detector"))

    @staticmethod
    def _parse_detector(saved) -> Optional[DetectorSettings]:
        """Keep only fields this version still has.

        Settings saved before the detector moved to local contrast describe
        absolute thresholds that no longer mean the same thing - carrying them
        over would re-create the very problem that change fixed - so they are
        dropped wholesale unless the file is from the current scheme.
        """
        if not isinstance(saved, dict) or "sensitivity" not in saved:
            return None
        known = {f.name for f in dataclasses.fields(DetectorSettings)}
        return DetectorSettings(**{k: v for k, v in saved.items() if k in known})

    @staticmethod
    def _parse(data: dict) -> "Calibration":
        data = json.loads(json.dumps(data))
        return Calibration(
            H=np.array(data["homography"], dtype=np.float64),
            camera_size=tuple(data["camera_size"]),
            screen_size=tuple(data["screen_size"]),
            error=data.get("error", 0.0),
            created=data.get("created", 0.0),
            points=data.get("points"),
            lens=LensModel.from_dict(data.get("lens")),
            error_no_lens=data.get("error_no_lens", 0.0),
            dropped=data.get("dropped", 0),
            worst=data.get("worst", 0.0),
        )


def solve(cam_pts: Sequence[Sequence[float]],
          screen_pts: Sequence[Sequence[float]],
          camera_size: Tuple[int, int],
          screen_size: Tuple[int, int],
          lens: bool = True,
          lens_model: Optional[LensModel] = None) -> Calibration:
    """Fit lens distortion (optional) and the camera -> screen homography.

    `lens_model` reuses a distortion already measured (from the chessboard
    step) instead of fitting a new one - far better than re-deriving it from a
    handful of laser samples.

    Fitted twice: once on everything, then again without any point whose
    residual is wildly out of line with the rest - that is a capture where the
    laser slipped off the marker, and a plain least-squares fit would smear its
    error across the whole screen. RANSAC with a fixed pixel threshold is the
    usual tool here, but it cannot tell a slipped capture from the large,
    perfectly legitimate residuals a distorted lens produces before correction,
    so the cut-off is taken from the median residual instead.
    """
    cam = np.array(cam_pts, dtype=np.float64).reshape(-1, 2)
    screen = np.array(screen_pts, dtype=np.float64).reshape(-1, 2)
    keep = np.ones(len(cam), dtype=bool)
    dropped = 0

    for attempt in range(2):
        model, error, error_plain = (lens_model, 0.0, 0.0)
        if lens and lens_model is None:
            model, _, error_plain = fit_lens(cam[keep], screen[keep], camera_size)
        elif lens_model is not None:
            error_plain = _plain_error(cam[keep], screen[keep])

        src = cam[keep] if model is None else model.undistort(cam[keep])
        H, _ = cv2.findHomography(src.reshape(-1, 1, 2),
                                  screen[keep].reshape(-1, 1, 2), 0)
        if H is None:
            raise RuntimeError("Homography could not be solved - points are degenerate")

        all_src = cam if model is None else model.undistort(cam)
        residual = np.linalg.norm(
            cv2.perspectiveTransform(all_src.reshape(-1, 1, 2), H).reshape(-1, 2) - screen,
            axis=1)
        error = float(residual[keep].mean())
        if model is None:
            error_plain = error

        if attempt == 0 and len(cam) >= 10:
            limit = max(6.0, 5.0 * float(np.median(residual[keep])))
            outliers = residual > limit
            if outliers.any() and (~outliers).sum() >= 8:
                keep = ~outliers
                dropped = int(outliers.sum())
                continue
        break

    return Calibration(
        H=H,
        camera_size=camera_size,
        screen_size=screen_size,
        error=error,
        created=time.time(),
        points=[[float(p[0]), float(p[1])] for p in screen],
        lens=model,
        error_no_lens=error_plain,
        dropped=dropped,
        worst=float(residual[keep].max()),
    )


def _plain_error(cam: np.ndarray, screen: np.ndarray) -> float:
    """Mean error of a homography fitted to the raw, uncorrected points."""
    H, _ = cv2.findHomography(cam.reshape(-1, 1, 2), screen.reshape(-1, 1, 2), 0)
    if H is None:
        return 0.0
    projected = cv2.perspectiveTransform(cam.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - screen, axis=1)))


def grid_points(screen_size: Tuple[int, int],
                cols: int = 3, rows: int = 3,
                margin: float = 0.12) -> List[Tuple[float, float]]:
    w, h = screen_size
    xs = np.linspace(w * margin, w * (1 - margin), cols)
    ys = np.linspace(h * margin, h * (1 - margin), rows)
    return [(float(x), float(y)) for y in ys for x in xs]


class CalibrationSession:
    """Walks the user through the marker grid, auto-capturing steady points."""

    SETTLE = 0.8          # seconds of cooldown after each captured point
    HOLD_SAMPLES = 12     # consecutive steady detections needed
    HOLD_TOLERANCE = 5.0  # max spread (camera px) within those samples
    MOVE_MIN = 20.0       # camera px the laser must have moved since the last point

    def __init__(self, screen_size: Tuple[int, int], camera_size: Tuple[int, int],
                 cols: int = 3, rows: int = 3, lens: bool = True,
                 lens_model: Optional[LensModel] = None) -> None:
        self.screen_size = screen_size
        self.camera_size = camera_size
        self.lens = lens
        self.lens_model = lens_model
        self.targets = grid_points(screen_size, cols, rows)
        self.index = 0
        self.captured: List[Tuple[float, float]] = []
        self.samples: Deque[Tuple[float, float]] = deque(maxlen=self.HOLD_SAMPLES)
        self.cooldown_until = time.time() + self.SETTLE
        self.message = "Наведи лазер на мітку й тримай нерухомо"
        self.result: Optional[Calibration] = None
        self.unmoved = False      # the laser is still where the last point was taken
        self._seen_seq: Optional[int] = None

    # -- state --------------------------------------------------------------
    @property
    def current_target(self) -> Tuple[float, float]:
        return self.targets[min(self.index, len(self.targets) - 1)]

    @property
    def progress(self) -> float:
        return len(self.samples) / self.HOLD_SAMPLES

    @property
    def done(self) -> bool:
        return self.index >= len(self.targets)

    # -- input --------------------------------------------------------------
    def update(self, det: Optional[Detection], seq: Optional[int] = None) -> None:
        """Feed one detection. `seq` is the camera frame it came from: the
        screen is redrawn more often than the camera delivers, and the same
        frame seen twice is not two steady readings."""
        if self.done or time.time() < self.cooldown_until:
            self.samples.clear()
            return
        if det is None:
            self.samples.clear()
            self.unmoved = False
            return
        if seq is not None:
            if seq == self._seen_seq:
                return
            self._seen_seq = seq
        # A laser still resting on the marker just captured would be recorded
        # for this one as well, pairing it with the wrong place on screen.
        self.unmoved = bool(self.captured) and float(np.hypot(
            det.x - self.captured[-1][0], det.y - self.captured[-1][1])) < self.MOVE_MIN
        if self.unmoved:
            self.samples.clear()
            return
        self.samples.append((det.x, det.y))
        if len(self.samples) < self.HOLD_SAMPLES:
            return
        pts = np.array(self.samples, dtype=np.float64)
        centre = pts.mean(axis=0)
        if np.max(np.linalg.norm(pts - centre, axis=1)) <= self.HOLD_TOLERANCE:
            self._accept(centre)

    def force_capture(self, det: Optional[Detection]) -> bool:
        """Manual capture (SPACE) for when auto-hold is being fussy."""
        if self.done or det is None:
            return False
        if self.samples:
            pts = np.array(self.samples, dtype=np.float64)
            self._accept(pts.mean(axis=0))
        else:
            self._accept(np.array([det.x, det.y]))
        return True

    def back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.captured.pop()
        self.samples.clear()
        self.cooldown_until = time.time() + self.SETTLE

    def _accept(self, centre: np.ndarray) -> None:
        self.captured.append((float(centre[0]), float(centre[1])))
        self.index += 1
        self.samples.clear()
        self.cooldown_until = time.time() + self.SETTLE
        if self.done:
            self._finish()

    def _finish(self) -> None:
        try:
            self.result = solve(self.captured, self.targets,
                                self.camera_size, self.screen_size,
                                lens=self.lens, lens_model=self.lens_model)
            r = self.result
            extra = f", відкинуто точок: {r.dropped}" if r.dropped else ""
            if r.lens is not None:
                self.message = (f"Відкалібровано — {r.error:.1f} пкс "
                                f"(об'єктив k1={r.lens.k1:+.2f} k2={r.lens.k2:+.2f}; "
                                f"без корекції {r.error_no_lens:.1f} пкс{extra})")
            else:
                self.message = f"Відкалібровано — середня похибка {r.error:.1f} пкс{extra}"
        except Exception as exc:  # degenerate point set
            self.result = None
            self.message = f"Калібрування не вдалося: {exc}"
