"""Radial lens distortion.

A wide-angle webcam bows straight lines outwards ("barrel" distortion), and a
homography cannot express that - it maps straight lines to straight lines by
construction. So the distortion is removed first, and the homography is fitted
to the straightened points.

The coefficients are recovered from the calibration points themselves: the
nine-or-more laser samples are known to lie on a regular grid on a plane, so
the amount of bowing that makes them fit a single homography best *is* the
lens distortion. No chessboard, no separate calibration run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class LensModel:
    """OpenCV's radial model, k1/k2 only - enough for a webcam."""

    k1: float = 0.0
    k2: float = 0.0
    width: int = 0
    height: int = 0

    @property
    def identity(self) -> bool:
        return self.k1 == 0.0 and self.k2 == 0.0

    @property
    def K(self) -> np.ndarray:
        f = self.width / 2.0            # fixed: k1/k2 absorb the true focal length
        return np.array([[f, 0, self.width / 2.0],
                         [0, f, self.height / 2.0],
                         [0, 0, 1]], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        return np.array([self.k1, self.k2, 0.0, 0.0], dtype=np.float64)

    # -- points -------------------------------------------------------------
    def undistort(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        """Observed camera pixels -> straightened pixels. Shape (N, 2)."""
        src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.identity:
            return src.reshape(-1, 2).copy()
        return cv2.undistortPoints(src, self.K, self.D, P=self.K).reshape(-1, 2)

    def distort(self, pts: Sequence[Sequence[float]]) -> np.ndarray:
        """Straightened pixels -> where they actually appear in the raw frame."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.identity:
            return p.copy()
        K = self.K
        f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        x = (p[:, 0] - cx) / f
        y = (p[:, 1] - cy) / f
        r2 = x * x + y * y
        scale = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack([x * scale * f + cx, y * scale * f + cy], axis=1)

    # -- images -------------------------------------------------------------
    def maps(self, size: Tuple[int, int]):
        """remap() tables that straighten a frame of `size`.

        `size` may be smaller than the camera frame - the intrinsics are scaled
        to match, so a preview can be downscaled first and straightened at that
        cheaper resolution.
        """
        if self.identity:
            return None
        K = self.K
        scale = size[0] / float(self.width)
        if scale != 1.0:
            K = K.copy()
            K[:2, :] *= scale
        return cv2.initUndistortRectifyMap(K, self.D, None, K, size, cv2.CV_16SC2)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"k1": float(self.k1), "k2": float(self.k2),
                "width": int(self.width), "height": int(self.height)}

    @staticmethod
    def from_dict(d: Optional[dict]) -> Optional["LensModel"]:
        return LensModel(**d) if d else None


def _fit_error(lens: LensModel, cam_pts, screen_pts) -> Tuple[float, Optional[np.ndarray]]:
    """Mean reprojection error (screen px) of the best homography for `lens`."""
    src = lens.undistort(cam_pts).reshape(-1, 1, 2)
    dst = np.asarray(screen_pts, dtype=np.float64).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, 0)          # all points, no RANSAC
    if H is None:
        return float("inf"), None
    err = float(np.mean(np.linalg.norm(cv2.perspectiveTransform(src, H) - dst, axis=2)))
    return err, H


def fit(cam_pts, screen_pts, camera_size: Tuple[int, int],
        max_k1: float = 0.8, improvement: float = 0.15):
    """Search k1/k2 for the smallest reprojection error.

    For any candidate distortion the best homography follows from a linear
    solve, so only the two distortion coefficients are actually searched:
    a coarse sweep followed by three refinement passes around the winner.

    Returns (lens, error_with_lens, error_without_lens). `lens` is None when
    correcting does not meaningfully beat the plain homography - a nearly
    distortion-free camera, or too few points to tell.
    """
    w, h = camera_size
    base_lens = LensModel(0.0, 0.0, w, h)
    base_err, _ = _fit_error(base_lens, cam_pts, screen_pts)

    if len(cam_pts) < 8:                 # 8 unknowns in H alone - nothing to spare
        return None, base_err, base_err

    k1_lo, k1_hi = -max_k1, max_k1 / 2
    k2_lo, k2_hi = -0.4, 0.4
    best = (base_err, 0.0, 0.0)
    for level in range(4):
        n = 25 if level == 0 else 11
        for k1 in np.linspace(k1_lo, k1_hi, n):
            for k2 in np.linspace(k2_lo, k2_hi, n):
                err, _ = _fit_error(LensModel(float(k1), float(k2), w, h),
                                    cam_pts, screen_pts)
                if err < best[0]:
                    best = (err, float(k1), float(k2))
        span1 = (k1_hi - k1_lo) / (n - 1) * 2
        span2 = (k2_hi - k2_lo) / (n - 1) * 2
        k1_lo, k1_hi = best[1] - span1, best[1] + span1
        k2_lo, k2_hi = best[2] - span2, best[2] + span2

    err, k1, k2 = best
    if abs(k1) > max_k1 or err > base_err * (1.0 - improvement):
        return None, base_err, base_err
    return LensModel(k1, k2, w, h), err, base_err
