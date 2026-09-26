"""Screen geometry helpers."""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Tuple

DEFAULT_SIZE = (1920, 1080)


def enable_dpi_awareness() -> None:
    """On Windows, ask for real pixels instead of DPI-scaled ones.

    Without this a 1920x1080 display at 150% scaling reports 1280x720, and the
    fullscreen window is stretched - every drawn point lands in the wrong
    place relative to the calibration.
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _from_win32() -> Tuple[int, int] | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return None


def _from_tkinter() -> Tuple[int, int] | None:
    try:
        import tkinter as tk
    except Exception:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return None


def _from_xrandr() -> Tuple[int, int] | None:
    try:
        out = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return None
    # Prefer the mode marked with '*' (currently active).
    m = re.search(r"^\s*(\d+)x(\d+)\s+[\d.]+\*", out, re.MULTILINE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def detect_screen_size() -> Tuple[int, int]:
    """Best-effort desktop resolution, falling back to 1920x1080."""
    for probe in (_from_win32, _from_tkinter, _from_xrandr):
        size = probe()
        if size and size[0] > 0 and size[1] > 0:
            return size
    return DEFAULT_SIZE
