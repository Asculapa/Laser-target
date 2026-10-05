"""All on-screen drawing.

Everything rendered here is deliberately free of red: the camera is looking
at this very screen, so red graphics would feed straight back into the
detector.

What counts is the red the *camera* sees, which is more than the screen
sends: measured through a webcam, small text in plain yellow or magenta was
taken for the laser in every frame, and even white and grey came out faintly
red. So the colours below are all held a little short on red - the yellow is
nearer lime, the white and greys slightly cool.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

CYAN = (255, 220, 60)
GREEN = (120, 255, 120)
YELLOW = (60, 255, 190)
WHITE = (255, 255, 220)
GREY = (140, 140, 118)
DIM = (70, 70, 60)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# From OpenCV 5 on, putText draws with a real font that has more than ASCII
# in it - Cyrillic, for one. Before that, anything else came out as "???",
# which is why the Ukrainian interface needs opencv-python 5 or later.
UNICODE = int(cv2.__version__.split(".")[0]) >= 5


def blank(size: Tuple[int, int]) -> np.ndarray:
    return np.zeros((size[1], size[0], 3), dtype=np.uint8)


def text(img, s, org, scale=0.6, color=WHITE, thickness=1, shadow=True):
    if shadow:
        # The same weight as the text: a heavier shadow is also a wider one,
        # and its last letters would show beyond the end of the word.
        cv2.putText(img, s, (org[0] + 2, org[1] + 2), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_target(img, x: float, y: float, t: float, strength: float = 1.0,
                color=CYAN, crosshair: bool = True) -> None:
    """Animated reticle at screen position (x, y). `strength` fades it out."""
    h, w = img.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    c = tuple(int(v * strength) for v in color)

    if crosshair:
        cv2.line(img, (0, iy), (w, iy), tuple(int(v * 0.25 * strength) for v in color), 1, cv2.LINE_AA)
        cv2.line(img, (ix, 0), (ix, h), tuple(int(v * 0.25 * strength) for v in color), 1, cv2.LINE_AA)

    pulse = 1.0 + 0.12 * math.sin(t * 5.0)
    for radius, thick in ((int(34 * pulse), 2), (int(52 * pulse), 1)):
        cv2.circle(img, (ix, iy), radius, c, thick, cv2.LINE_AA)

    # Rotating corner ticks.
    spin = t * 1.2
    for k in range(4):
        a = spin + k * math.pi / 2
        r0, r1 = 14, 26
        p0 = (int(ix + r0 * math.cos(a)), int(iy + r0 * math.sin(a)))
        p1 = (int(ix + r1 * math.cos(a)), int(iy + r1 * math.sin(a)))
        cv2.line(img, p0, p1, c, 2, cv2.LINE_AA)

    cv2.circle(img, (ix, iy), 3, tuple(int(v * strength) for v in WHITE), -1, cv2.LINE_AA)


# -- the games' pointers ------------------------------------------------------
# Each drawn twice: dark and a little wider first, then in colour, so that it
# stands out on a bright picture as well as on a dark one. `kick` (0-1) is how
# lately the gun went off.

def _crosshair(img, x: float, y: float, colour) -> None:
    h, w = img.shape[:2]
    faint = tuple(int(v * 0.2) for v in colour)
    cv2.line(img, (0, int(y)), (w, int(y)), faint, 1, cv2.LINE_AA)
    cv2.line(img, (int(x), 0), (int(x), h), faint, 1, cv2.LINE_AA)


def draw_sight(img, x: float, y: float, colour, k: float = 1.0, kick: float = 0.0,
               crosshair: bool = False) -> None:
    """A gun sight: a ring, four posts that leave the middle clear to see
    the target through, and a bead. Kicks open, with a flash, as it fires."""
    if crosshair:
        _crosshair(img, x, y, colour)
    r = 30 * k * (1.0 + 0.45 * kick)
    c = (int(x), int(y))
    posts = [((int(x + dx * r * 0.42), int(y + dy * r * 0.42)),
              (int(x + dx * r * 1.55), int(y + dy * r * 1.55)))
             for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    for pen, extra in (((0, 0, 0), 2), (colour, 0)):
        cv2.circle(img, c, int(r), pen, 2 + extra, cv2.LINE_AA)
        for p0, p1 in posts:
            cv2.line(img, p0, p1, pen, 2 + extra, cv2.LINE_AA)
        cv2.circle(img, c, max(2, int(3 * k)) + extra // 2, pen, -1, cv2.LINE_AA)
    if kick > 0:
        flash = tuple(int(v * kick) for v in colour)
        for i in range(8):
            a = math.pi / 8 + i * math.pi / 4
            cv2.line(img, (int(x + math.cos(a) * r * 1.15), int(y + math.sin(a) * r * 1.15)),
                     (int(x + math.cos(a) * r * 1.6), int(y + math.sin(a) * r * 1.6)),
                     flash, 2, cv2.LINE_AA)


def draw_scope(img, x: float, y: float, colour, k: float = 1.0, kick: float = 0.0,
               crosshair: bool = False) -> None:
    """A rifle scope, for the ranges: a fine cross through a ring, heavy
    posts on the ring's outside, ticks to judge by, and recoil on a shot."""
    if crosshair:
        _crosshair(img, x, y, colour)
    y = y - 10 * k * kick                          # the muzzle climbs, and settles
    r = 44 * k
    c = (int(x), int(y))
    for pen, extra in (((0, 0, 0), 2), (colour, 0)):
        cv2.circle(img, c, int(r), pen, 2 + extra, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # thin inside the ring, thick outside it
            cv2.line(img, (int(x + dx * r * 0.12), int(y + dy * r * 0.12)),
                     (int(x + dx * r), int(y + dy * r)), pen, 1 + extra, cv2.LINE_AA)
            cv2.line(img, (int(x + dx * r * 0.7), int(y + dy * r * 0.7)),
                     (int(x + dx * r * 1.35), int(y + dy * r * 1.35)), pen, 3 + extra, cv2.LINE_AA)
            for m in (0.33, 0.55):                 # the ticks
                px, py = x + dx * r * m, y + dy * r * m
                cv2.line(img, (int(px - dy * 4 * k), int(py - dx * 4 * k)),
                         (int(px + dy * 4 * k), int(py + dx * 4 * k)), pen, 1 + extra, cv2.LINE_AA)
    if kick > 0:
        cv2.circle(img, c, int(r * (1.1 + 0.5 * (1 - kick))),
                   tuple(int(v * kick * 0.8) for v in colour), 2, cv2.LINE_AA)


def draw_glow(img, x: float, y: float, colour, k: float = 1.0, t: float = 0.0) -> None:
    """A soft spot of light, for the games that are not about shooting:
    a balloon is picked, a lamp is lit, nothing is fired at."""
    breathe = 1.0 + 0.08 * math.sin(t * 4.0)
    h, w = img.shape[:2]
    r = int(60 * k * breathe)
    x0, y0, x1, y1 = max(0, int(x) - r), max(0, int(y) - r), min(w, int(x) + r), min(h, int(y) + r)
    if x1 > x0 and y1 > y0:
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / r
        a = np.clip(1.0 - d, 0.0, 1.0) ** 1.4 * 0.85
        roi = img[y0:y1, x0:x1].astype(np.float32)
        roi += (np.array(colour, np.float32) - roi) * a[..., None]
        img[y0:y1, x0:x1] = roi.astype(np.uint8)
    cv2.circle(img, (int(x), int(y)), int(13 * k * breathe), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), int(13 * k * breathe), colour, 2, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), max(3, int(5 * k)), WHITE, -1, cv2.LINE_AA)


def draw_marker(img, x: float, y: float, progress: float, t: float) -> None:
    """Calibration marker: a bull's-eye with a progress ring while holding."""
    ix, iy = int(round(x)), int(round(y))
    cv2.circle(img, (ix, iy), 46, DIM, 1, cv2.LINE_AA)
    cv2.circle(img, (ix, iy), 30, WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (ix - 60, iy), (ix + 60, iy), GREY, 1, cv2.LINE_AA)
    cv2.line(img, (ix, iy - 60), (ix, iy + 60), GREY, 1, cv2.LINE_AA)
    cv2.circle(img, (ix, iy), 5, WHITE, -1, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(img, (ix, iy), (46, 46), -90, 0, 360 * min(progress, 1.0), GREEN, 3, cv2.LINE_AA)
    else:
        glow = 0.5 + 0.5 * math.sin(t * 4)
        cv2.circle(img, (ix, iy), 56, tuple(int(v * glow) for v in CYAN), 1, cv2.LINE_AA)


def text_centered(img, s, cy, scale=0.7, color=WHITE, thickness=1) -> int:
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thickness)
    text(img, s, ((img.shape[1] - tw) // 2, cy), scale, color, thickness)
    return tw


def text_fit(img, s, centre, max_w: float, max_h: float, color=WHITE,
             thickness: int = 2) -> None:
    """Text centred on `centre`, as large as fits in a max_w x max_h box."""
    (tw, th), _ = cv2.getTextSize(s, FONT, 1.0, thickness)
    scale = min(max_w / tw, max_h / th)
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, thickness)
    text(img, s, (int(centre[0] - tw / 2), int(centre[1] + th / 2)), scale, color, thickness)


def draw_panel(img, lines, cy: float) -> None:
    """Stack of (text, scale, colour, thickness) lines centred on row `cy`,
    over a dark backdrop so it stays readable on any background."""
    sizes = [cv2.getTextSize(t, FONT, sc, th)[0] for t, sc, _, th in lines]
    gap = 16
    total = sum(h for _, h in sizes) + gap * (len(lines) - 1)
    width = max(w for w, _ in sizes) + 60
    x0 = (img.shape[1] - width) // 2
    y0 = int(cy - total / 2) - 24

    box = img.copy()
    cv2.rectangle(box, (x0, y0), (x0 + width, y0 + total + 48), (18, 18, 18), -1)
    cv2.addWeighted(box, 0.75, img, 0.25, 0, img)

    y = y0 + 24
    for (t, scale, colour, thick), (_, h) in zip(lines, sizes):
        y += h
        text_centered(img, t, y, scale, colour, thick)
        y += gap


def draw_done_markers(img, points, colour=DIM) -> None:
    for px, py in points:
        cv2.circle(img, (int(px), int(py)), 8, colour, 1, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 2, colour, -1, cv2.LINE_AA)


class Trail:
    """Fading path of recent laser positions."""

    def __init__(self, seconds: float = 1.2, maxlen: int = 256) -> None:
        self.seconds = seconds
        self.points: Deque[Tuple[float, float, float]] = deque(maxlen=maxlen)

    def add(self, x: float, y: float) -> None:
        self.points.append((x, y, time.time()))

    def clear(self) -> None:
        self.points.clear()

    def draw(self, img, color=GREEN) -> None:
        now = time.time()
        pts = [p for p in self.points if now - p[2] <= self.seconds]
        for i in range(1, len(pts)):
            age = (now - pts[i][2]) / self.seconds
            fade = max(0.0, 1.0 - age)
            c = tuple(int(v * fade) for v in color)
            cv2.line(img, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                     (int(pts[i][0]), int(pts[i][1])), c, max(1, int(3 * fade)), cv2.LINE_AA)


def preview_size(canvas_width: int, frame_shape, mode: int) -> Tuple[int, int]:
    """Pixel size of the preview panel's camera view."""
    h, w = frame_shape[:2]
    width = 420 if mode == 1 else max(420, int(canvas_width * 0.42))
    return width, int(h * width / float(w))


def draw_preview(img, frame: np.ndarray, mask: Optional[np.ndarray],
                 point: Optional[Tuple[float, float]], quad: Optional[np.ndarray],
                 mode: int = 2, camera_label: str = "", maps=None) -> None:
    """Live camera view, inset top-right.

    mode 1 = narrow, with the detector mask stacked underneath (for tuning)
    mode 2 = wide camera view only (for aiming the camera at the screen)

    Shown in greyscale on purpose: the camera is pointed at this screen, and a
    colour preview would feed its own red pixels back into the detector.
    """
    if mode <= 0:
        return
    h, w = frame.shape[:2]
    width, height = preview_size(img.shape[1], frame.shape, mode)
    scale = width / float(w)

    # Downscale first, then straighten: same picture, a third of the work.
    view = cv2.resize(frame, (width, height))
    if maps is not None:
        view = cv2.remap(view, maps[0], maps[1], cv2.INTER_LINEAR)
    view = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    if quad is not None:
        cv2.polylines(view, [(quad * scale).astype(np.int32)], True, CYAN, 1, cv2.LINE_AA)
    if point is not None:
        c = (int(point[0] * scale), int(point[1] * scale))
        cv2.circle(view, c, 12, GREEN, 2, cv2.LINE_AA)
        cv2.line(view, (c[0] - 20, c[1]), (c[0] + 20, c[1]), GREEN, 1, cv2.LINE_AA)
        cv2.line(view, (c[0], c[1] - 20), (c[0], c[1] + 20), GREEN, 1, cv2.LINE_AA)

    panel = view
    if mode == 1 and mask is not None:
        m = cv2.resize(mask, (width, height))
        if maps is not None:
            m = cv2.remap(m, maps[0], maps[1], cv2.INTER_NEAREST)
        panel = np.vstack([view, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)])

    ph, pw = panel.shape[:2]
    x0 = img.shape[1] - pw - 16
    y0 = 16
    if x0 < 0 or y0 + ph > img.shape[0]:
        return

    # Slightly translucent, so a reticle underneath the panel still shows.
    region = img[y0:y0 + ph, x0:x0 + pw]
    cv2.addWeighted(panel, 0.88, region, 0.12, 0, region)
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), GREY, 1)

    label = camera_label or "камера"
    if mode == 1:
        label += "  /  маска детектора"
    text(img, label, (x0 + 8, y0 + 20), 0.5, YELLOW)
    if quad is None:
        text(img, "не відкалібровано — наведи камеру так, щоб було видно весь екран",
             (x0 + 8, y0 + ph - 12), 0.45, YELLOW)


def draw_camera_picker(img, cams, highlight: int, current: Optional[int],
                       frame: Optional[np.ndarray], message: str = "",
                       can_cancel: bool = True) -> list:
    """Full-screen camera chooser: a list on the left, a live greyscale view
    of the highlighted camera on the right.

    Returns [(x0, y0, x1, y1), ...] - one clickable box per list entry.
    """
    h, w = img.shape[:2]
    text(img, "Обери камеру", (60, 90), 1.2, CYAN, 2)
    hint = "1–9 або клік — переглянути    ENTER або ще клік — вибрати    N — наступна"
    if can_cancel:
        hint += "    ESC — назад"
    text(img, hint, (60, 130), 0.6, GREY)

    boxes = []
    list_w = min(620, w // 2 - 80)
    y = 190
    for i, cam in enumerate(cams):
        box = (50, y - 34, 50 + list_w, y + 16)
        boxes.append(box)
        chosen = i == highlight
        if chosen:
            cv2.rectangle(img, box[:2], box[2:], (60, 50, 20), -1)
            cv2.rectangle(img, box[:2], box[2:], CYAN, 1)
        key = str(i + 1) if i < 9 else " "
        text(img, key, (64, y), 0.8, YELLOW, 2)
        name = cam.name if len(cam.name) <= 40 else cam.name[:39] + "..."
        text(img, name, (100, y), 0.7, WHITE if chosen else GREY, 2 if chosen else 1)
        tags = []
        if not cam.external:
            tags.append("вбудована")
        if cam.index == current:
            tags.append("зараз працює")
        if tags:
            (tw, _), _ = cv2.getTextSize(name, FONT, 0.7, 2 if chosen else 1)
            text(img, "  (" + ", ".join(tags) + ")", (100 + tw, y), 0.55, GREY)
        y += 60
    if not cams:
        text(img, "камер не знайдено — під'єднай камеру й натисни N, щоб пошукати знову",
             (60, y), 0.7, YELLOW)

    # Live view of the highlighted camera.
    px0, py0 = 60 + list_w + 40, 170
    pw = w - px0 - 60
    if pw > 200:
        if frame is not None:
            fh, fw = frame.shape[:2]
            ph = int(fh * pw / float(fw))
            if py0 + ph > h - 60:
                ph = h - 60 - py0
                pw = int(fw * ph / float(fh))
            view = cv2.resize(frame, (pw, ph))
            view = cv2.cvtColor(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            img[py0:py0 + ph, px0:px0 + pw] = view
            cv2.rectangle(img, (px0 - 1, py0 - 1), (px0 + pw, py0 + ph), GREY, 1)
            text(img, f"{fw}x{fh}", (px0 + 8, py0 + 22), 0.5, YELLOW)
        else:
            text(img, message or "відкриваю камеру...", (px0, py0 + 40), 0.7, YELLOW)
    if frame is not None and message:
        text(img, message, (60, h - 50), 0.65, YELLOW)
    return boxes


HELP_LINES = [
    ("G", "ігри              P  пауза"),
    ("M", "миша замість лазера"),
    ("K", "автокалібрування (лазер не потрібен)"),
    ("C", "калібрування лазером"),
    ("SPACE", "зафіксувати точку вручну (під час калібрування)"),
    ("B", "на точку назад (під час калібрування)"),
    ("ESC", "скасувати калібрування"),
    ("F", "не зважати на те, що зараз на екрані (спершу відведи лазер)"),
    ("D", "перегляд камери: широкий / з маскою / вимк."),
    ("V", "вибір камери     N  наступна камера"),
    ("U", "корекція об'єктива увімк./вимк."),
    ("T", "слід увімк./вимк.      X  перехрестя увімк./вимк."),
    ("[ ]", "чутливість -/+"),
    (", .", "поріг червоного -/+"),
    ("E / R", "експозиція: темніше / світліше    A  автоекспозиція"),
    ("S", "зберегти налаштування      H  ця довідка      Q  вихід"),
]


def draw_help(img) -> None:
    h, w = img.shape[:2]
    widest = max(cv2.getTextSize(desc, FONT, 0.55, 1)[0][0] for _, desc in HELP_LINES)
    bw, bh = 110 + widest + 30, 40 + 28 * len(HELP_LINES)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), CYAN, 1)
    text(img, "Лазерний тир — клавіші", (x0 + 20, y0 + 28), 0.7, CYAN)
    for i, (key, desc) in enumerate(HELP_LINES):
        y = y0 + 58 + i * 28
        text(img, key.ljust(7), (x0 + 24, y), 0.55, YELLOW)
        text(img, desc, (x0 + 110, y), 0.55, WHITE)
