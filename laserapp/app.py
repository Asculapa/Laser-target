"""Laser Target - point a red laser at the screen, a reticle follows it."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import autocal, overlay, sound
from .game import HighScores, Round
from .menu import Factory, GameMenu
from .calibration import Calibration, CalibrationSession
from .camera import Camera, CameraInfo, list_cameras, resolve_camera
from .detector import Detection, LaserDetector
from .screen import detect_screen_size, enable_dpi_awareness

WINDOW = "Laser Target"
MODE_TRACK = "track"
MODE_CALIB = "calibrate"
MODE_AUTO = "auto"
MODE_GAME = "game"
MODE_MENU = "menu"
MODE_PICK = "pick"

# A camera picture older than this says nothing about where the laser is now.
STALE = 0.3


class LaserApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.screen_size: Tuple[int, int] = tuple(args.screen) if args.screen else detect_screen_size()

        self.calib_path = Path(args.calibration).expanduser()
        self.calibration, saved_detector = Calibration.load(self.calib_path)

        # A calibration whose own fit error is a sizeable fraction of the
        # screen is not a calibration - refuse it rather than silently sending
        # the reticle somewhere random.
        if self.calibration is not None:
            limit = max(25.0, 0.02 * float(np.hypot(*self.calibration.screen_size)))
            if self.calibration.error > limit:
                print(f"[warn] ignoring {self.calib_path}: mean error "
                      f"{self.calibration.error:.0f} px is far too large - recalibrate")
                self.calibration = None

        self.detector = LaserDetector(saved_detector)
        self.settings_path = self.calib_path.parent / "settings.json"
        self.settings = self._load_settings()

        self.mode = MODE_TRACK
        self.camera: Optional[Camera] = None
        self.camera_size: Tuple[int, int] = (args.width, args.height)
        self.picker_cams: list = []
        self.picker_sel = 0
        self.picker_boxes: list = []
        self.picker_msg = ""
        self._picker_return = MODE_TRACK
        self._picker_initial = False
        self._picker_original: Optional[CameraInfo] = None
        self._pending_pick: Optional[int] = None
        self._pending_confirm = False

        # No --camera: reuse the camera picked last time if it is still
        # plugged in, otherwise let the user choose when there is a choice.
        spec = args.camera
        chosen: Optional[Tuple[int, str]] = None
        if spec is None:
            cams = list_cameras()
            remembered = next((c for c in cams if c.name == self.settings.get("camera")), None)
            if remembered:
                chosen = remembered.index, remembered.name
            spec = "pick" if len(cams) > 1 else "auto"
        if spec == "pick" and chosen is None:
            self.camera_index, self.camera_name = -1, "none"
            self.open_picker(initial=True)
        else:
            self.camera_index, self.camera_name = chosen or resolve_camera(spec)
            self.camera = self._open_camera(self.camera_index, self.camera_name)
            self.camera_size = self.camera.size
            print(f"[camera] using [{self.camera_index}] {self.camera_name} "
                  f"at {self.camera_size[0]}x{self.camera_size[1]}")
            self._check_calibration_camera()
        self.session: Optional[CalibrationSession] = None
        self.auto: Optional[autocal.AutoCalibration] = None
        self._auto_calibrated = False
        self._started = time.time()
        self._window_size: Tuple[int, int] = (0, 0)
        self._window_since = 0.0
        self.smoothed: Optional[np.ndarray] = None
        self.last_seen = 0.0
        self.trail = overlay.Trail()
        self.show_trail = not args.no_trail
        self.show_crosshair = True
        self.preview = 0 if args.no_preview else 2   # 0 off, 1 small+mask, 2 large
        self.correct_preview = True                  # straighten it when a lens is known
        self._maps = None
        self._maps_key = None
        self._roi_mask: Optional[np.ndarray] = None
        self._roi_key = None
        self.game: Optional[Round] = None
        self.menu: Optional[GameMenu] = None
        sound.enabled = not args.no_sound
        self._game_factory: Optional[Factory] = None
        self._scores: dict = {}                      # high-score tables, by game
        self._game_rank: Optional[int] = None
        self._blind_pause = False                    # paused because the camera went blind
        self._preview_before_game = 0
        self.mouse_input = args.mouse_pointer
        self._mouse_pos = (0, 0)
        self._mouse_down = False
        self.show_help = True
        self.help_until = time.time() + 6.0
        self.verify_until = 0.0
        self.status = ""
        self.status_until = 0.0
        self.frame_seq = -1
        self.detection: Optional[Detection] = None
        self._other_dots: List[Detection] = []       # weaker spots in the same frame
        self.render_fps = 0.0
        self.mouse = self._setup_mouse() if args.mouse else None

        if self.mode == MODE_PICK:
            pass
        elif self.calibration is None:
            self.notify("No calibration found - press C to calibrate")
        else:
            self.notify(f"Calibration loaded (error {self.calibration.error:.1f} px)")

    def _load_settings(self) -> dict:
        try:
            return json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_settings(self) -> None:
        try:
            self.settings_path.write_text(json.dumps(self.settings, indent=2))
        except OSError as exc:
            print(f"[warn] cannot save {self.settings_path}: {exc}")

    def _check_calibration_camera(self) -> None:
        if self.calibration and tuple(self.calibration.camera_size) != tuple(self.camera_size):
            print(f"[warn] calibration was made at {self.calibration.camera_size}, "
                  f"camera is now {self.camera_size} - recalibrate with 'C'")

    def _roi(self, frame) -> Optional[np.ndarray]:
        """Screen area in camera pixels, rebuilt only when it changes."""
        # Not while calibrating: if the camera has moved, the old screen
        # outline is wrong and would reject the very points being captured.
        if self.calibration is None or self.mode not in (MODE_TRACK, MODE_GAME, MODE_MENU):
            return None
        size = frame.shape[1::-1]
        key = (id(self.calibration), size)
        if key != self._roi_key:
            self._roi_mask = self.calibration.camera_roi(size)
            self._roi_key = key
        return self._roi_mask

    def _lens_maps(self, lens, size):
        """remap() tables for the preview, rebuilt only when the lens changes."""
        key = (lens.k1, lens.k2, size)
        if key != self._maps_key:
            self._maps = lens.maps(size)
            self._maps_key = key
        return self._maps

    def _open_camera(self, index: int, name: str = "") -> Camera:
        return Camera(
            index=index,
            name=name,
            width=self.args.width,
            height=self.args.height,
            fps=self.args.fps,
            exposure=self.args.exposure,
        )

    def _use_camera(self, cam: CameraInfo) -> bool:
        """Open `cam` in place of the current camera. False if it won't open."""
        if self.camera is not None and cam.index == self.camera_index:
            return True
        # Release first: on Windows a device that is already open elsewhere
        # in the process refuses a second open.
        previous = CameraInfo(self.camera_index, self.camera_name, True) \
            if self.camera is not None else None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        try:
            self.camera = self._open_camera(cam.index, cam.name)
        except RuntimeError:
            self.camera_index, self.camera_name = -1, "none"
            if previous is not None:          # fall back to what was working
                self._use_camera(previous)
            return False
        self.camera_index = cam.index
        self.camera_name = cam.name
        time.sleep(0.4)                      # let the first frames arrive
        self.camera_size = self.camera.size
        self.detection = None
        self.frame_seq = -1
        return True

    def switch_camera(self) -> None:
        """Cycle to the next capture device without restarting."""
        cams = list_cameras()
        if len(cams) < 2:
            self.notify("only one camera available")
            return
        order = [c.index for c in cams]
        pos = (order.index(self.camera_index) + 1) % len(order) \
            if self.camera_index in order else 0
        self._select_camera(cams[pos])

    def _select_camera(self, cam: CameraInfo) -> None:
        """Switch to `cam` for good and remember it for next time."""
        if not self._use_camera(cam):
            self.notify(f"cannot open camera [{cam.index}] {cam.name}", 4.0)
            return
        self.settings["camera"] = cam.name
        self._save_settings()
        msg = f"camera -> [{cam.index}] {cam.name}"
        if self.calibration and tuple(self.camera_size) != tuple(self.calibration.camera_size):
            msg += "  - calibration is for another camera, press K"
        self.notify(msg, 4.0)

    # -- camera picker ------------------------------------------------------
    def open_picker(self, initial: bool = False) -> None:
        self.picker_cams = list_cameras()
        indices = [c.index for c in self.picker_cams]
        if self.camera_index in indices:
            self.picker_sel = indices.index(self.camera_index)
        else:
            auto = next((i for i, c in enumerate(self.picker_cams) if c.external), 0)
            self.picker_sel = auto
        self.picker_msg = ""
        self._picker_original = CameraInfo(self.camera_index, self.camera_name, True) \
            if self.camera is not None else None
        if self.mode not in (MODE_PICK, MODE_AUTO, MODE_CALIB):
            self._picker_return = self.mode if self.mode != MODE_GAME else MODE_TRACK
        if self.mode == MODE_GAME:
            self.end_game()
        self.mode = MODE_PICK
        self._picker_initial = initial
        self.show_help = False
        self._picker_preview()

    def _picker_preview(self) -> None:
        """Open the highlighted camera so its live view shows in the picker."""
        if not self.picker_cams:
            return
        cam = self.picker_cams[self.picker_sel]
        self.picker_msg = ""
        if not self._use_camera(cam):
            self.picker_msg = f"cannot open {cam.name} - is another program using it?"

    def picker_move(self, sel: int) -> None:
        if not self.picker_cams:
            return
        sel %= len(self.picker_cams)
        if sel != self.picker_sel or self.camera is None:
            self.picker_sel = sel
            self._picker_preview()

    def picker_confirm(self) -> None:
        if not self.picker_cams or self.camera is None:
            self.picker_msg = self.picker_msg or "no working camera selected"
            return
        cam = self.picker_cams[self.picker_sel]
        self.mode = self._picker_return
        self._select_camera(cam)
        print(f"[camera] using [{self.camera_index}] {self.camera_name} "
              f"at {self.camera_size[0]}x{self.camera_size[1]}")
        self._check_calibration_camera()
        if self._picker_initial:
            self.show_help = True
            self.help_until = time.time() + 6.0

    def step_picker(self, canvas) -> None:
        frame = None
        if self.camera is not None:
            frame, _ = self.camera.read()
        self.picker_boxes = overlay.draw_camera_picker(
            canvas, self.picker_cams, self.picker_sel,
            self._picker_original.index if self._picker_original else None, frame,
            self.picker_msg, can_cancel=not self._picker_initial)

    def handle_picker_key(self, key: int) -> bool:
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if key in (13, 10):                        # ENTER
            self.picker_confirm()
        elif ch.isdigit() and ch != "0":
            self.picker_move(int(ch) - 1)
        elif ch == "n" or key == 9:                # N or TAB
            if not self.picker_cams:
                self.picker_cams = list_cameras()
                self.picker_sel = 0
                self._picker_preview()
            else:
                self.picker_move(self.picker_sel + 1)
        elif key == 27 and not self._picker_initial:   # ESC: back to the old camera
            original = self._picker_original
            if original is None or not self._use_camera(original):
                self.picker_msg = "no camera open - choose one"
                return True
            self.mode = self._picker_return
        elif ch == "q":
            return False
        return True

    # -- helpers ------------------------------------------------------------
    def _setup_mouse(self):
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            return pyautogui
        except Exception:
            print("[warn] --mouse needs pyautogui (pip install pyautogui); continuing without")
            return None

    def notify(self, msg: str, seconds: float = 3.0) -> None:
        self.status = msg
        self.status_until = time.time() + seconds

    def pointer(self) -> Optional[Tuple[float, float]]:
        """Where the user is pointing, in screen pixels, or None.

        With mouse input the held left button stands in for the beam being
        lit, which is what the game's flash/dwell logic reads.
        """
        if self.mouse_input:
            return self._mouse_pos if self._mouse_down else None
        return self.map_to_screen(self.detection) if self.detection else None

    def pointers(self) -> List[Tuple[float, float]]:
        """Every dot on the screen, strongest first, for a game with two
        players. The mouse is one pointer, wherever it is."""
        if self.mouse_input or self.detection is None:
            point = self.pointer()
            return [point] if point is not None else []
        points = [self.map_to_screen(d) for d in [self.detection] + self._other_dots]
        return [p for p in points if p is not None]

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
        try:
            rect = cv2.getWindowImageRect(WINDOW)
            if rect[2] > 0 and rect[3] > 0:      # window may be scaled
                x = x * self.screen_size[0] / rect[2]
                y = y * self.screen_size[1] / rect[3]
        except Exception:
            pass
        self._mouse_pos = (float(x), float(y))
        if self.mode == MODE_PICK and event == cv2.EVENT_LBUTTONDOWN:
            # First click previews a camera, a second click on it picks it.
            for i, (x0, y0, x1, y1) in enumerate(self.picker_boxes):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if i == self.picker_sel and self.camera is not None:
                        self._pending_confirm = True
                    else:
                        self._pending_pick = i
                    break

    def map_to_screen(self, det: Detection) -> Optional[Tuple[float, float]]:
        if self.calibration is not None:
            sx, sy = self.calibration.to_screen(det.x, det.y)
            if not self.calibration.in_bounds(sx, sy, margin=self.args.bounds_margin):
                return None
            return sx, sy
        # Uncalibrated fallback: straight scaling of the camera frame.
        cw, ch = self.camera_size
        return det.x / cw * self.screen_size[0], det.y / ch * self.screen_size[1]

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self._on_mouse)
        # The fullscreen hint is only honoured once the window has been shown
        # with a frame, and a later resize/move would undo it - so paint one
        # frame, go fullscreen, and leave the geometry alone after that.
        cv2.imshow(WINDOW, overlay.blank(self.screen_size))
        cv2.waitKey(200)
        if self.args.windowed:
            cv2.resizeWindow(WINDOW, self.screen_size[0] // 2, self.screen_size[1] // 2)
        else:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        last_render = time.perf_counter()
        try:
            while True:
                # Mouse callbacks only record clicks; opening a camera from
                # inside one would stall the GUI thread mid-event.
                if self._pending_pick is not None:
                    self.picker_move(self._pending_pick)
                    self._pending_pick = None
                if self._pending_confirm:
                    self._pending_confirm = False
                    self.picker_confirm()

                if self.mode == MODE_PICK or self.camera is None:
                    if self.mode != MODE_PICK:
                        self.open_picker()
                    canvas = overlay.blank(self.screen_size)
                    self.step_picker(canvas)
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(15) & 0xFF
                    if key != 255 and not self.handle_picker_key(key):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

                frame, seq = self.camera.read()
                # A replugged camera can come back under a different number.
                self.camera_index = self.camera.index
                if frame is None:
                    # A freshly opened camera takes up to a second to deliver
                    # its first frame, and one that has stalled is being
                    # reopened; keep drawing so the screen never freezes.
                    self.detection = None
                    doing = "reconnecting" if self.camera.stalled else "starting"
                    canvas = overlay.blank(self.screen_size)
                    overlay.draw_panel(canvas, [
                        (f"{doing} camera [{self.camera_index}] {self.camera_name}",
                         0.9, overlay.CYAN, 2),
                        ("V choose camera    N next camera    Q quit", 0.6, overlay.GREY, 1),
                    ], self.screen_size[1] / 2)
                    cv2.imshow(WINDOW, canvas)
                    key = cv2.waitKey(30) & 0xFF
                    if key != 255 and not self.handle_key(key):
                        break
                    continue

                if seq != self.frame_seq:
                    self.frame_seq = seq
                    found = self.detector.detect_all(frame, self._roi(frame))
                    self.detection = found[0] if found else None
                    self._other_dots = found[1:]
                elif self.camera.age > STALE:
                    # The camera is dropping frames: better no pointer than
                    # one left standing where the laser last was.
                    self.detection = None

                if (self.calibration is None and not self._auto_calibrated
                        and not self.args.no_auto_calibrate and self.mode == MODE_TRACK
                        and self._window_settled()):
                    self._auto_calibrated = True
                    self.start_auto()

                canvas = overlay.blank(self.screen_size)
                if self.mode == MODE_AUTO:
                    # Nothing but the dot: a HUD or preview drawn over the
                    # screen would land in the camera's view of it.
                    self.step_auto(canvas, frame, seq)
                elif self.mode == MODE_GAME:
                    self.step_game(canvas)
                    self.draw_hud(canvas, frame)
                elif self.mode == MODE_MENU:
                    self.step_menu(canvas)
                    self.draw_hud(canvas, frame)
                elif self.mode == MODE_CALIB:
                    self.step_calibration(canvas)
                    self.draw_hud(canvas, frame)
                else:
                    self.step_tracking(canvas)
                    self.draw_hud(canvas, frame)
                cv2.imshow(WINDOW, canvas)

                now = time.perf_counter()
                dt = now - last_render
                last_render = now
                if dt > 0:
                    self.render_fps = 0.9 * self.render_fps + 0.1 / dt if self.render_fps else 1.0 / dt

                key = cv2.waitKey(1) & 0xFF
                if key != 255 and not self.handle_key(key):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self._close_game()
            if self.camera is not None:
                self.camera.release()
            cv2.destroyAllWindows()

    # -- games --------------------------------------------------------------
    def open_menu(self) -> None:
        if self.calibration is None and not self.mouse_input:
            self.notify("calibrate first - press K", 4.0)
            return
        if self.mode not in (MODE_GAME, MODE_MENU):
            self._preview_before_game = self.preview
            self.preview = 0            # tiles and targets need the whole screen
        self._close_game()
        self.menu = GameMenu(self.screen_size)
        self.mode = MODE_MENU
        self.show_help = False
        self.trail.clear()
        self.smoothed = None

    def step_menu(self, canvas) -> None:
        menu = self.menu
        if menu is None:
            self.mode = MODE_TRACK
            return
        t = time.time()
        point = self.pointer()
        choice = menu.update(t, point)
        menu.draw(canvas)
        if point is not None:
            overlay.draw_target(canvas, point[0], point[1], t, 1.0, crosshair=False)
        if choice is not None:
            self.start_game(choice)

    def start_game(self, factory: Optional[Factory] = None) -> None:
        """Start the chosen game, or the last one again."""
        factory = factory or self._game_factory
        if factory is None:
            self.open_menu()
            return
        self._game_factory = factory
        self.menu = None
        self._close_game()
        self.game = factory(self.screen_size)
        self.game.home = self.calib_path.parent
        self.game.start(time.time())
        self._game_rank = None
        self._blind_pause = False
        self.mode = MODE_GAME
        self.show_help = False
        self.trail.clear()
        self.smoothed = None

    def _close_game(self) -> None:
        if self.game is not None:
            self.game.close()
        self.game = None

    def end_game(self) -> None:
        self.mode = MODE_TRACK
        self._close_game()
        self.menu = None
        self.preview = self._preview_before_game
        self.smoothed = None

    def _scores_for(self, game: Round) -> Optional[HighScores]:
        name = game.scores_name
        if name is None:
            return None
        if name not in self._scores:
            self._scores[name] = HighScores(self.calib_path.parent / f"{name}.json")
        return self._scores[name]

    def step_game(self, canvas) -> None:
        game = self.game
        if game is None:
            self.mode = MODE_TRACK
            return
        t = time.time()
        point = self.pointer()
        # With no picture the laser cannot be seen, and a round left running
        # would cost lives for targets nobody could have shot. The mouse
        # pointer does not need the camera.
        blind = self.camera.age > STALE and not self.mouse_input and game.needs_sight
        if blind and game.state == "playing":
            game.toggle_pause(t)
            self._blind_pause = True
        elif self._blind_pause and not blind:
            if game.state == "paused":
                game.toggle_pause(t)
            self._blind_pause = False
        points = [point] if point is not None else []
        if game.players > 1:
            points = self.pointers()
            game.update(t, point, points)
        else:
            game.update(t, point)

        scores = self._scores_for(game)
        if game.state == "over" and self._game_rank is None:
            self._game_rank = (scores.add(game.stats, time.strftime("%d %b %H:%M"))
                               if scores is not None else None) or 0

        game.draw(canvas)
        if game.state == "over":
            game.draw_over(canvas, scores, self._game_rank or None)
        else:
            # Crosshair lines run right across the screen - through the other
            # player's half, when there is one.
            for x, y in points:
                overlay.draw_target(canvas, x, y, t, 1.0,
                                    crosshair=self.show_crosshair and game.players == 1)

    # -- tracking -----------------------------------------------------------
    def step_tracking(self, canvas) -> None:
        t = time.time()
        point = self.pointer()

        if point is not None:
            p = np.array(point, dtype=np.float64)
            a = self.args.smoothing
            self.smoothed = p if self.smoothed is None else (1 - a) * self.smoothed + a * p
            self.last_seen = t
            self.trail.add(*self.smoothed)
            if self.mouse is not None:
                try:
                    self.mouse.moveTo(int(self.smoothed[0]), int(self.smoothed[1]), _pause=False)
                except Exception:
                    pass

        if self.show_trail:
            self.trail.draw(canvas)

        if t < self.verify_until and self.calibration and self.calibration.points:
            overlay.draw_done_markers(canvas, self.calibration.points, overlay.GREEN)
            overlay.text(canvas, "Check the markers line up with your laser",
                         (40, self.screen_size[1] - 80), 0.7, overlay.GREEN)

        if self.smoothed is not None:
            age = t - self.last_seen
            hold = self.args.hold
            if age <= hold:
                strength = 1.0 if age < hold * 0.5 else max(0.0, 1.0 - (age - hold * 0.5) / (hold * 0.5))
                overlay.draw_target(canvas, self.smoothed[0], self.smoothed[1], t,
                                    strength, crosshair=self.show_crosshair)
                overlay.text(canvas, f"{int(self.smoothed[0])}, {int(self.smoothed[1])}",
                             (int(self.smoothed[0]) + 60, int(self.smoothed[1]) - 40),
                             0.6, overlay.CYAN)
            else:
                self.smoothed = None
                self.trail.clear()

    def _window_settled(self) -> bool:
        """True once the window has stopped resizing.

        Going fullscreen takes the window manager a moment, and anything shown
        before that lands somewhere else a fraction of a second later - which
        would scramble a calibration that is mid-sequence.
        """
        try:
            rect = cv2.getWindowImageRect(WINDOW)
        except Exception:
            return time.time() - self._started > 1.0
        size = (rect[2], rect[3])
        if size != self._window_size:
            self._window_size = size
            self._window_since = time.time()
            return False
        return size[0] > 0 and time.time() - self._window_since > 0.4

    # -- automatic calibration ---------------------------------------------
    def start_auto(self) -> None:
        if self.mode in (MODE_GAME, MODE_MENU):
            self.end_game()                 # puts the preview back, too
        self._window_size = (0, 0)          # re-check the window before dotting
        self.auto = autocal.AutoCalibration(
            self.screen_size, self.camera_size, lens=not self.args.no_lens)
        self.mode = MODE_AUTO
        self.show_help = False

    def step_auto(self, canvas, frame, seq) -> None:
        session = self.auto
        if session is None:
            self.mode = MODE_TRACK
            return
        if self._window_settled():
            session.update(frame, seq)
        session.render(canvas)

        # This caption is drawn identically in every frame, including the ones
        # the reference is built from, so it cancels out of the difference and
        # cannot be mistaken for the dot.
        overlay.text_centered(canvas, "calibrating - keep the camera and screen still",
                              self.screen_size[1] - 40, 0.6, overlay.GREY)

        if not session.done:
            return
        if session.result is not None:
            self.calibration = session.result
            self.camera_size = frame.shape[1::-1]
            self.calibration.save(self.calib_path, self.detector.s)
            self.notify(session.message, 7.0)
            self.verify_until = time.time() + 6.0
        else:
            self.notify(session.message + " - press K to retry, or C for the laser", 8.0)
        self.auto = None
        self.mode = MODE_TRACK

    # -- calibration --------------------------------------------------------
    def start_calibration(self) -> None:
        if self.mode in (MODE_GAME, MODE_MENU):
            self.end_game()                 # puts the preview back, too
        self.session = CalibrationSession(
            self.screen_size, self.camera_size,
            cols=self.args.grid[0], rows=self.args.grid[1],
            lens=not self.args.no_lens,
            lens_model=self.calibration.lens if self.calibration else None,
        )
        self.mode = MODE_CALIB
        self.smoothed = None
        self.trail.clear()
        self.show_help = False

    def step_calibration(self, canvas) -> None:
        s = self.session
        assert s is not None
        t = time.time()
        s.update(self.detection, self.frame_seq)

        if s.done:
            if s.result is not None:
                self.calibration = s.result
                self.calibration.save(self.calib_path, self.detector.s)
                self.notify(f"{s.message} - saved to {self.calib_path}", 5.0)
                self.verify_until = t + 6.0
            else:
                self.notify(s.message, 5.0)
            self.mode = MODE_TRACK
            self.session = None
            return

        overlay.draw_done_markers(canvas, s.targets[:s.index])
        tx, ty = s.current_target
        overlay.draw_marker(canvas, tx, ty, s.progress, t)

        if self.detection is None:
            hint = ("laser not detected - press D for the detector view, "
                    "[ ] , . to adjust, E/R for exposure", 0.7, overlay.YELLOW, 1)
        elif t < s.cooldown_until:
            hint = ("hold on...", 0.7, overlay.GREY, 1)
        elif s.unmoved:
            hint = ("captured - now move the laser to this marker", 0.7, overlay.YELLOW, 1)
        else:
            hint = (f"holding {int(s.progress * 100)}%", 0.7, overlay.GREEN, 1)

        overlay.draw_panel(canvas, [
            (f"CALIBRATION   {s.index + 1} / {len(s.targets)}", 1.0, overlay.CYAN, 2),
            ("Point the laser at the centre of the marker and hold it steady",
             0.7, overlay.WHITE, 1),
            ("SPACE capture now    B back    ESC cancel", 0.6, overlay.GREY, 1),
            hint,
        ], self._panel_row(ty))

    def _panel_row(self, marker_y: float) -> float:
        """A screen row with no markers on it: the gap between marker rows that
        lies furthest from the marker currently being aimed at."""
        s = self.session
        rows = sorted({round(p[1]) for p in s.targets}) if s else []
        h = self.screen_size[1]
        if len(rows) < 2:
            return h * 0.75 if marker_y < h * 0.5 else h * 0.25
        bands = [(rows[i] + rows[i + 1]) / 2 for i in range(len(rows) - 1)]
        return max(bands, key=lambda b: abs(b - marker_y))

    # -- hud ----------------------------------------------------------------
    def draw_hud(self, canvas, frame) -> None:
        h, w = canvas.shape[:2]
        t = time.time()
        d = self.detector.s
        bits = [
            f"[{self.camera_index}] {self.camera_name}",
            f"{self.camera_size[0]}x{self.camera_size[1]} @ {self.camera.fps:4.1f}fps",
            f"render {self.render_fps:4.1f}fps",
            f"sens {d.sensitivity:.1f} floor {d.min_redness} thr {self.detector.threshold:.0f}",
            f"blobs {self.detector.candidates}",
            "LASER" if self.detection else "no laser",
        ]
        if self.calibration:
            bits.append(f"calib err {self.calibration.error:.1f}px")
            if self.calibration.lens:
                bits.append(f"lens k1={self.calibration.lens.k1:+.2f}")
        else:
            bits.append("UNCALIBRATED")
        if self.camera.auto_exposure:
            bits.append("auto-exp")
        else:
            bits.append(f"exp {self.camera.exposure:.0f}")
        overlay.text(canvas, "   ".join(bits), (20, h - 20), 0.55,
                     overlay.GREEN if self.detection else overlay.GREY)

        if self.camera.loss > 0.05:
            overlay.text(canvas,
                         f"the camera is losing {self.camera.loss * 100:.0f}% of its frames - "
                         "check its cable, or plug it straight into the computer",
                         (20, h - 72), 0.6, overlay.YELLOW)

        # Redness is what the detector runs on, and a blown-out sensor has
        # none of it: no threshold can recover a spot the camera clipped.
        if (self.detection is None and self.detector.saturation > 0.02
                and self.mode == MODE_TRACK):
            overlay.text(canvas,
                         f"{self.detector.saturation * 100:.0f}% of the image is blown out - "
                         "press E to darken the exposure",
                         (20, h - 46), 0.6, overlay.YELLOW)

        if t < self.status_until:
            overlay.text(canvas, self.status, (20, 30), 0.7, overlay.YELLOW)

        if self.preview:
            # The wide preview would sit on top of the calibration instructions,
            # so shrink it while calibrating.
            mode = 1 if self.mode == MODE_CALIB else self.preview
            lens = self.calibration.lens if self.calibration else None
            corrected = bool(lens) and self.correct_preview
            point = (self.detection.x, self.detection.y) if self.detection else None
            maps = None
            if corrected:
                maps = self._lens_maps(
                    lens, overlay.preview_size(canvas.shape[1], frame.shape, mode))
                if maps is None:
                    corrected = False
                elif point is not None:
                    point = tuple(lens.undistort([point])[0])
            quad = self.calibration.screen_quad(distorted=not corrected) \
                if self.calibration else None
            label = f"[{self.camera_index}] {self.camera_name}"
            if lens:
                label += "  lens-corrected" if corrected else "  raw (bowed)"
            overlay.draw_preview(canvas, frame, self.detector.mask, point, quad,
                                 mode=mode, camera_label=label, maps=maps)

        if self.show_help and (self.help_until == 0.0 or t < self.help_until):
            overlay.draw_help(canvas)
        elif self.show_help:
            self.show_help = False

    def auto_tune(self) -> None:
        """Sample ~0.5s of video and lift the thresholds above the background."""
        self.notify("auto-tuning - point the laser away from the screen", 1.0)
        frames = []
        seen = self.frame_seq
        deadline = time.time() + 1.0
        while len(frames) < 12 and time.time() < deadline:
            frame, seq = self.camera.read()
            if frame is not None and seq != seen:
                seen = seq
                frames.append(frame.copy())
        if not frames:
            self.notify("auto-tune failed - no frames")
            return
        floor, sens = self.detector.auto_threshold(frames)
        self.notify(f"auto-tuned: ignoring anything below {floor} local redness", 3.0)

    # -- input --------------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        ch = chr(key) if 32 <= key < 127 else ""
        low = ch.lower()

        if key == 27:  # ESC
            if self.mode == MODE_MENU:
                if self.menu is None or not self.menu.back():
                    self.end_game()
                return True
            if self.mode == MODE_GAME:
                self.end_game()
                return True
            if self.mode == MODE_AUTO:
                self.mode = MODE_TRACK
                self.auto = None
                self.notify("automatic calibration skipped - press K to retry")
                return True
            if self.mode == MODE_CALIB:
                self.mode = MODE_TRACK
                self.session = None
                self.notify("Calibration cancelled")
                return True
            return False
        if low == "q":
            return False

        if self.mode == MODE_CALIB and self.session is not None:
            if key == 32:  # SPACE
                if self.session.force_capture(self.detection):
                    self.notify("point captured", 1.0)
                else:
                    self.notify("no laser visible", 1.0)
                if self.session and self.session.done:
                    return True
            elif low == "b":
                self.session.back()
                return True

        if self.mode == MODE_MENU and self.menu is not None:
            # Only the menu's own keys: anything else would change mode from
            # under it.
            if ch.isdigit() and ch != "0":
                choice = self.menu.choose(int(ch) - 1)
                if choice is not None:
                    self.start_game(choice)
            elif low == "m":
                self.mouse_input = not self.mouse_input
            return True

        if self.mode == MODE_GAME and self.game is not None and self.game.key(key):
            return True

        if low == "g":
            # Mid-game G replays the same game; otherwise it opens the chooser.
            if self.mode == MODE_GAME:
                self.start_game()
            else:
                self.open_menu()
        elif low == "p" and self.mode == MODE_GAME and self.game is not None:
            self.game.toggle_pause(time.time())
            self._blind_pause = False       # the player has taken over
        elif low == "c":
            self.start_calibration()
        elif low == "k":
            self.start_auto()
        elif low == "f":
            self.auto_tune()
        elif low == "d":
            self.preview = (self.preview + 1) % 3
            self.notify(("preview off", "preview: camera + mask", "preview: camera")[self.preview], 1.5)
        elif low == "n":
            self.switch_camera()
        elif low == "v":
            self.open_picker()
        elif low == "m":
            self.mouse_input = not self.mouse_input
            self.notify("pointer: mouse (hold the left button)" if self.mouse_input
                        else "pointer: laser", 2.5)
        elif low == "u":
            self.correct_preview = not self.correct_preview
            if self.calibration and self.calibration.lens:
                self.notify("preview: lens-corrected" if self.correct_preview
                            else "preview: raw camera image", 1.5)
            else:
                self.notify("no lens correction yet - calibrate first", 2.0)
        elif low == "t":
            self.show_trail = not self.show_trail
            self.trail.clear()
        elif low == "x":
            self.show_crosshair = not self.show_crosshair
        elif low == "h":
            self.show_help = not self.show_help
            self.help_until = 0.0
        elif ch == "[":
            self.detector.adjust(sensitivity=-0.5)
        elif ch == "]":
            self.detector.adjust(sensitivity=+0.5)
        elif ch == ",":
            self.detector.adjust(redness=-5)
        elif ch == ".":
            self.detector.adjust(redness=+5)
        elif low == "e":
            self.notify(f"exposure {self.camera.nudge_exposure(0.7):.0f}", 1.5)
        elif low == "r":
            self.notify(f"exposure {self.camera.nudge_exposure(1.4):.0f}", 1.5)
        elif low == "a":
            self.camera.set_auto_exposure(not self.camera.auto_exposure)
            self.notify(f"auto exposure {'on' if self.camera.auto_exposure else 'off'}", 1.5)
        elif low == "s":
            if self.calibration:
                self.calibration.save(self.calib_path, self.detector.s)
                self.notify(f"saved {self.calib_path}")
            else:
                self.notify("nothing to save - calibrate first")
        return True


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track a red laser pointer on this screen")
    p.add_argument("--camera", default=None,
                   help="camera index, a name fragment (e.g. genius), 'auto' "
                        "(prefer a plugged-in USB camera) or 'pick' (show the "
                        "chooser). Default: the camera chosen last time, or the "
                        "chooser if there are several")
    p.add_argument("--list-cameras", action="store_true",
                   help="list capture devices and exit")
    p.add_argument("--width", type=int, default=1280, help="capture width")
    p.add_argument("--height", type=int, default=720, help="capture height")
    p.add_argument("--fps", type=int, default=30, help="capture fps")
    p.add_argument("--exposure", type=float, default=None,
                   help="manual exposure in driver units (Linux: e.g. 50, "
                        "Windows: log2 seconds, e.g. -7); low values make the dot stand out")
    p.add_argument("--screen", type=int, nargs=2, metavar=("W", "H"),
                   help="override detected screen resolution")
    p.add_argument("--windowed", action="store_true", help="do not go fullscreen")
    p.add_argument("--calibration", default="calibration.json", help="calibration file path")
    p.add_argument("--grid", type=int, nargs=2, default=(4, 4), metavar=("COLS", "ROWS"),
                   help="calibration marker grid (default 4 4; 3 3 is quicker but "
                        "leaves less data for the lens fit)")
    p.add_argument("--no-auto-calibrate", action="store_true",
                   help="do not calibrate automatically at startup")
    p.add_argument("--no-lens", action="store_true",
                   help="skip lens-distortion fitting, homography only")
    p.add_argument("--smoothing", type=float, default=0.45,
                   help="0-1 exponential smoothing, higher = snappier")
    p.add_argument("--hold", type=float, default=0.4,
                   help="seconds the reticle lingers after the laser disappears")
    p.add_argument("--bounds-margin", type=float, default=0.05,
                   help="fraction of screen size tolerated outside the screen edges")
    p.add_argument("--no-trail", action="store_true", help="start with the trail off")
    p.add_argument("--no-preview", action="store_true",
                   help="start without the camera preview")
    p.add_argument("--mouse-pointer", action="store_true",
                   help="use the mouse as the pointer (hold the left button) - "
                        "handy for trying the game without a laser")
    p.add_argument("--no-sound", action="store_true",
                   help="no voices, music or sound effects")
    p.add_argument("--mouse", action="store_true",
                   help="also move the real mouse cursor (needs pyautogui)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    enable_dpi_awareness()
    args = parse_args(argv)
    if args.list_cameras:
        cams = list_cameras()
        print("capture devices:" if cams else "no capture devices found")
        for c in cams:
            print("  " + c.label)
        return 0
    try:
        LaserApp(args).run()
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        pass
    return 0
