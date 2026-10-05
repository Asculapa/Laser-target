"""Offline sanity checks - no camera or display needed.

Renders synthetic frames with a fake laser dot, runs the detector on them,
solves a homography from the results and checks the round-trip accuracy.
"""
from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from laserapp import overlay, story_art
from laserapp import sound as audio
from laserapp import story_script as script
from laserapp.calibration import grid_points, solve
from laserapp.lens import LensModel
from laserapp.autocal import AutoCalibration, find_dot
from laserapp.camera import FrameDecoder, jpeg_intact
from laserapp.game import Game, HighScores, Stats, NORMAL, BONUS, DECOY
from laserapp.mathgames import PACKS, BalloonMath, NumberHunt, make_task
from laserapp.menu import AGAIN, GAMES, LEARNING, OTHER, SHOOTING, EndButtons, GameMenu, entry
from laserapp.silhouette import CENTRE_V, RING, Silhouette, zone
from laserapp.detector import LaserDetector
from laserapp import duel_art
from laserapp.duel import ASSETS as DUEL_ASSETS, MUSIC, Duel, Lane
from laserapp import jars
from laserapp.jars import KINDS, SHELVES, JarLane, JarRange, QuickDraw
from laserapp import west, west_art
from laserapp.west import CLEAR, STAGES, West
from laserapp.story import (ASSETS, END, INTRO, LEVELS, OUTRO, PLAY, RESULT, RETRY, TITLE,
                            Hold, Story)
from laserapp.story_levels import (BRUTE, EYE, GLOAMLING, MOTH, SKY, Constellations, Ships,
                                   Siege, Sparks, Star, Storm)

audio.enabled = False               # the games are tested without their sound


def synth_frame(size=(1280, 720), dot=None, noise=True, glare=0.0, bands=0.0,
                dot_strength=1.0, decoy=False, also=None):
    """A camera's view of a screen, with optional sunlight and display banding.

    `glare` washes one corner out the way sunlight on a screen does; `bands`
    adds the drifting rolling-shutter stripes a camera picks up when pointed
    straight at a display. `also` is a second laser: (x, y, strength).
    """
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (60, 40), (w - 60, h - 40), (90, 70, 40), -1)      # screen glow
    cv2.putText(img, "SCREEN CONTENT", (140, 200), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (200, 200, 200), 4)                                   # white text
    if decoy:
        cv2.circle(img, (int(w * 0.8), int(h * 0.8)), 40, (40, 40, 160), -1)  # red screen content

    if bands > 0:
        ys = np.arange(h, dtype=np.float32)
        profile = (np.sin(ys / 28.0) * 0.5 + 0.5) ** 2 * (bands * 255)
        img = cv2.add(img, np.repeat(profile[:, None], w, axis=1)
                      .astype(np.uint8)[:, :, None].repeat(3, axis=2))

    if glare > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        falloff = np.clip(1.0 - np.hypot(xx - w * 0.25, yy - h * 0.3) / (w * 0.55), 0, 1)
        wash = (falloff ** 2 * glare * 255).astype(np.uint8)
        img = cv2.add(img, cv2.merge([wash, wash, wash]))

    if noise:
        img = cv2.add(img, np.random.randint(0, 12, img.shape, dtype=np.uint8))
    spots = [dot + (dot_strength,)] if dot is not None else []
    for x, y, strength in spots + ([also] if also is not None else []):
        # A laser drives the red channel almost alone; only a strong spot
        # blows its core out to white.
        layer = np.zeros_like(img)
        cv2.circle(layer, (int(x), int(y)), 7,
                   (int(20 * strength), int(20 * strength), int(255 * strength)), -1)
        if strength > 0.8:
            cv2.circle(layer, (int(x), int(y)), 3, (230, 230, 255), -1)
        layer = cv2.GaussianBlur(layer, (9, 9), 0)
        img = cv2.add(img, layer)
    return img


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    passed = True
    det = LaserDetector()

    print("detector:")
    # Conditions the camera actually sees: sunlight washing the screen, and
    # the rolling-shutter banding a camera picks up off a display.
    scenes = [
        ("plain screen", dict()),
        ("sunlit screen", dict(glare=0.75)),
        ("display banding", dict(bands=0.35)),
        ("sunlit + banding", dict(glare=0.75, bands=0.35)),
    ]
    errors = []
    for label, kw in scenes:
        passed &= check(f"no false positive: {label}", det.detect(synth_frame(**kw)) is None,
                        f"{det.candidates} blobs, threshold {det.threshold:.0f}")
        for pt in ((640, 360), (1000, 200), (700, 500)):
            d = det.detect(synth_frame(dot=pt, **kw))
            if d is None:
                passed &= check(f"dot {pt} on {label}", False, "not detected")
                continue
            err = float(np.hypot(d.x - pt[0], d.y - pt[1]))
            errors.append(err)
            passed &= check(f"dot {pt} on {label}", err < 1.5, f"error {err:.2f}px, area {d.area}")
        for strength in (0.5, 0.35):
            faint = det.detect(synth_frame(dot=(700, 500), dot_strength=strength, **kw))
            passed &= check(f"dim dot ({strength:.0%}) on {label}", faint is not None,
                            "not detected" if faint is None else
                            f"error {np.hypot(faint.x-700, faint.y-500):.2f}px")
    print(f"  mean centroid error {np.mean(errors):.2f}px over {len(errors)} spots")

    # A spot on a blown-out area has no redness left to find - the app warns
    # about this instead of pretending otherwise.
    det.detect(synth_frame(glare=0.75))
    passed &= check("overexposure is measured", det.saturation > 0.005,
                    f"{det.saturation*100:.1f}% of pixels clipped")

    # Red screen content is legitimately red; 'F' is what tells the detector
    # to stop counting whatever is on screen right now.
    tuned = LaserDetector()
    passed &= check("red screen content is picked up before tuning",
                    tuned.detect(synth_frame(decoy=True)) is not None)
    floor, _ = tuned.auto_threshold([synth_frame(decoy=True) for _ in range(3)])
    passed &= check("...and ignored after tuning",
                    tuned.detect(synth_frame(decoy=True)) is None, f"floor now {floor}")

    # Two players, a laser each.
    for strength in (1.0, 0.35):
        both = det.detect_all(synth_frame(dot=(400, 300), also=(900, 500, strength)))
        passed &= check(f"two lasers are both found (the second at {strength:.0%})",
                        sorted((round(d.x), round(d.y)) for d in both)
                        == [(400, 300), (900, 500)],
                        ", ".join(f"({d.x:.0f}, {d.y:.0f})" for d in both))
    passed &= check("...and one laser is still one",
                    len(det.detect_all(synth_frame(dot=(400, 300)))) == 1)

    print("calibration:")
    # Pretend the camera views the screen from an angle: build a known homography,
    # push the screen markers through its inverse to make 'camera' observations.
    screen = (1920, 1080)
    cam = (1280, 720)
    src = np.float32([[0, 0], [screen[0], 0], [screen[0], screen[1]], [0, screen[1]]])
    dst = np.float32([[180, 90], [1130, 150], [1180, 640], [120, 590]])
    true_H = cv2.getPerspectiveTransform(src, dst)          # screen -> camera
    targets = grid_points(screen)
    cam_pts = cv2.perspectiveTransform(
        np.array(targets, np.float64).reshape(-1, 1, 2), true_H).reshape(-1, 2)
    # add sub-pixel detection jitter
    rng = np.random.default_rng(0)
    cam_pts_noisy = cam_pts + rng.normal(0, 0.4, cam_pts.shape)

    calib = solve(cam_pts_noisy, targets, cam, screen)
    passed &= check("homography solved", calib.error < 3.0, f"mean error {calib.error:.2f}px")

    round_trip = [np.hypot(*(np.subtract(calib.to_screen(*c), t)))
                  for c, t in zip(cam_pts, targets)]
    passed &= check("camera->screen round trip", max(round_trip) < 4.0,
                    f"max {max(round_trip):.2f}px")

    quad = calib.screen_quad()
    passed &= check("screen quad recovered", np.allclose(quad, dst, atol=6.0),
                    f"max delta {np.abs(quad - dst).max():.2f}px")
    passed &= check("off-screen point rejected", not calib.in_bounds(-400, -400))

    # The laser calibration: steady readings, one per camera frame.
    from laserapp.calibration import CalibrationSession
    from laserapp.detector import Detection
    spot = lambda x, y: Detection(x, y, 20, 100.0, 3.0)
    cs = CalibrationSession((1920, 1080), (1280, 720), 4, 4)
    cs.cooldown_until = 0.0
    for _ in range(40):
        cs.update(spot(400.0, 300.0), 7)            # one frame, drawn forty times
    passed &= check("one camera frame redrawn is not a steady hold", cs.index == 0)
    for seq in range(8, 8 + cs.HOLD_SAMPLES):
        cs.update(spot(400.0, 300.0), seq)
    passed &= check("twelve frames of a steady laser capture the point", cs.index == 1)
    cs.cooldown_until = 0.0
    for seq in range(100, 140):
        cs.update(spot(401.0, 300.0), seq)
    passed &= check("a laser left on the last marker is not captured again",
                    cs.index == 1 and cs.unmoved)
    for seq in range(140, 140 + cs.HOLD_SAMPLES):
        cs.update(spot(560.0, 300.0), seq)
    passed &= check("...and is once it has moved on", cs.index == 2 and not cs.unmoved)

    print("lens distortion:")
    # A wide-angle camera: bow the projected grid outwards with a known model,
    # then check calibration recovers it from the points alone.
    true_lens = LensModel(k1=-0.30, k2=0.08, width=cam[0], height=cam[1])
    grid = grid_points(screen, cols=4, rows=4)
    straight = cv2.perspectiveTransform(
        np.array(grid, np.float64).reshape(-1, 1, 2), true_H).reshape(-1, 2)
    observed = true_lens.distort(straight)
    bow = np.linalg.norm(observed - straight, axis=1).max()
    passed &= check("test data is actually distorted", bow > 20,
                    f"max {bow:.1f}px of bowing")

    noisy = observed + rng.normal(0, 0.4, observed.shape)
    calib = solve(noisy, grid, cam, screen)
    passed &= check("distortion detected", calib.lens is not None,
                    "none fitted" if calib.lens is None else
                    f"k1={calib.lens.k1:+.3f} (true {true_lens.k1:+.3f}), "
                    f"k2={calib.lens.k2:+.3f} (true {true_lens.k2:+.3f})")
    if calib.lens is not None:
        passed &= check("corrected error beats uncorrected",
                        calib.error < calib.error_no_lens / 3,
                        f"{calib.error:.2f}px vs {calib.error_no_lens:.2f}px")
        worst = max(np.hypot(*np.subtract(calib.to_screen(*o), g))
                    for o, g in zip(observed, grid))
        passed &= check("camera->screen round trip through the lens", worst < 4.0,
                        f"max {worst:.2f}px")

    # One capture where the laser slipped off the marker must not drag the
    # whole fit with it.
    slipped = noisy.copy()
    slipped[5] += [70.0, -55.0]
    robust = solve(slipped, grid, cam, screen)
    passed &= check("slipped capture discarded", robust.dropped == 1,
                    f"dropped {robust.dropped}")
    passed &= check("fit survives the slip", robust.error < calib.error * 1.5,
                    f"{robust.error:.2f}px vs {calib.error:.2f}px clean")

    # A camera with no distortion must not have a lens invented for it.
    clean = solve(cam_pts_noisy, targets, cam, screen)
    passed &= check("no lens invented for a rectilinear camera", clean.lens is None,
                    "none" if clean.lens is None else f"k1={clean.lens.k1:+.3f}")

    print("automatic calibration:")
    ref = (rng.normal(60, 6, (720, 1280, 3)).clip(0, 255)).astype(np.uint8)
    ref_grey = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    def lit(where, radius=9):
        f = ref.copy()
        cv2.circle(f, where, radius, (255, 255, 255), -1)
        return cv2.GaussianBlur(f, (7, 7), 0)

    passed &= check("nothing lit -> no dot", find_dot(ref, ref_grey) is None)
    spot = find_dot(lit((820, 300)), ref_grey)
    passed &= check("lit dot located", spot is not None and
                    np.hypot(spot[0] - 820, spot[1] - 300) < 1.5,
                    "not found" if spot is None else
                    f"error {np.hypot(spot[0]-820, spot[1]-300):.2f}px, peak {spot[2]:.0f}")
    glare = ref.copy()
    cv2.rectangle(glare, (200, 100), (600, 500), (255, 255, 255), -1)
    passed &= check("glare is not a dot", find_dot(glare, ref_grey) is None)

    # Full run against a simulated camera that lags the screen, which is what
    # a real display + webcam does: the previous dot is still in view when the
    # next one lights up.
    cam_quad = np.float32([[180, 90], [1130, 150], [1180, 640], [120, 590]])
    H_screen_to_cam = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [screen[0], 0], [screen[0], screen[1]], [0, screen[1]]]), cam_quad)

    def simulate(lag: int):
        session = AutoCalibration(screen, cam, lens=False, settle=0.0,
                                  dwell=0.0, timeout=0.4)
        pipeline, seq = [], 0
        for _ in range(4000):
            canvas = np.zeros((screen[1], screen[0], 3), np.uint8)
            session.render(canvas)
            view = cv2.warpPerspective(canvas, H_screen_to_cam, cam)
            pipeline.append(cv2.add(view, rng.integers(0, 10, view.shape, dtype=np.uint8)))
            if len(pipeline) > lag:
                seq += 1
                session.update(pipeline.pop(0), seq)
            if session.done:
                break
        return session

    for lag in (1, 4):
        session = simulate(lag)
        ok = session.result is not None and len(session.kept) == len(session.targets)
        passed &= check(f"survives a {lag}-frame camera lag", ok and session.result.error < 3.0,
                        f"{len(session.kept)}/{len(session.targets)} points, " +
                        (f"error {session.result.error:.2f}px" if session.result else "no fit"))

    print("camera frames:")
    # What a USB camera sends, and the ways it arrives damaged.
    picture = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    jpg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_RST_INTERVAL, 4])[1].reshape(-1)
    rst = [i for i in range(len(jpg) - 1) if jpg[i] == 0xFF and 0xD0 <= jpg[i + 1] <= 0xD7]
    junk = np.full(600, 0x5A, np.uint8)
    passed &= check("a whole frame is accepted", jpeg_intact(jpg) and len(rst) > 20,
                    f"{len(jpg)} bytes, {len(rst)} restart markers")
    passed &= check("...also in a zero-padded buffer, or with junk before its end marker",
                    jpeg_intact(np.concatenate([jpg, np.zeros(500, np.uint8)]))
                    and jpeg_intact(np.concatenate([jpg[:-2], junk, jpg[-2:]])))
    passed &= check("...and one without restart markers",
                    jpeg_intact(cv2.imencode(".jpg", picture)[1].reshape(-1)))
    passed &= check("a frame cut short is rejected", not jpeg_intact(jpg[:len(jpg) // 3]))
    passed &= check("two frames run together are rejected",
                    not jpeg_intact(np.concatenate([jpg[:len(jpg) // 3], jpg])))
    passed &= check("a frame with a stretch missing is rejected",
                    not jpeg_intact(np.concatenate([jpg[:rst[5]], jpg[rst[8]:]]))
                    and not jpeg_intact(np.concatenate([jpg[:rst[5]], jpg[rst[13]:]])))

    # A camera that has slipped: each frame's last 600 bytes arrive at the
    # start of the next buffer, after that buffer's own start marker and comment.
    opening = np.array([0xFF, 0xD8, 0xFF, 0xFE, 0x00, 0x06, 1, 2, 3, 4], np.uint8)
    sent = []
    for shade in (40, 90, 140, 190):
        img = picture // 4 + shade
        sent.append((img, np.concatenate(
            [opening, cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_RST_INTERVAL, 4])[1].reshape(-1)[2:]])))
    slipped = [np.concatenate([opening, sent[k - 1][1][-600:], sent[k][1][:-600]])
               for k in range(1, len(sent))]
    dec = FrameDecoder()
    got = [dec.decode(sent[0][1])] + [dec.decode(b) for b in slipped]
    passed &= check("a slipped stream loses only the frame it slipped on",
                    got[0] is not None and got[1] is None and got[2] is not None
                    and got[3] is not None)
    passed &= check("...and what comes out is the picture that was sent, one frame late",
                    all(g is not None and abs(float(g.mean()) - float(sent[k][0].mean())) < 2
                        for g, k in ((got[2], 1), (got[3], 2))),
                    f"loss shown {dec.loss:.0%}")
    passed &= check("a stream back in step is taken as it comes",
                    dec.decode(sent[3][1]) is not None)

    print("shooting game:")

    def fresh(**kw):
        """A game holding one target of a known kind, with the clock at 10s."""
        g = Game((1920, 1080), seed=7, **kw)
        g.start(0.0)
        g.update(g.started + 0.5, None)
        g.targets.clear()
        return g

    def place(g, kind=NORMAL, x=900.0, y=500.0, radius=60.0):
        from laserapp.game import Target
        t = Target(x=x, y=y, radius=radius, born=g.now, lifetime=5.0, kind=kind)
        g.targets.append(t)
        return t

    # Flash: the beam appearing on a target is a trigger pull.
    g = fresh(); t = place(g)
    g.update(g.now + 0.05, None)
    g.update(g.now + 0.05, (t.x, t.y))
    passed &= check("flash on a target scores", g.stats.hits == 1 and g.stats.score > 0,
                    f"score {g.stats.score}, combo {g.combo}")

    # Flash on nothing is a miss and breaks the streak.
    g.combo = 4
    g.update(g.now + 0.3, None)
    g.update(g.now + 0.4, (100.0, 100.0))
    passed &= check("flash on empty space misses", g.stats.misses == 1 and g.combo == 0,
                    f"misses {g.stats.misses}, combo {g.combo}")

    # Dwell: holding the beam still on a target also fires, once.
    g = fresh(); t = place(g)
    g.update(g.now + 0.05, (200.0, 200.0))      # beam lit away from the target
    g.update(g.now + 0.05, (t.x, t.y))          # swept onto it while already lit
    passed &= check("an already-lit beam does not flash-fire", g.stats.hits == 0,
                    f"hits {g.stats.hits}")
    for _ in range(12):
        g.update(g.now + 0.05, (t.x, t.y))
    passed &= check("dwelling on a target fires", g.stats.hits == 1,
                    f"hits {g.stats.hits} after {Game.DWELL_TIME}s of dwell")
    held = g.stats.hits
    for _ in range(12):
        g.update(g.now + 0.05, (t.x, t.y))
    passed &= check("holding does not re-fire on the same target", g.stats.hits == held)

    # A real camera: the detector misses a frame now and then.
    g = fresh(); t = place(g); g.next_spawn = g.now + 99
    g.update(g.now + 1 / 30, (200.0, 200.0))
    misses = g.stats.misses
    for i in range(1, 60):
        g.update(g.now + 1 / 30, None if i % 6 == 0 else (t.x, t.y))
    passed &= check("a hold survives lost frames", g.stats.hits == 1 and g.stats.misses == misses)
    g = fresh(); g.next_spawn = g.now + 99
    g.update(g.now + 1 / 30, (200.0, 200.0))
    g.combo, misses = 4, g.stats.misses
    for i in range(30):
        g.update(g.now + 1 / 30, None if 10 <= i < 16 else (200.0, 200.0))
    passed &= check("a dot lost for 0.2s is not a trigger pull",
                    g.stats.misses == misses and g.combo == 4)
    for i in range(12):
        g.update(g.now + 1 / 30, None)
    g.update(g.now + 1 / 30, (200.0, 200.0))
    passed &= check("...the beam switched off for 0.4s is", g.stats.misses == misses + 1)

    # Escaped targets cost a life; decoys cost points and a life.
    g = fresh(); t = place(g); t.lifetime = 0.2
    lives = g.lives
    g.update(g.now + 0.3, None)
    passed &= check("a target that gets away costs a life",
                    g.stats.escaped == 1 and g.lives == lives - 1)
    g = fresh(); d = place(g, DECOY); g.stats.score = 100; g.combo = 5
    g.update(g.now + 0.05, None)
    g.update(g.now + 0.05, (d.x, d.y))
    passed &= check("shooting a decoy is punished",
                    g.stats.score == 75 and g.combo == 0 and g.stats.decoys == 1,
                    f"score {g.stats.score}, combo {g.combo}")

    # Streaks multiply, and a bonus target is worth appreciably more.
    g = fresh(); t = place(g)
    g.update(g.now + 0.05, None); g.update(g.now + 0.05, (t.x, t.y))
    plain = g.stats.score
    g2 = fresh(); b = place(g2, BONUS)
    g2.update(g2.now + 0.05, None); g2.update(g2.now + 0.05, (b.x, b.y))
    passed &= check("bonus targets are worth more", g2.stats.score > plain * 2,
                    f"{g2.stats.score} vs {plain}")
    g3 = fresh(); g3.combo = 10
    t3 = place(g3)
    g3.update(g3.now + 0.05, None); g3.update(g3.now + 0.05, (t3.x, t3.y))
    passed &= check("a streak multiplies the score",
                    abs(g3.stats.score - plain * 2.0) <= 1,
                    f"{g3.stats.score} vs {plain} at x{g3.multiplier:.1f}")

    # Difficulty ramp and pause.
    g = Game((1920, 1080), seed=3, duration=60)
    g.start(0.0); g.update(g.started + 0.1, None)
    early = (g._spawn_gap(), g._radius(), g._lifetime())
    g.update(g.started + 55, None)
    late = (g._spawn_gap(), g._radius(), g._lifetime())
    passed &= check("it gets harder", all(l < e for l, e in zip(late, early)),
                    f"gap {early[0]:.2f}->{late[0]:.2f}s, radius {early[1]:.0f}->{late[1]:.0f}px, "
                    f"life {early[2]:.1f}->{late[2]:.1f}s")
    g = Game((1920, 1080), seed=3, duration=60)
    g.start(0.0)
    g.update(g.started + 10, None)
    before, at = g.time_left, g.now
    g.toggle_pause(at)
    g.toggle_pause(at + 30)
    g.update(at + 30.01, None)
    passed &= check("pausing does not burn the clock", abs(g.time_left - before) < 0.1,
                    f"{before:.1f}s -> {g.time_left:.1f}s across a 30s pause")

    # A full round, played by a bot that flashes at the nearest target.
    g = Game((1920, 1080), seed=11, duration=20)
    g.start(0.0)
    now, canvas = 0.0, overlay.blank((1920, 1080))
    while g.state != "over" and now < 60:
        # off for a third of a second, then on the target: that is a flash
        for _ in range(20):
            now += 1 / 60
            g.update(now, None)
        target = next((t for t in g.targets if t.dying is None and t.kind != DECOY), None)
        now += 1 / 60
        g.update(now, (target.x, target.y) if target else None)
        g.draw(canvas)
    st = g.stats
    passed &= check("a full round plays out", st.hits > 5 and st.score > 0,
                    f"{st.hits} hits, score {st.score}, accuracy {st.accuracy:.0%}, "
                    f"best streak {st.best_combo}, lives left {g.lives}")
    passed &= check("targets stay on screen",
                    all(0 <= t.x <= 1920 and 0 <= t.y <= 1080 for t in g.targets))

    # Nobody shooting, so the screen fills up as far as it ever will.
    worst = float("inf")
    for seed in range(8):
        g = Game((1920, 1080), seed=seed, duration=60, lives=10 ** 6)
        g.start(0.0)
        now = g.started
        while g.state != "over":
            now += 1 / 30
            g.update(now, None)
            live = [t for t in g.targets if t.dying is None]
            for i, a in enumerate(live):
                for b in live[i + 1:]:
                    worst = min(worst, float(np.hypot(a.x - b.x, a.y - b.y)) - a.radius - b.radius)
    passed &= check("targets never overlap", worst > 0, f"closest rims {worst:.0f} px apart")

    # High scores: ordering, truncation, persistence.
    import tempfile
    path = Path(tempfile.mkdtemp()) / "highscores.json"
    hs = HighScores(path, limit=3)
    for score in (100, 500, 300, 50, 400):
        s_ = Stats(); s_.score = score; s_.hits = 5
        hs.add(s_, "01 Jan 00:00")
    passed &= check("high scores are ranked and capped",
                    [e["score"] for e in hs.entries] == [500, 400, 300],
                    str([e["score"] for e in hs.entries]))
    passed &= check("high scores persist",
                    [e["score"] for e in HighScores(path, limit=3).entries] == [500, 400, 300])
    s_ = Stats(); s_.score = 450
    passed &= check("rank is reported for a new best", hs.rank_of(450) == 2,
                    f"rank {hs.rank_of(450)}")
    passed &= check("a low score does not make the table", hs.rank_of(10) is None)

    print("silhouette range:")
    step = RING[0]
    across = [zone(k * step + step / 2, CENTRE_V) for k in range(7)]
    passed &= check("the rings count down from 10 across the body",
                    across == [10, 9, 8, 7, 6, 5, 0], str(across))
    passed &= check("the head and the hips are worth 5, the air beside them nothing",
                    zone(0, 0.12) == 5 and zone(0.26, 0.99) == 5
                    and zone(0.2, 0.1) == 0 and zone(0, 1.02) == 0)

    def lane(**kw):
        """A series with its first figure up."""
        s = Silhouette((1920, 1080), seed=6, **kw)
        s.start(0.0)
        s.update(s.started + 0.05, None)
        return s, s.targets[0]

    s, t = lane()
    s.update(s.now + 0.05, (t.x, t.y))
    passed &= check("a flash on the middle of the chest scores 10",
                    s.results == [10] and s.stats.score == 10 and t.hole is not None)
    s.update(s.now + 0.2, None)
    s.update(s.now + 0.05, (t.x, t.y))
    passed &= check("a figure takes one shot only", s.results == [10])

    s, t = lane()
    s.update(s.now + 0.05, (t.x + 2.5 * step * t.height, t.y))
    passed &= check("a shot scores the ring it lands in", s.results == [8], str(s.results))
    s, t = lane()
    s.update(s.now + 0.05, (t.x + t.height, t.y))
    passed &= check("a shot off the figure scores nothing and uses it up",
                    s.results == [0] and s.stats.misses == 1 and t.dying is not None)

    s = Silhouette((1920, 1080), seed=6)
    s.start(0.0)
    s.update(s.started - 0.1, (5.0, 5.0))                   # lit before it comes up
    s.update(s.started + 0.05, (5.0, 5.0))
    t = s.targets[0]
    passed &= check("a beam that was already lit does not flash-fire", not s.results)
    for i in range(20):                                     # swept across the chest
        s.update(s.now + 0.05, (t.x + (i - 10) * 0.025 * t.height, t.y))
    passed &= check("sweeping a lit beam across the figure does not fire", not s.results)
    for _ in range(12):
        s.update(s.now + 0.05, (t.x, t.y + 1.5 * RING[1] * t.height))
    passed &= check("holding it still does", s.results == [9], str(s.results))

    for label, lost in (("a hold on the figure survives lost frames", 6),
                        ("a dot lost for 0.2s does not spend the shot", 0)):
        s = Silhouette((1920, 1080), seed=6)
        s.start(0.0)
        s.update(s.started - 0.1, (5.0, 5.0))
        s.update(s.started + 0.05, (5.0, 5.0))
        t = s.targets[0]
        if lost:
            for i in range(1, 30):
                s.update(s.now + 1 / 30, None if i % lost == 0 else (t.x, t.y))
            passed &= check(label, s.results == [10], str(s.results))
        else:
            for i in range(30):
                s.update(s.now + 1 / 30, None if 10 <= i < 16 else (5.0, 5.0))
            passed &= check(label, not s.results, str(s.results))

    s, t = lane()
    s.update(s.now + t.lifetime + 0.1, None)
    passed &= check("a figure nobody shot scores nothing",
                    s.results == [0] and s.stats.escaped == 1 and t.label == "запізно")

    s, t = lane()
    left, at = t.remaining(s.now), s.now
    s.toggle_pause(at)
    s.toggle_pause(at + 30)
    s.update(at + 30.01, None)
    passed &= check("pausing does not burn a figure's time",
                    t.dying is None and abs(t.remaining(s.now) - left) < 0.01)

    # A whole series by a bot that flashes at the middle of every figure.
    s = Silhouette((1920, 1080), seed=12)
    s.start(0.0)
    now, canvas = 0.0, overlay.blank((1920, 1080))
    heights, moved, on_screen = [], 0, True
    while s.state != "over" and now < 120:
        now += 1 / 60
        s.update(now, None)
        t = next((t for t in s.targets if t.dying is None), None)
        if t is not None and t.age(now) < 0.5:               # give it time to walk
            on_screen &= 0 <= t.x - 0.31 * t.height and t.x + 0.31 * t.height <= 1920 \
                and t.y - CENTRE_V * t.height >= 100
            continue
        if t is not None:
            heights.append(t.height)
            moved += t.vx != 0
        now += 1 / 60
        s.update(now, (t.x, t.y) if t else None)
        s.draw(canvas)
    s.draw_over(canvas, None, None)
    passed &= check("a full series plays out", s.state == "over" and s.results == [10] * 10,
                    f"{s.stats.score} of 100 in {now:.0f}s")
    passed &= check("the figures get further away, and the last ones walk",
                    heights == sorted(heights, reverse=True) and heights[-1] < 0.7 * heights[0]
                    and 0 < moved < 10,
                    f"{heights[0]:.0f}px -> {heights[-1]:.0f}px, {moved} moving")
    passed &= check("figures stay on screen, clear of the score line", on_screen)

    print("range duel:")
    size = (1920, 1080)
    FOE, FRIEND = duel_art.FOE, duel_art.FRIEND
    sides, reddest = {}, 0
    for name in duel_art.LOOKS:
        img, whose = duel_art.render(name, 380)
        sides[name] = set(np.unique(whose)) - {0}
        reddest = max(reddest, int((img[..., 2].astype(np.int16) - img[..., 1]).max()))
    passed &= check("every figure is all armed or all unarmed - but for the one with a hostage",
                    all(sides[n] == ({FOE, FRIEND} if l.mixed else {FOE} if l.foe else {FRIEND})
                        for n, l in duel_art.LOOKS.items()),
                    f"{len(duel_art.FOES)} armed, {len(duel_art.FRIENDS)} unarmed, "
                    f"{len(duel_art.MIXED)} with a hostage")
    passed &= check("no figure has any red in it", reddest <= 0)

    def aim(t, side):
        """A spot on figure `t` that belongs to `side`."""
        _, whose = duel_art.render(t.look, t.height)
        ys, xs = np.nonzero(whose == side)
        k = len(xs) // 3
        return (t.x - whose.shape[1] / 2 + xs[k] + 0.5, t.y - t.height / 2 + ys[k] + 0.5)

    def duel(look, seed=4):
        """A duel just past its countdown, with `look` up in both lanes."""
        d = Duel(size, seed=seed)
        d.start(0.0)
        d.update(d.started - 0.2, None, [])
        d.plan[:] = [type(d.plan[0])(0.0, 4, look, 3.0)]
        d.update(d.started + 0.2, None, [])
        return d, [lane.targets[0] for lane in d.lanes]

    d = Duel(size, seed=4)
    d.start(0.0)
    passed &= check("both lanes are given the same figures",
                    d.lanes[0].plan is d.lanes[1].plan and len(d.plan) > 40, f"{len(d.plan)}")
    armed = sum(duel_art.LOOKS[p.look].foe for p in d.plan) / len(d.plan)
    passed &= check("about two figures in three are armed", 0.5 <= armed <= 0.78, f"{armed:.0%}")
    apart = True
    for i, a in enumerate(d.plan):
        apart &= not any(b.station == a.station and b.at < a.at + a.lifetime + Lane.DROP
                         for b in d.plan[i + 1:])
    passed &= check("no figure comes up where one is still standing", apart)
    fits = True
    for lane in d.lanes:
        for x, foot, height in lane.stations:
            half = duel_art.ASPECT * height / 2
            fits &= lane.span[0] <= x - half and x + half <= lane.span[1] \
                and foot - height >= 150 and foot <= 1080
    passed &= check("every station is inside its own half, clear of the scores", fits)

    d, (left, right) = duel("gunman")
    d.update(d.now + 0.05, None, [aim(left, FOE)])
    a, b = d.lanes
    passed &= check("a flash on an armed figure scores for the player whose half it is in",
                    a.score > 0 and a.stats.hits == 1 and b.score == 0 and left.dying is not None
                    and right.dying is None, f"{a.score} : {b.score}")
    d.update(d.now + 0.05, None, [aim(left, FOE), aim(right, FOE)])
    passed &= check("a second dot, in the other half, is the other player's",
                    b.stats.hits == 1 and a.stats.hits == 1 and b.score > 0)
    d, (left, right) = duel("gunman")
    d.update(d.now + 0.05, None, [aim(right, FOE), aim(left, FOE)])
    passed &= check("two players can shoot in the same frame",
                    [lane.stats.hits for lane in d.lanes] == [1, 1]
                    and d.lanes[0].score == d.lanes[1].score)
    d, (left, right) = duel("rifleman")
    d.update(d.now + 0.05, aim(right, FOE))
    passed &= check("a single pointer plays for the half it is in",
                    [lane.stats.hits for lane in d.lanes] == [0, 1])
    d, (left, right) = duel("gunman")
    d.update(d.now + 0.05, None, [(left.x, left.y - left.height)])
    passed &= check("a flash past the figure is a miss, and costs nothing",
                    d.lanes[0].stats.misses == 1 and d.lanes[0].score == 0)

    d, (left, right) = duel("woman")
    d.update(d.now + 0.05, None, [aim(left, FRIEND)])
    passed &= check("shooting someone unarmed costs more than a hit pays",
                    d.lanes[0].score == -Lane.PENALTY and d.lanes[0].stats.decoys == 1
                    and d.lanes[0].stats.hits == 0 and Lane.PENALTY > 100)
    d.update(d.now + 5.0, None, [])
    passed &= check("...and leaving them alone costs nothing",
                    d.lanes[1].score == 0 and d.lanes[1].stats.escaped == 0)
    d, (left, right) = duel("thug")
    d.update(d.now + 5.0, None, [])
    passed &= check("an armed figure nobody shot is counted as got away",
                    [lane.stats.escaped for lane in d.lanes] == [1, 1])

    d, (left, right) = duel("hostage")
    d.update(d.now + 0.05, None, [aim(left, FRIEND), aim(right, FOE)])
    a, b = d.lanes
    plain, (_, other) = duel("gunman")
    plain.update(plain.now + 0.05, None, [aim(other, FOE)])
    passed &= check("with a hostage, it matters who the shot lands on",
                    a.score == -Lane.PENALTY and b.stats.hits == 1
                    and b.score > 1.8 * plain.lanes[1].score, f"{a.score} : {b.score}")

    d, (left, right) = duel("gunman")
    for _ in range(4):                                  # lit, then walked on to the figure
        d.update(d.now + 0.05, None, [(left.x, left.y - left.height)])
    held = 0
    while left.dying is None and held < 40:
        held += 1
        d.update(d.now + 1 / 30, None, [aim(left, FOE)])
    passed &= check("a beam held on an armed figure shoots it as well",
                    d.lanes[0].stats.hits == 1 and d.lanes[0].stats.misses == 1,
                    f"after {held / 30:.2f}s")

    d, (left, right) = duel("gunman")
    remaining, at, clock = left.remaining(d.now), d.now, d.time_left
    d.toggle_pause(at)
    d.update(at + 10, None, [aim(left, FOE)])
    passed &= check("nothing can be shot while paused", d.lanes[0].stats.hits == 0)
    d.toggle_pause(at + 30)
    d.update(at + 30.01, None, [])
    passed &= check("pausing stops the clock and the figures, in both lanes",
                    abs(d.time_left - clock) < 0.05 and left.dying is None
                    and abs(left.remaining(d.now) - remaining) < 0.01
                    and abs(right.remaining(d.now) - remaining) < 0.01)

    def play(d, bots):
        """A round played by a bot per lane: (reaction time, shoots the unarmed too)."""
        now, canvas, reddest = d.now, overlay.blank(size), 0
        while d.state != "over" and now < d.now + 200:
            now += 1 / 30
            dots = []
            for lane, (delay, careless) in zip(d.lanes, bots):
                for t in lane.targets:
                    foe = duel_art.LOOKS[t.look].foe
                    if t.dying is None and t.age(now) > delay and (foe or careless) \
                            and int(now * 30) % 6 < 3:      # the beam flashes on and off
                        dots.append(aim(t, FOE if foe else FRIEND))
                        break
            d.update(now, dots[0] if dots else None, dots)
            if int(now * 30) % 20 == 0:
                canvas[:] = 0
                d.draw(canvas)
                reddest = max(reddest, int((canvas[..., 2].astype(np.int16)
                                            - canvas[..., 1]).max()))
        d.draw(canvas)
        d.draw_over(canvas, None, None)
        return reddest

    d = Duel(size, seed=9)
    d.start(0.0)
    reddest = play(d, ((0.5, False), (0.9, True)))
    a, b = d.lanes
    passed &= check("a full round plays out, and the careful player wins it",
                    d.state == "over" and d.winner == 0 and d.wins == [1, 0]
                    and a.stats.decoys == 0 and b.stats.decoys > 5 and a.score > b.score,
                    f"{a.score} : {b.score}")
    passed &= check("nothing drawn in a duel is red", reddest <= 0)
    passed &= check("G is a rematch: a new round, the rounds won kept",
                    d.key(ord("g")) and d.state == "playing" and d.wins == [1, 0]
                    and all(lane.score == 0 for lane in d.lanes))
    play(d, ((0.6, False), (0.6, False)))
    passed &= check("two players who shoot alike draw, and nobody is given the round",
                    d.winner is None and d.wins == [1, 0]
                    and d.lanes[0].score == d.lanes[1].score > 0,
                    f"{d.lanes[0].score} : {d.lanes[1].score}")
    x0, y0, x1, y1 = d.rematch_box()
    button, now = ((x0 + x1) / 2, (y0 + y1) / 2), d.now
    for _ in range(30):                                 # a beam still there from the round
        now += 1 / 30
        d.update(now, button, [button])
    passed &= check("the rematch button waits for the last shots to be over",
                    d.state == "over")
    held = now
    while d.state == "over" and now < held + 5:
        now += 1 / 30
        d.update(now, button, [(200.0, 300.0), button])
        d.draw_over(overlay.blank(size), None, None)
    passed &= check("a laser held on the rematch button starts the next round",
                    d.state == "playing" and d.wins == [1, 0] and d.counting_down
                    and all(lane.score == 0 for lane in d.lanes), f"after {now - held:.1f}s")
    length = audio.Player(DUEL_ASSETS).length(MUSIC)
    if length is None:
        print("  SKIP  no music (tools/duel_audio.py) - the duel plays without")
    else:
        passed &= check("the music lasts the round out, countdown included",
                        length >= Duel.COUNTDOWN + Duel.DURATION,
                        f"{length:.0f}s")

    print("jar shoot:")
    size = (1920, 1080)

    def reddest_of(canvas):
        return int((canvas[..., 2].astype(np.int16) - canvas[..., 1]).max())

    canvas = overlay.blank(size)
    for n, look in enumerate(KINDS):
        jars.draw_jar(canvas, 200 + n * 300, 600, 220, look, 0.6)
    passed &= check("no jar has any red in it", reddest_of(canvas) <= 0)
    # A dot has to stand out on top of a jar: anything wider than a thin line
    # is dim, so what is left bright after an opening the width of a dot is nothing.
    bright = (canvas.max(axis=2) > jars.LARGE).astype(np.uint8)
    dot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    passed &= check("nothing large on a jar is bright - a dot on it would be lost",
                    not cv2.morphologyEx(bright, cv2.MORPH_OPEN, dot).any(),
                    f"{int(cv2.morphologyEx(bright, cv2.MORPH_OPEN, dot).sum())} px")
    worth = [KINDS[k] for k in jars.SIZES]
    passed &= check("the smaller the jar, the more it pays and the longer it stays",
                    all(a.height > b.height and a.points < b.points and a.lifetime < b.lifetime
                        for a, b in zip(worth, worth[1:])))

    g = JarRange(size, seed=4)
    g.start(0.0)
    looks = {p.look for p in g.plan}
    passed &= check("both lanes are given the same jars, of every kind",
                    g.lanes[0].plan is g.lanes[1].plan and len(g.plan) > 40
                    and looks == set(KINDS), f"{len(g.plan)} jars")
    passed &= check("a gold jar only after the first ten seconds",
                    all(p.at > 10 for p in g.plan if p.look == "gold"))
    apart = True
    for i, a in enumerate(g.plan):
        apart &= not any(b.station == a.station and b.at < a.at + a.lifetime + JarLane.DROP
                         for b in g.plan[i + 1:])
    passed &= check("no jar comes up where one is still standing", apart)
    fits = True
    for lane in g.lanes:
        for p in g.plan:
            t = lane._arrive(p)
            above = max([f * size[1] for f in SHELVES if f * size[1] < t.foot] + [150])
            fits &= lane.span[0] <= t.x - t.width / 2 and t.x + t.width / 2 <= lane.span[1] \
                and t.foot - t.height >= above + 10
    passed &= check("every jar fits its half, under the scores and the shelf above", fits)

    def range_with(look, seed=4, station=1):
        """A jar range just past its countdown, with `look` up in both lanes."""
        g = JarRange(size, seed=seed)
        g.start(0.0)
        g.update(g.started - 0.2, None, [])
        g.plan[:] = [jars.JarPopup(0.0, station, look, KINDS[look].lifetime)]
        g.update(g.started + 0.2, None, [])
        return g, [lane.targets[0] for lane in g.lanes]

    g, (left, right) = range_with("small")
    g.update(g.now + 0.05, None, [(left.x, left.y)])
    a, b = g.lanes
    passed &= check("a flash on a jar breaks it, for the player whose half it is in",
                    a.stats.hits == 1 and a.score > 0 and b.score == 0
                    and left.shot and right.dying is None, f"{a.score} : {b.score}")
    small = a.score
    g, (left, right) = range_with("big")
    g.update(g.now + 0.05, None, [(left.x, left.y)])
    passed &= check("a small jar pays more than a big one", small > g.lanes[0].score,
                    f"{small} against {g.lanes[0].score}")
    g, (left, right) = range_with("mid")
    g.update(g.now + 0.05, None, [(left.x, left.y), (right.x, right.y)])
    quick = g.lanes[0].score
    g, (left, right) = range_with("mid")
    g.update(left.born + 0.8 * left.lifetime, None, [(left.x, left.y)])
    passed &= check("...and a jar shot quickly more than one shot late",
                    quick > g.lanes[0].score and g.lanes[1].stats.hits == 0,
                    f"{quick} against {g.lanes[0].score}")
    g, (left, right) = range_with("mid")
    g.update(g.now + 0.05, None, [(left.x, left.y - left.height)])
    passed &= check("a flash beside the jar is a miss, and costs nothing",
                    g.lanes[0].stats.misses == 1 and g.lanes[0].score == 0)
    g, (left, right) = range_with("big")
    g.update(g.now + 5.0, None, [])
    passed &= check("a jar left until its time is up is counted as got away",
                    [lane.stats.escaped for lane in g.lanes] == [1, 1] and left.dying is not None
                    and not left.shot)
    g, (left, right) = range_with("small")
    for _ in range(4):
        g.update(g.now + 0.05, None, [(left.x, left.y - left.height)])
    held = 0
    while left.dying is None and held < 40:
        held += 1
        g.update(g.now + 1 / 30, None, [(left.x, left.y)])
    passed &= check("a beam held on a jar breaks it as well", g.lanes[0].stats.hits == 1,
                    f"after {held / 30:.2f}s")
    g, (left, right) = range_with("small")
    remaining, at, clock = left.remaining(g.now), g.now, g.time_left
    g.toggle_pause(at)
    g.update(at + 10, None, [(left.x, left.y)])
    g.toggle_pause(at + 30)
    g.update(at + 30.01, None, [])
    passed &= check("pausing stops the clock and the jars",
                    g.lanes[0].stats.hits == 0 and abs(g.time_left - clock) < 0.05
                    and abs(left.remaining(g.now) - remaining) < 0.01)

    def play_range(g, delays):
        """A round with a bot per lane, which shoots each jar after `delay`."""
        now, canvas, reddest = g.now, overlay.blank(size), 0
        while g.state != "over" and now < g.now + 200:
            now += 1 / 30
            dots = []
            for lane, delay in zip(g.lanes, delays):
                for t in lane.targets:
                    if t.dying is None and t.age(now) > delay and int(now * 30) % 6 < 3:
                        dots.append((t.x, t.y))
                        break
            g.update(now, dots[0] if dots else None, dots)
            if int(now * 30) % 15 == 0:
                canvas[:] = 0
                g.draw(canvas)
                reddest = max(reddest, reddest_of(canvas))
        g.draw_over(canvas, None, None)
        return max(reddest, reddest_of(canvas))

    g = JarRange(size, seed=9)
    g.start(0.0)
    reddest = play_range(g, (0.4, 1.2))
    a, b = g.lanes
    passed &= check("a full round plays out, and the quicker player wins it",
                    g.state == "over" and g.winner == 0 and g.wins == [1, 0]
                    and a.score > b.score,
                    f"{a.score} : {b.score}, {a.stats.hits} and {b.stats.hits} jars")
    passed &= check("nothing drawn at the jars is red", reddest <= 0)
    length = audio.Player(jars.ASSETS).length(JarRange.TUNE)
    if length is None:
        print("  SKIP  no music (tools/jars_audio.py) - the jar range plays without")
    else:
        passed &= check("the music lasts the round out, countdown included",
                        length >= JarRange.COUNTDOWN + JarRange.DURATION, f"{length:.0f}s")

    def drawn(seed=5):
        """A quick draw at the moment its first jar is up."""
        q = QuickDraw(size, seed=seed)
        q.start(0.0)
        now = 0.0
        while q.phase != jars.DRAW:
            now += 1 / 30
            q.update(now, None, [])
        return q, [lane.targets[0] for lane in q.lanes]

    q, (left, right) = drawn()
    passed &= check("a jar comes up in both halves, in the same place",
                    abs((left.x - q.lanes[0].span[0]) - (right.x - q.lanes[1].span[0])) < 1
                    and left.y == right.y and left.height == right.height)
    q.update(q.now + 0.4, None, [(right.x, right.y)])
    q.update(q.now + 0.2, None, [(left.x, left.y), (right.x, right.y)])
    a, b = q.lanes
    passed &= check("the first to break their jar takes the point, and both times are kept",
                    q.phase == jars.SHOWN and q.point_to == 1 and b.score == 1 and a.score == 0
                    and a.reaction is not None and b.reaction < a.reaction,
                    f"{b.reaction:.2f}s against {a.reaction:.2f}s")

    q = QuickDraw(size, seed=5)
    q.start(0.0)
    now = q.started + 0.5
    q.update(now, None, [])
    q.update(now + 0.1, None, [(300.0, 500.0)])
    passed &= check("a shot before the jar is up gives the point away",
                    q.phase == jars.SHOWN and q.point_to == 1 and q.lanes[1].score == 1
                    and q.lanes[0].fouled and q.lanes[0].verdict == "ЗАРАНО")
    q = QuickDraw(size, seed=5)
    q.start(0.0)
    q.update(q.started + 0.5, None, [])
    q.update(q.started + 0.6, None, [(300.0, 500.0), (1500.0, 500.0)])
    passed &= check("...and when both shoot too soon, nobody has it",
                    q.phase == jars.SHOWN and q.point_to is None
                    and all(lane.score == 0 for lane in q.lanes))
    q, (left, right) = drawn()
    q.update(q.now + 5.0, None, [])
    passed &= check("a jar nobody breaks is nobody's point",
                    q.phase == jars.SHOWN and q.point_to is None
                    and [lane.verdict for lane in q.lanes] == ["запізно"] * 2)
    q, (left, right) = drawn()
    q.update(q.now + 0.3, None, [(left.x, left.y), (right.x, right.y)])
    passed &= check("two shots in the same moment are a dead heat",
                    q.phase == jars.SHOWN and q.point_to is None)

    q = QuickDraw(size, seed=5)
    q.start(0.0)
    q.update(q.started + 0.1, None, [])
    wait_left = q._until - q.now
    q.toggle_pause(q.now)
    q.update(q.now + 20, None, [])
    q.toggle_pause(q.now)
    q.update(q.now + 0.01, None, [])
    passed &= check("pausing holds the wait", q.phase == jars.WAIT
                    and abs((q._until - q.now) - (wait_left - 0.01)) < 0.02)

    def play_draw(q, delays):
        """A match with a bot per lane: it flashes on its jar after `delay`."""
        now, canvas, reddest = q.now, overlay.blank(size), 0
        while q.state != "over" and now < q.now + 300:
            now += 1 / 30
            dots = []
            for lane, delay in zip(q.lanes, delays):
                live = [t for t in lane.targets if t.dying is None]
                if live and live[0].age(now) > delay:
                    dots.append((live[0].x, live[0].y))
            q.update(now, dots[0] if dots else None, dots)
            if int(now * 30) % 10 == 0:
                canvas[:] = 0
                q.draw(canvas)
                reddest = max(reddest, reddest_of(canvas))
        q.draw_over(canvas, None, None)
        return max(reddest, reddest_of(canvas))

    q = QuickDraw(size, seed=7)
    q.start(0.0)
    reddest = play_draw(q, (0.3, 0.5))
    a, b = q.lanes
    passed &= check("the quicker player wins the match, first to five",
                    q.state == "over" and q.winner == 0 and q.wins == [1, 0]
                    and a.score == QuickDraw.TARGET and b.score == 0
                    and len(b.stats.reactions) == QuickDraw.TARGET,
                    f"{a.score} : {b.score}")
    passed &= check("nothing drawn in the quick draw is red", reddest <= 0)
    passed &= check("G is a rematch, the matches won kept",
                    q.key(ord("g")) and q.state == "playing" and q.wins == [1, 0]
                    and all(lane.score == 0 for lane in q.lanes) and q.phase == jars.WAIT)

    print("wild west:")
    size = (1920, 1080)
    reds, brightest = 0, 0
    for look in west_art.LOOKS:
        poses = ("stand", "shoot", "idle", "walk") if west_art.LOOKS[look].armed else ("stand",)
        for pose in poses:
            for img, _ in west_art.frames(look, pose):
                reds = max(reds, reddest_of(img))
                brightest = max(brightest, int(img.max()))
    passed &= check("the CC0 sprites are recoloured: no red, nothing bright",
                    reds <= 0 and brightest <= west_art.BRIGHT,
                    f"{len(west_art.LOOKS)} characters, brightest {brightest}")
    # Every figure has to take a laser dot - not only one that saturates the
    # camera, but a moderate one too: the detector looks for red over the
    # stronger of green and blue, so a figure strong in either would hide it.
    small = (960, 540)
    yy, xx = np.mgrid[:small[1], :small[0]]
    dice = np.random.default_rng(1)
    back = west_art.backdrop(west_art.SCENES[0], small)
    found = {}
    for look in west_art.LOOKS:
        s_ = west_art.sprite(look, "stand", 0, 200, False)
        ys, xs = np.nonzero(s_.solid)
        found[look] = 0
        for _ in range(20):
            i = dice.integers(len(xs))
            frame = back.copy()
            west_art.blit(frame, s_, 480, 450)
            x, y = 480 - s_.fx + xs[i], 450 - s_.fy + ys[i]
            frame = cv2.GaussianBlur(frame, (7, 7), 0).astype(np.float32)
            spot = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 18.0)
            frame += np.dstack([12 * spot, 12 * spot, 80 * spot])     # a moderate red dot
            frame += dice.normal(0, 4, frame.shape)
            d = LaserDetector().detect(np.clip(frame, 0, 255).astype(np.uint8))
            found[look] += d is not None and abs(d.x - x) < 6 and abs(d.y - y) < 6
    passed &= check("a moderate laser dot is found on every part of every figure",
                    min(found.values()) >= 19,
                    "  ".join(f"{k} {v}/20" for k, v in found.items()))
    town = [west_art.backdrop(sc, size) for sc in west_art.SCENES]
    passed &= check("...and the photographs are a dim town by moonlight",
                    all(reddest_of(t) <= 0 and t.max() <= 90 for t in town),
                    ", ".join(sc.name for sc in west_art.SCENES))
    g = West(size, seed=1)
    g.start(0.0)
    canvas = overlay.blank(size)
    for n, kind in enumerate(("coin", "star")):
        g._draw_pickup(canvas, west.Pickup(x=400 + 400 * n, y=300, radius=40, born=0,
                                           lifetime=9, kind=kind), 0.0)
    bright = (canvas.max(axis=2) > jars.LARGE).astype(np.uint8)
    dot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    passed &= check("a star or a coin to shoot is bright only at its edge",
                    not cv2.morphologyEx(bright, cv2.MORPH_OPEN, dot).any()
                    and reddest_of(canvas) <= 0)
    passed &= check("six streets, each with its music",
                    len(west_art.SCENES) == len(STAGES) == 6
                    and {sc.music for sc in west_art.SCENES} == {1, 2, 3})
    sources = lambda side: {west_art.LOOKS[n].source for n in side}
    passed &= check("bandits and townsfolk tell apart: nobody is on both sides",
                    sources(west_art.BANDITS).isdisjoint(sources(west_art.TOWNSFOLK))
                    and len(west_art.BANDITS) >= 2 and len(west_art.TOWNSFOLK) >= 4
                    and west_art.BOSS in west_art.BANDITS,
                    f"{len(west_art.BANDITS)} bandits, {len(west_art.TOWNSFOLK)} townsfolk")
    fits = []
    for sc in west_art.SCENES:
        for spot in sc.spots + (sc.showdown,):
            x, foot, h, clip = west_art.place(sc, spot, size)
            fits.append(60 < x < size[0] - 60 and foot - h >= 110
                        and (foot <= size[1] if clip is None else foot - h < clip < size[1]))
    passed &= check("every place a figure stands is on screen, under the scores",
                    all(fits), f"{sum(fits)} of {len(fits)}")

    g = West(size, seed=2)
    g.start(0.0)
    g.update(g.started + 0.01, None, [])
    shows, kinds = [], set()
    for n in range(600):
        g.stage = n % len(STAGES)
        g.targets.clear()
        g._left = 99
        if not g._spawn(g.now):
            continue
        t = g.targets[0]
        kinds.add(t.kind)
        if t.armed and t.clip is not None:
            s_ = t.sprite(0)
            shows.append(t.foot - s_.fy + s_.muzzle[1] < t.clip)
    passed &= check("a bandit behind cover shows his gun over it",
                    shows and all(shows), f"{len(shows)} bandits behind sills and barrels")
    passed &= check("one player meets pop-ups, peekers and runners - no team shots",
                    kinds == {west.POP, west.PEEK, west.RUN}, " ".join(sorted(kinds)))

    def west_with(*looks, seed=3, players=1, level="sheriff", xs=None):
        """A town just past its countdown, with `looks` up and risen."""
        g = West(size, players=players, level=level, seed=seed)
        g.start(0.0)
        if players == 2:
            g.key(32)
        g.update(g.started + 0.01, None, [])
        g.targets[:] = [t for t in g.targets if isinstance(t, west.Strip)]
        g._next = math.inf
        g._drops = []
        for i, look in enumerate(looks):
            spot = g.scene.spots[(12 + i) % len(g.scene.spots)]
            x, foot, h, clip = west_art.place(g.scene, spot, size)
            if xs is not None:
                x = xs[i]
            g.targets.append(west.Figure(x=x, y=foot - h / 2, radius=h / 2, born=g.now,
                                         lifetime=2.0, look=look, spot=12 + i, foot=foot,
                                         height=h, clip=clip, side=g.side_of((x, foot))))
        g.update(g.now + 0.3, None, [])
        return g, [t for t in g.targets if isinstance(t, west.Figure)]

    def off(g):
        """The beam off long enough that the next dot is a new shot."""
        for _ in range(12):
            g.update(g.now + 1 / 30, None, [])

    def on(t, half=None):
        """A spot on the part of `t` that shows; `half` keeps to that half of the screen."""
        s_ = west_art.sprite(t.look, "stand", 0, t.height, t.flip)
        ys, xs = np.nonzero(s_.solid)
        X, Y = t.x - s_.fx + xs, t.foot - s_.fy + ys
        keep = np.ones(len(X), bool)
        if t.clip is not None:
            keep &= Y < t.clip - 4
        if half is not None:
            keep &= (X < size[0] / 2 - 8) if half == 0 else (X > size[0] / 2 + 8)
        X, Y = X[keep], Y[keep]
        k = len(X) // 2
        return (float(X[k]), float(Y[k]))

    g, (a,) = west_with("gunslinger")
    g.update(g.now + 0.05, None, [on(a)])
    passed &= check("a flash on a bandit brings him down, and scores",
                    a.end == "hit" and g.stats.hits == 1 and g.stats.score > 0
                    and g.lives == g.level.lives and g.team[0].ammo == 5)
    g, (a,) = west_with("lady")
    g.update(g.now + 0.05, None, [on(a)])
    passed &= check("shooting the townsfolk costs a life",
                    a.end == "hit" and g.lives == g.level.lives - 1 and g.stats.decoys == 1)
    g, (a,) = west_with("gunslinger")
    g.update(g.now + 2.0, None, [])
    passed &= check("a bandit left until his ring closes fires, and that costs a life",
                    a.end == "fired" and g.lives == g.level.lives - 1 and g.stats.escaped == 1)
    g, (a, b) = west_with("outlaw", "gunslinger")
    g.update(g.now + 0.05, None, [on(a), on(b)])
    passed &= check("two lasers, two shots in the same moment",
                    a.end == b.end == "hit" and g.stats.hits == 2)
    g, (a, b) = west_with("outlaw", "gunslinger")
    x0, y0 = a.x, a.y - a.height
    g.update(g.now + 0.05, None, [(x0, y0)])               # lit above him: a miss
    for k in range(1, 40):                                 # then walked over him, lit
        g.update(g.now + 1 / 30, None, [(x0 + (on(a)[0] - x0) * min(1, k / 10),
                                         y0 + (on(a)[1] - y0) * min(1, k / 10))])
    passed &= check("a lit beam walked on to a bandit is one gun: a hold, not a new shot",
                    g.stats.misses == 1 and g.stats.hits == 1 and a.end == "hit")
    g, (a,) = west_with("outlaw")
    left, at = a.remaining(g.now), g.now
    g.toggle_pause(at)
    g.update(at + 20, None, [on(a)])
    g.toggle_pause(at + 30)
    g.update(at + 30.01, None, [])
    passed &= check("pausing holds the bandits' fire",
                    a.dying is None and abs(a.remaining(g.now) - left) < 0.02
                    and g.lives == g.level.lives)

    # The six-shooter.
    g, (a,) = west_with("gunslinger")
    a.lifetime = 99
    for k in range(7):                                     # seven shots at nothing
        off(g)
        g.update(g.now + 0.05, None, [(300.0, 200.0)])
    passed &= check("six shots, then the gun is empty", g.team[0].ammo == 0
                    and g.stats.misses == 6, f"{g.stats.misses} misses")
    off(g)
    g.update(g.now + 0.05, None, [on(a)])
    passed &= check("...and an empty gun shoots nobody", a.dying is None)
    off(g)
    g.update(g.now + 0.05, None, [(900.0, size[1] - 10.0)])
    off(g)
    g.update(g.now + 0.05, None, [on(a)])
    passed &= check("a shot at the RELOAD strip fills it again",
                    a.end == "hit" and g.team[0].ammo == 5)

    g, (a,) = west_with("gunslinger")
    a.lifetime = 99
    for k in range(7):
        off(g)
        g.update(g.now + 0.05, None, [(300.0, 200.0)])
    passed &= check("running dry the first time stops and says how to reload",
                    "НАБОЇ СКІНЧИЛИСЯ" in g._message[0] and "ПЕРЕЗАРЯДКИ" in g._sub[0])
    canvas = overlay.blank(size)
    g.draw(canvas)
    bar = canvas[int(size[1] * west.STRIP) + 4:, :]
    passed &= check("...and the RELOAD bar lights up in the player's colour",
                    bar.reshape(-1, 3).max(axis=0)[0] > 200 and reddest_of(canvas) <= 0)
    g, _ = west_with(level="deputy")
    passed &= check("a deputy's gun never runs dry",
                    g.level.ammo == 0 and not any(isinstance(t, west.Strip) for t in g.targets))

    # Peekers and runners.
    g, (a,) = west_with("outlaw")
    a.kind, a.lifetime = west.PEEK, 5.0
    seen = []
    for k in range(60):
        g.update(g.now + 1 / 30, None, [])
        seen.append(a.hidden)
    passed &= check("a peeker ducks down and comes up again", any(seen) and not all(seen))
    while not a.hidden:
        g.update(g.now + 1 / 30, None, [])
    off(g)
    g.update(g.now + 0.05, None, [on(a)])
    passed &= check("...and while he is down there is nothing to hit",
                    a.dying is None and g.stats.misses == 1)
    a.lifetime = a.age(g.now) + 0.05
    while a.hidden and a.dying is None:
        g.update(g.now + 1 / 30, None, [])
    passed &= check("...nor does he fire from behind his cover",
                    a.dying is None or not a.hidden)
    g = West(size, seed=4)
    g.start(0.0)
    g.update(g.started + 0.01, None, [])
    g.stage = 2
    g._left = 99
    while not any(t.kind == west.RUN for t in g.figures()):
        g.targets[:] = [t for t in g.targets if isinstance(t, west.Strip)]
        g._spawn(g.now)
    r = next(t for t in g.figures() if t.kind == west.RUN)
    x0 = r.x
    for _ in range(15):
        g.update(g.now + 1 / 30, None, [])
    passed &= check("a runner crosses the street", abs(r.x - x0) > 50, f"{abs(r.x - x0):.0f}px")

    # Two players.
    def two(*looks, xs, **kw):
        return west_with(*looks, players=2, xs=xs, **kw)

    g, (a, b) = two("gunslinger", "outlaw", xs=[500.0, 1400.0])
    g.update(g.now + 0.05, None, [on(a), on(b)])
    p1, p2 = g.team
    passed &= check("two players: a shot counts for the side it lands on",
                    p1.stats.hits == 1 and p2.stats.hits == 1 and p1.stats.score > 0
                    and p2.stats.score > 0 and g.stats.score == p1.stats.score + p2.stats.score)
    g, (a,) = two("gunslinger", xs=[1400.0])
    start = (500.0, 500.0)
    g.update(g.now + 0.05, None, [start])                  # P1's beam comes on, on the left
    for k in range(1, 40):                                 # ...and is slid across, lit
        g.update(g.now + 1 / 30, None, [(start[0] + (on(a)[0] - start[0]) * min(1, k / 12),
                                         start[1] + (on(a)[1] - start[1]) * min(1, k / 12))])
    passed &= check("a beam slid across to the partner's side is a save",
                    a.end == "hit" and g.team[0].saves == 1 and g.team[0].stats.hits == 1
                    and g.team[1].stats.hits == 0)
    g, (a, b, c) = two("gunslinger", "outlaw", "gunslinger", xs=[400.0, 1450.0, 700.0])
    g.update(g.now + 0.05, None, [on(a)])
    off(g)
    g.update(g.now + 0.05, None, [on(b)])
    passed &= check("hits in turn by the two make a crossfire", g.crossfire == 1
                    and g.multiplier > 1.0 + 0.1 * g.combo)
    g, (a,) = two("gunslinger", xs=[960.0])
    a.kind, a.lifetime = west.TEAM, 5.0
    g.update(g.now + 0.05, None, [on(a, 0)])
    passed &= check("a big one: one player's shot alone does not bring him down",
                    a.dying is None and a.first is not None)
    off(g)
    g.update(g.now + 0.05, None, [on(a, 0), on(a, 1)])
    passed &= check("...a shot from each, together, does", a.end == "hit"
                    and g.team[0].stats.score > 0 and g.team[1].stats.score > 0)

    def to_showdown(players=1, seed=3):
        g, _ = west_with(seed=seed, players=players, xs=[] if players == 2 else None)
        g._left = 0
        g.update(g.now + 0.05, None, [])
        return g

    g = to_showdown()
    g.update(g.now + 0.5, None, [(300.0, 300.0)])
    passed &= check("a shot before the draw costs a life", g.phase == west.SHOWDOWN
                    and g.lives == g.level.lives - 1 and not g._drawn)
    g = to_showdown()
    while not g._drawn:
        g.update(g.now + 1 / 30, None, [])
    g.update(g.now + 0.3, None, [on(g.bosses[0])])
    for _ in range(60):
        g.update(g.now + 1 / 30, None, [])
    passed &= check("...and one after it wins the showdown, and the street",
                    g.phase == CLEAR and g.team[0].draws and g.lives == g.level.lives,
                    f"in {g.team[0].draws[0]:.2f}s" if g.team[0].draws else "")
    g = to_showdown()
    while not g._drawn:
        g.update(g.now + 1 / 30, None, [])
    g.update(g.now + STAGES[0].draw + 0.05, None, [])
    passed &= check("too slow on the draw costs a life", g.lives == g.level.lives - 1
                    and g.bosses[0].end == "fired")
    g = to_showdown(players=2)
    while not g._drawn:
        g.update(g.now + 1 / 30, None, [])
    a, b = g.bosses
    g.update(g.now + 0.3, None, [on(a)])
    for _ in range(60):
        g.update(g.now + 1 / 30, None, [])
    passed &= check("two players face two bosses: one down is not enough",
                    len(g.bosses) == 2 and g.phase == west.SHOWDOWN
                    and g.lives == g.level.lives - 1)

    g = West(size, players=2, seed=3)
    g.start(0.0)
    passed &= check("two players choose before the countdown", g.phase == west.SETUP)
    boxes = [g.setup_boxes(0)[1], g.setup_boxes(1)[0]]
    now = 0.0
    while g.phase == west.SETUP and now < 5:
        now += 1 / 30
        g.update(now, None, [((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in boxes])
    passed &= check("...each with a laser on their side: player 1 easier, player 2 as it is",
                    g.phase == west.STREET and g.team[0].easy and not g.team[1].easy,
                    f"after {now:.1f}s")

    # High Noon.
    g, _ = two(xs=[])
    g._noon_for = -1
    g.stage = 1
    g.phase = CLEAR
    g._banner = ("", g.now)
    g.update(g.now + 0.05, None, [])
    passed &= check("after the second street, the two meet at High Noon", g.phase == west.NOON)
    for _ in range(2):
        while not g._drawn:
            g.update(g.now + 1 / 30, None, [])
        b = g.bottles[1]
        g.update(g.now + 0.3, None, [(b.x, b.y)])
        for _ in range(70):
            g.update(g.now + 1 / 30, None, [])
    passed &= check("...first to break their bottle twice wins it",
                    g.team[1].noon == 1 and g.phase == CLEAR, str(g.noon_score))
    for _ in range(120):
        g.update(g.now + 1 / 30, None, [])
    passed &= check("...and then on to the next street", g.phase == west.STREET and g.stage == 2)

    g, _ = two(xs=[])
    now = g.now
    for _ in range(int(West.ABSENT * 30) + 30):
        now += 1 / 30
        g.update(now, None, [(400.0, 500.0)] if int(now * 2) % 2 else [])
    passed &= check("a side with no laser for a long time: the game waits for its player",
                    g.state == "paused" and g._absent == 1)
    g.update(now + 0.1, None, [(1400.0, 500.0)])
    passed &= check("...and goes on when that laser is back", g.state == "playing")

    def bot_dots(g, bots, now):
        dots = []
        for side, delay, memo in bots:
            if g.level.ammo and g.phase != west.NOON and g.team[side].ammo == 0:
                target, pt = "reload", (size[0] * (0.25 + 0.5 * side), size[1] - 5.0)
            else:
                best = None
                for t in g.targets:
                    if t.dying is not None or isinstance(t, west.Strip):
                        continue
                    if g.two and not (isinstance(t, west.Figure) and t.kind == west.TEAM):
                        if isinstance(t, west.Bottle):
                            if t.side != side:
                                continue
                        elif (t.x < size[0] / 2) != (side == 0):
                            continue
                    if isinstance(t, west.Figure) and (not t.armed or t.hidden):
                        continue
                    if t.age(now) < (0.25 if t in g.bosses else delay):
                        continue
                    pt = on(t, side if g.two and t.kind == west.TEAM else None) \
                        if isinstance(t, west.Figure) else (t.x, t.y)
                    urgency = t.remaining(now) if t.lifetime < 1e9 else 9
                    if best is None or urgency < best[0]:
                        best = (urgency, t, pt)
                if best is None:
                    memo["t"] = None
                    continue
                _, target, pt = best
            if target is not memo.get("t"):
                memo["t"], memo["since"] = target, now
            if (now - memo["since"]) % 0.5 > 0.4:          # a flash every half second
                dots.append(pt)
        return dots

    def play_west(g, delay, players=1):
        now, canvas, reddest = g.now, overlay.blank(size), 0
        bots = [(i, delay, {}) for i in range(players)]
        while g.state != "over" and now < 900:
            now += 1 / 30
            dots = bot_dots(g, bots, now)
            g.update(now, dots[0] if dots else None, dots)
            if int(now * 30) % 25 == 0:
                g.draw(canvas)
                reddest = max(reddest, reddest_of(canvas))
        g.draw(canvas)
        g.draw_over(canvas, None, None)
        return max(reddest, reddest_of(canvas)), now

    g = West(size, level="deputy", seed=5)
    g.start(0.0)
    reddest, took = play_west(g, 0.7)
    passed &= check("a steady deputy saves the town: six streets, six showdowns",
                    g.state == "over" and g.won and len(g.team[0].draws) == len(STAGES),
                    f"score {g.stats.score}, {took / 60:.1f} minutes, {g.rank()}")
    passed &= check("nothing drawn in the town is red", reddest <= 0)
    g = West(size, players=2, level="deputy", seed=5)
    g.start(0.0)
    g.key(32)
    reddest, took = play_west(g, 0.7, players=2)
    passed &= check("two deputies save it together, with a High Noon or two on the way",
                    g.won and sum(p.noon for p in g.team) == 2
                    and all(p.stats.score > 0 for p in g.team),
                    f"{g.team[0].stats.score} + {g.team[1].stats.score}, "
                    f"{took / 60:.1f} minutes")
    passed &= check("...and each goes home with an award", all(g.awards()),
                    " / ".join(", ".join(a) for a in g.awards()))
    passed &= check("nothing drawn for two is red", reddest <= 0)
    g = West(size, level="marshal", seed=5)
    g.start(0.0)
    play_west(g, 99.0)
    passed &= check("a marshal who never shoots loses the town",
                    g.state == "over" and not g.won and g.lives == 0 and g.rank() == "ЖОВТОДЗЬОБ")

    pieces = [audio.Player(west.SOUNDS).length(f"music{n + 1}") for n in range(3)]
    if not any(pieces):
        print("  SKIP  no music (tools/west_audio.py) - the town plays without")
    else:
        passed &= check("each piece of music is a piece of its own",
                        all(pieces) and len(set(round(l, 1) for l in pieces)) == len(pieces),
                        "  ".join(f"{l:.0f}s" for l in pieces))

    class Tape(audio.Player):
        def __init__(self):
            super().__init__(west.SOUNDS)
            self.played = []

        def music(self, name):
            if not self.played or self.played[-1] != name:
                self.played.append(name)

    g = West(size, level="deputy", seed=5)
    g.sound = Tape()
    g.start(0.0)
    play_west(g, 0.5)
    passed &= check("...and each street plays its own",
                    g.sound.played == [f"music{sc.music}" for sc in west_art.SCENES]
                    or [n for i, n in enumerate(g.sound.played)
                        if i == 0 or n != g.sound.played[i - 1]] ==
                    ["music1", "music2", "music3"], " ".join(g.sound.played))

    print("balloon math:")
    import random

    def evaluate(text):
        a, op, b = text.split()
        a, b = int(a), int(b)
        return a + b if op == "+" else a - b if op == "-" else a * b

    limits = {1: 20, 2: 100, 3: 81}
    for level in (1, 2, 3):
        rng_ = random.Random(level)
        tasks = [make_task(rng_, level, 3 if level == 1 else 4) for _ in range(500)]
        sound = all(evaluate(t.text) == t.answer and t.choices.count(t.answer) == 1
                    and len(set(t.choices)) == len(t.choices) and min(t.choices) >= 0
                    for t in tasks)
        in_range = all(0 <= t.answer <= limits[level] and
                       all(int(n) <= limits[level] for n in t.text.split()[::2])
                       for t in tasks)
        passed &= check(f"level {level}: one right answer among distinct choices", sound,
                        f"e.g. {tasks[0].text} = {tasks[0].answer} from {tasks[0].choices}")
        passed &= check(f"level {level}: numbers stay within {limits[level]}", in_range)

    def balloons(level=1, **kw):
        b = BalloonMath((1920, 1080), level, seed=5, **kw)
        b.start(0.0)
        b.update(b.started + 0.05, None)
        for _ in range(40):                       # let them float into place
            b.update(b.now + 0.05, None)
        return b

    def hold(b, target, seconds=0.7):
        for _ in range(int(seconds / 0.05)):
            if target not in b.targets or target.dying is not None:
                break
            b.update(b.now + 0.05, (target.x, target.y))

    def pick(b, right=True):
        return next(t for t in b.targets if t.dying is None and t.tried is None
                    and (int(t.label) == b.task.answer) == right)

    b = balloons()
    right = pick(b)
    b.update(b.now + 0.05, None)
    b.update(b.now + 0.05, (right.x, right.y))
    passed &= check("a flash alone does not pick a balloon", not b.results)
    hold(b, right)
    passed &= check("holding on the right balloon solves the sum", b.results == [True])

    b = balloons()
    right = pick(b)
    for i in range(1, 40):
        if b.results:
            break
        b.update(b.now + 1 / 30, None if i % 8 == 0 else (right.x, right.y))
    passed &= check("...and so does a hold that loses frames", b.results == [True],
                    f"after {i / 30:.1f}s")

    b = balloons()
    wrong = pick(b, right=False)
    hold(b, wrong)
    passed &= check("a wrong balloon costs nothing and the sum stays",
                    not b.results and b.task is not None and wrong.tried is not None)
    hold(b, wrong)
    passed &= check("a tried balloon cannot be picked again", b.stats.misses == 1,
                    f"{b.stats.misses} wrong picks counted")
    hold(b, pick(b))
    passed &= check("...and the sum no longer counts as first-time", b.results == [False])

    def play(first_wrong: int):
        """A whole round; the bot picks a wrong balloon first on some sums."""
        b = BalloonMath((1920, 1080), 2, seed=9)
        b.start(0.0)
        canvas = overlay.blank((1920, 1080))
        now = b.started
        while b.state != "over" and now < 600:
            now += 0.05
            aim = None
            if b.task is not None and b.targets:
                clean = len(b.results) >= first_wrong or not b._clean
                aim = pick(b, right=clean)
            b.update(now, (aim.x, aim.y) if aim else None)
            b.draw(canvas)
        b.draw_over(canvas, None, None)
        return b

    b = play(0)
    passed &= check("a perfect round earns three stars",
                    b.state == "over" and b.first_try == 10 and b.stars == 3,
                    f"{b.first_try}/10, {b.stars} stars")
    b = play(3)
    passed &= check("three slips still earn two stars", b.first_try == 7 and b.stars == 2,
                    f"{b.first_try}/10, {b.stars} stars")
    b = play(10)
    passed &= check("every round ends with at least one star",
                    b.state == "over" and b.first_try == 0 and b.stars == 1)

    print("number hunt:")

    def value(label):
        from fractions import Fraction
        return Fraction(label) if "/" in label else int(label)

    # Each rule is re-checked here against an independent statement of it.
    def oracle(text):
        w = text.split()
        if text.startswith("кратні"):
            return lambda v: v % int(w[1]) == 0
        if text.startswith("дільники"):
            return lambda v: int(w[1]) % v == 0
        if text in ("парні числа", "непарні числа"):
            return lambda v: v % 2 == (0 if w[0] == "парні" else 1)
        if text == "прості числа":
            return lambda v: v in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
        if text.startswith("дорівнюють"):
            return lambda v: v == value(w[1])
        if text == "квадрати чисел":
            return lambda v: round(v ** 0.5) ** 2 == v
        if text.startswith("степені числа"):
            return lambda v: v in [int(w[2]) ** e for e in range(1, 12)]
        expr = text.replace("x", "*(v)").replace("|*(v)|", "abs(v)").replace(" *(v)", " (v)")
        expr = expr[1:] if expr.startswith("*") else expr
        return lambda v: eval(expr, {"v": v, "abs": abs})

    for pack, makers in PACKS.items():
        bad, seen = [], set()
        for seed in range(60):
            for make in makers:
                rule = make(random.Random(seed))
                seen.add(rule.text)
                truth = oracle(rule.text)
                if (len(rule.yes) < 3 or len(rule.no) < 3 or set(rule.yes) & set(rule.no)
                        or not all(truth(value(v)) for v in rule.yes)
                        or any(truth(value(v)) for v in rule.no)):
                    bad.append(rule.text)
        passed &= check(f"pack {pack}: every rule sorts its numbers correctly", not bad,
                        f"{len(seen)} rules" if not bad else f"wrong: {sorted(set(bad))}")

    def hunt(**kw):
        h = NumberHunt((1920, 1080), "5-6", seed=4, **kw)
        h.start(0.0)
        h.update(h.started + 0.5, None)
        h.targets.clear()
        return h

    def flash(h, t):
        h.update(h.now + 0.05, None)
        h.update(h.now + 0.05, (t.x, t.y))

    h = hunt(); t = h.spawn(h.now); t.kind = NORMAL
    flash(h, t)
    passed &= check("a number that fits scores", h.stats.hits == 1 and h.stats.score > 0
                    and h.lives == 3, f"score {h.stats.score}")
    h = hunt(); t = h.spawn(h.now); t.kind = DECOY; h.stats.score = 100
    flash(h, t)
    passed &= check("a number that does not fit costs points and a life",
                    h.stats.score == 75 and h.lives == 2)
    h = hunt(); t = h.spawn(h.now); t.kind = NORMAL; t.lifetime = 0.1
    h.next_spawn = h.now + 99
    h.update(h.now + 0.2, None)
    passed &= check("a fitting number that gets away costs a life", h.lives == 2)
    h = hunt(); t = h.spawn(h.now); t.kind = DECOY; t.lifetime = 0.1
    h.next_spawn = h.now + 99
    h.update(h.now + 0.2, None)
    passed &= check("a non-fitting number may leave freely", h.lives == 3)

    # A whole round by a bot that knows the rule: it should never lose a life.
    h = NumberHunt((1920, 1080), "7-9", seed=2)
    h.start(0.0)
    now, canvas = 0.0, overlay.blank((1920, 1080))
    rules, kinds, mislabelled, penalised = [], [], 0, 0
    known = []
    while h.state != "over" and now < 120:
        if h.rule.text not in rules:
            if rules:                     # targets on screen at the change
                penalised += 3 - h.lives
            rules.append(h.rule.text)
        for t in h.targets:
            if not any(t is k for k in known):
                known.append(t)
                kinds.append(t.kind)
                pool = h.rule.yes if t.kind == NORMAL else h.rule.no
                mislabelled += t.label not in pool
        for _ in range(20):
            now += 1 / 60
            h.update(now, None)
        target = next((t for t in h.targets if t.dying is None and t.kind == NORMAL), None)
        now += 1 / 60
        h.update(now, (target.x, target.y) if target else None)
        h.draw(canvas)
    h.draw_over(canvas, None, None)
    share = kinds.count(NORMAL) / max(1, len(kinds))
    passed &= check("a full round plays out", h.stats.hits > 5 and h.lives == 3,
                    f"{h.stats.hits} hits, score {h.stats.score}, lives left {h.lives}")
    passed &= check("the rule changes during the round", len(rules) == 4 and len(set(rules)) == 4,
                    " / ".join(rules))
    passed &= check("a rule change costs no lives", penalised == 0)
    passed &= check("every target is labelled for the rule in force", mislabelled == 0)
    passed &= check("about four targets in ten fit the rule", 0.3 <= share <= 0.55,
                    f"{share:.0%} of {len(kinds)}")

    print("game menu:")
    menu = GameMenu((1920, 1080))
    boxes = menu.boxes()
    centre = lambda box: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    passed &= check("tiles fit on screen", len(boxes) == len(GAMES) and
                    all(0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080 for x0, y0, x1, y1 in boxes))
    apart = all(not (a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])
                for i, a in enumerate(boxes) for b in boxes[i + 1:])
    passed &= check("the games are in two rows, shooting and learning, none crowded",
                    [key for key, _ in menu.rows()] == [SHOOTING, LEARNING] and apart
                    and all(len(ids) <= GameMenu.MOST for _, ids in menu.rows())
                    and min(x1 - x0 for x0, _, x1, _ in boxes) >= 300,
                    f"tiles {min(x1 - x0 for x0, _, x1, _ in boxes)}px wide")
    roomy = GameMenu((1920, 1080))
    roomy.choose(GAMES.index(entry("Дикий Захід")))
    passed &= check("the Wild West's levels are a table: a row for one player, a row for two",
                    [(key, len(ids)) for key, ids in roomy.rows()] ==
                    [("1 гравець", 3), ("2 гравці", 3)])
    now, picked = 0.0, None
    while now < 3 and menu.page is None:
        now += 0.05
        picked = menu.update(now, centre(boxes[0]))
    passed &= check("holding on the gallery opens its two games",
                    picked is None and menu.page == 0 and len(menu.boxes()) == 2,
                    f"after {now:.1f}s")
    passed &= check("...the targets and the silhouette range",
                    [type(menu.choose(i)((1920, 1080))) for i in (0, 1)] == [Game, Silhouette])
    menu = GameMenu((1920, 1080))
    balloons = GAMES.index(entry("Математичні кульки"))
    now, picked = 0.0, None
    while now < 3 and menu.page is None:
        now += 0.05
        picked = menu.update(now, centre(boxes[balloons]))
    opened = now
    for _ in range(10):                           # the same hold, half a second on
        now += 0.05
        picked = picked or menu.update(now, centre(boxes[balloons]))
    passed &= check("a game with levels opens its level page first",
                    picked is None and menu.page == balloons, f"after {opened:.1f}s")
    while now < opened + 3 and picked is None:
        now += 0.05
        picked = menu.update(now, centre(menu.boxes()[1]))
    passed &= check("...and a second hold picks the level",
                    picked is not None and picked((1920, 1080)).level == 2)
    # A real laser: the detector loses the dot now and then, the hand shakes
    # it over the tile's edge, and a reflection shows up somewhere else.
    flaky = GameMenu((1920, 1080))
    x0, y0, x1, y1 = boxes[0]
    now, step = 0.0, 0
    while now < 3 and flaky.page is None:
        now += 1 / 30
        step += 1
        point = centre(boxes[0])
        if step % 9 in (0, 1, 2, 3):
            point = None                          # 130 ms without a detection
        elif step % 9 == 5:
            point = (x1 + 8, (y0 + y1) / 2)       # just off the edge
        elif step % 9 == 7:
            point = centre(boxes[2])              # one stray frame elsewhere
        flaky.update(now, point)
    passed &= check("a flickering, shaky hold still picks the tile", flaky.page == 0,
                    f"after {now:.1f}s")
    flaky = GameMenu((1920, 1080))
    now = 0.0
    while now < 3:
        now += 1 / 30
        flaky.update(now, centre(boxes[0]) if 0.8 <= now < 1.4 else None)
    passed &= check("...but a hold that is let go picks nothing",
                    flaky.page is None and flaky._hover is None)
    made = [o.start((1920, 1080)) for e in GAMES for o in e.options]
    names = [g.scores_name for g in made]
    passed &= check("every menu entry builds its game",
                    [type(g).__name__ for g in made] ==
                    ["Game", "Silhouette"] + ["West"] * 6 + ["Duel", "JarRange", "QuickDraw"]
                    + ["BalloonMath"] * 3 + ["NumberHunt"] * 2 + ["Story"] * 2 + ["TeacherSettings"])
    passed &= check("each scored game keeps its own table",
                    names == ["highscores", "highscores-silhouette"]
                    + [f"highscores-west-{n}p-{l}" for n in (1, 2)
                       for l in ("deputy", "sheriff", "marshal")]
                    + [None] * 6 + ["highscores-hunt-5-6", "highscores-hunt-7-9", None, None, None],
                    str(names))
    passed &= check("ESC steps back a page, then out", menu.back() and not menu.back())

    end = EndButtons((1920, 1080), 0.0)
    again, other = (centre(b) for b in end.boxes())
    now, picked = 0.0, None
    while now < EndButtons.SETTLE - 0.1:
        now += 1 / 30
        picked = picked or end.update(now, [again])
    passed &= check("at the end, the buttons wait for the last shots to be over",
                    picked is None)
    while now < 4 and picked is None:
        now += 1 / 30
        picked = end.update(now, [(300.0, 300.0), again])   # either laser will do
    passed &= check("...then a laser held on PLAY AGAIN plays again", picked == AGAIN,
                    f"after {now - EndButtons.SETTLE:.1f}s")
    end = EndButtons((1920, 1080), 0.0)
    now, picked = 0.0, None
    while now < 5 and picked is None:
        now += 1 / 30
        picked = end.update(now, [other] if int(now * 30) % 5 else [])   # a flicker
    passed &= check("...and one on OTHER GAMES goes back to the chooser", picked == OTHER)
    own = {type(o.start((1920, 1080))).__name__: getattr(o.start((1920, 1080)), "own_buttons", False)
           for e in GAMES for o in e.options}
    passed &= check("every game ends with them - but the duels, which have REMATCH",
                    {n for n, v in own.items() if v} == {"Duel", "JarRange", "QuickDraw"})
    menu.draw(overlay.blank((1920, 1080)))

    print("story game:")
    size = (1920, 1080)
    english = script.ENGLISH
    offered = [o.name for o in entry("Історія").options]
    passed &= check("the chooser offers the story in each language the font can draw",
                    offered == [l.name for l in script.LANGUAGES.values()
                                if overlay.UNICODE or l.name.isascii()], ", ".join(offered))
    lines = english.all_lines()
    ids = [l.id for l in lines]
    passed &= check("every line has its own id", len(set(ids)) == len(ids), f"{len(lines)} lines")
    passed &= check("every chapter has its scenes", len(english.chapters) == len(LEVELS) and
                    all(c.intro and c.outro for c in english.chapters))
    fields = lambda text: sorted(re.findall(r"{(\w+)}", text))
    for code, lang in script.LANGUAGES.items():
        said = lang.all_lines()
        passed &= check(f"{lang.name}: every line and every label is there",
                        [l.id for l in said] == ids and all(l.text.strip() for l in said) and
                        lang.ui.keys() == english.ui.keys() and
                        all(fields(lang.ui[k]) == fields(english.ui[k]) for k in english.ui),
                        f"{len(said)} lines, {len(lang.ui)} labels")
        player = audio.Player(ASSETS / code)
        if player.texts:
            silent = [l.id for l in said if player.length(l.id, l.text) is None]
            spoken = sum(player.length(l.id) or 0.0 for l in said)
            passed &= check(f"{lang.name}: every line's recording says what the script says",
                            not silent, ", ".join(silent) or f"{spoken / 60:.1f} minutes of speech")
        else:
            print(f"  SKIP  {lang.name}: no recordings (build_story_audio.sh) - it plays as text")

    def level(cls, seed=5):
        """A chapter, just past its countdown."""
        g = cls(size, seed=seed)
        g.start(0.0)
        g.update(g.started + 0.01, None)
        return g

    def run(g, seconds, aim=None):
        end = g.now + seconds
        while g.now < end and g.state != "over":
            g.update(g.now + 1 / 30, aim(g) if aim else None)

    def shoot(g, t):
        """A flash: the beam off for half a second, then on, on the target."""
        g.update(g.now + 0.5, None)
        g.update(g.now + 0.03, (t.x, t.y))

    def bot(g):
        """Where a competent keeper would be pointing right now."""
        live = [t for t in g.targets if t.dying is None and t.kind != MOTH]
        if isinstance(g, Ships):
            boats = [t for t in live if not t.guided]
            return (boats[0].x, boats[0].y) if boats else None
        if isinstance(g, Constellations):
            live = [t for t in live if t.order == g.progress] if g.phase == g.PLAY else []
        eye = [t for t in live if t.kind == EYE]
        if eye:
            return eye[0].x, eye[0].y
        if not live or g.counting_down or g.now % 0.6 < 0.5:
            return None                 # dark, then three frames on the target
        t = min(live, key=lambda t: t.born)
        return t.x, t.y

    # 1 - sparks
    g = level(Sparks)
    run(g, 200, bot)
    passed &= check("chapter 1: catching twenty sparks wins it",
                    g.won and g.caught == 20 and g.stars == 3, f"{g.caught} caught")
    g = level(Sparks)
    run(g, 200)
    passed &= check("...and eight lost to the wind lose it",
                    g.state == "over" and not g.won and g.escaped == 8, f"{g.escaped} escaped")

    # 2 - the siege
    g = level(Siege)
    run(g, 4.0)
    foe = next(t for t in g.targets if t.kind == GLOAMLING)
    far = math.hypot(foe.x - g.centre[0], foe.y - g.centre[1])
    run(g, 1.0)
    passed &= check("chapter 2: gloamlings make for the cage",
                    math.hypot(foe.x - g.centre[0], foe.y - g.centre[1]) < far - 50)
    while g.lives == g.LIVES and g.now < 60:
        run(g, 0.1)
    passed &= check("one that gets there puts out a spark", g.lives == g.LIVES - 1,
                    f"after {g.elapsed:.0f}s")
    g = level(Siege)
    brute = g.spawn(g.now, BRUTE, (300.0, 300.0))
    shoot(g, brute)
    halves = [t for t in g.targets if t.dying is None]
    passed &= check("a big one comes apart into two small ones",
                    len(halves) == 2 and all(t.radius < brute.radius / 1.5 for t in halves))
    g = level(Siege)
    g._spawn_moth(g.now)
    friend = g.targets[-1]
    run(g, 1.0)
    shoot(g, friend)
    passed &= check("shooting a glow-moth costs a spark",
                    g.lives == g.LIVES - 1 and g.stats.decoys == 1)
    g = level(Siege)
    run(g, 200, bot)
    passed &= check("three waves beaten back win it", g.won and g.wave == 2 and g.stars == 3,
                    f"{g.stats.hits} hits, {g.lives} sparks left")
    g = level(Siege)
    run(g, 6.0)
    before = (g.elapsed, [(t.x, t.y) for t in g.targets])
    g.toggle_pause(g.now)
    for _ in range(30):
        g.update(g.now + 1.0, None)
    g.toggle_pause(g.now)
    passed &= check("pausing a chapter stops its clock and everything in it",
                    abs(g.elapsed - before[0]) < 0.01 and
                    before[1] == [(t.x, t.y) for t in g.targets])

    # 3 - the ships
    g = level(Ships)
    run(g, 1.0)
    first = g.targets[0]
    shoot(g, first)
    passed &= check("chapter 3: a flash does not guide a boat", not first.guided)
    run(g, 2.0, lambda g: (first.x, first.y))
    passed &= check("...light held on it does, and it turns for the channel",
                    first.guided and (first.vy > 0) == (g.channel > first.y))
    while first in g.targets and g.now < 60:
        run(g, 0.1)
    passed &= check("...and comes home through the gap", g.safe == 1 and g.lives == g.LIVES,
                    f"after {g.elapsed:.0f}s")
    g = level(Ships)
    run(g, 12.0)
    passed &= check("a boat nobody lights is lost on the reef", g.lost == 1 and g.lives == 2)
    g = level(Ships)
    # A hand that shakes: every third frame the light is off the boat.
    run(g, 200, lambda g: bot(g) if int(g.now * 30) % 3 else None)
    passed &= check("the whole fleet can be brought home", g.won and g.safe == 9 and g.stars == 3,
                    f"{g.safe} home, {g.lost} lost")

    # 4 - the constellations
    g = level(Constellations)
    shoot(g, g.targets[0])
    passed &= check("chapter 4: nothing can be lit while the pattern is shown",
                    g.phase == g.SHOW and g.progress == 0 and g.lives == g.LIVES)
    while g.phase != g.PLAY:
        run(g, 0.1)
    shoot(g, next(t for t in g.targets if t.order == 2))
    passed &= check("a star out of order costs a spark, and the pattern is shown again",
                    g.lives == g.LIVES - 1 and g.progress == 0 and g.phase == g.SHOW)
    run(g, 200, bot)
    passed &= check("tracing all three wins it - two stars, after that slip",
                    g.won and g.index == 2 and g.stars == 2)
    g = level(Constellations)
    run(g, 200, bot)
    passed &= check("...and three for a clean trace", g.won and g.stars == 3)
    close = min(math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius
                for _, path, _, strays in SKY
                for stars in [[Star(x=u * size[0], y=v * size[1], radius=g.unit * 0.05, born=0,
                                    lifetime=1) for u, v in path + strays]]
                for i, a in enumerate(stars) for b in stars[i + 1:])
    passed &= check("no two stars are close enough to be confused", close > 20,
                    f"closest rims {close:.0f} px apart")

    # 5 - the storm
    g = level(Storm)
    while not g.targets:
        run(g, 0.1)
    shoot(g, g.targets[0])
    passed &= check("chapter 5: striking an open knot hurts the Gloam",
                    g.health == g.HEALTH - 1)
    g = level(Storm)
    run(g, 6.0)
    thrown = [t for t in g.targets if t.kind == GLOAMLING]
    passed &= check("a knot left alone closes, and sends a gloamling at the lamp",
                    len(thrown) == 1 and thrown[0].goal == g.lamp and g.health == g.HEALTH)
    g._throw(g.now, g.heart)
    passed &= check("...but never two in quick succession",
                    len([t for t in g.targets if t.kind == GLOAMLING]) == 1)
    g = level(Storm)
    g.health = 1
    while not g.targets:
        run(g, 0.1)
    shoot(g, g.targets[0])
    eye = next((t for t in g.targets if t.kind == EYE), None)
    passed &= check("with its strength gone, the eye opens", g.final and eye is not None)
    shoot(g, eye)
    passed &= check("a flash is not enough for the eye", eye.dying is None and not g.ending)
    run(g, 1.5, lambda g: (eye.x, eye.y))
    held = eye.dying is None
    run(g, 1.0, lambda g: (eye.x, eye.y))
    passed &= check("...two seconds of light are", held and g.won and eye.dying is not None)
    g = level(Storm)
    run(g, 300, bot)
    passed &= check("the whole fight can be won", g.won and g.stars == 3,
                    f"in {g.elapsed:.0f}s, {g.lives} sparks left")

    # buttons
    hold, box, pressed = Hold(), [(100, 100, 300, 200)], []
    for step in range(60):
        on = (200.0, 150.0) if step % 6 else None       # the dot drops out now and then
        pressed.append(hold.update(1 / 30, on, box))
    passed &= check("a held button is pressed once, through lost frames",
                    [p for p in pressed if p is not None] == [0],
                    f"after {pressed.index(0) / 30:.1f}s")
    hold = Hold()
    pressed = [hold.update(1 / 30, (200.0, 150.0) if step < 20 else None, box)
               for step in range(60)]
    passed &= check("...and one that is let go is not", not any(p is not None for p in pressed)
                    and hold.dwell == 0.0)

    # the story around the chapters
    with tempfile.TemporaryDirectory() as tmp:
        def story():
            s = Story(size, seed=2)
            s.home = Path(tmp)
            s.start(0.0)
            return s

        def step(s, seconds, point=None):
            end = s.now + seconds
            while s.now < end:
                s.update(s.now + 1 / 30, point)

        middle = lambda box: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        s = story()
        boxes = s.title_boxes()
        passed &= check("the story opens on its chapters, only the first unlocked",
                        s.stage == TITLE and s.progress.unlocked == 1 and len(boxes) == 5 and
                        all(0 <= x0 < x1 <= size[0] for x0, _, x1, _ in boxes))
        step(s, 3.0, middle(boxes[1]))
        passed &= check("a locked chapter cannot be chosen",
                        s.stage == TITLE and not s.key(ord("2")))
        step(s, 3.0, middle(boxes[0]))
        passed &= check("holding the pointer on a chapter begins its scene",
                        (s.stage, s.chapter, s.line.id) == (INTRO, 0, "c1i01"))
        step(s, s._line_len + 1.0)
        passed &= check("a line gives way to the next when it has been said",
                        s.line_index == 1, s.line.id)
        words = middle(s.dialogue_box())
        step(s, 0.6)
        step(s, 1.0, words)
        passed &= check("a flash on the words asks for the next line - a resting beam does not",
                        s.line_index == 2, s.line.id)
        at = s.clock
        s.toggle_pause(s.now)
        step(s, 60.0)
        s.toggle_pause(s.now)
        passed &= check("pausing holds a scene where it is",
                        s.line_index == 2 and s.clock == at and s.state == "playing")
        passed &= check("a scene does not need to see the laser - only a chapter does",
                        not s.needs_sight)
        step(s, 1.5, middle(s.skip_box()))
        passed &= check("holding on 'skip' goes straight to the chapter",
                        s.stage == PLAY and isinstance(s.level, Sparks) and s.needs_sight)
        while s.stage == PLAY:
            s.update(s.now + 1 / 30, bot(s.level))
        saved = json.loads((Path(tmp) / "story.json").read_text())
        passed &= check("a chapter won is recorded, and unlocks the next",
                        s.stage == RESULT and saved["unlocked"] == 2 and
                        saved["best"]["1"]["stars"] == 3, str(saved))
        step(s, 2.0, middle(s.button_box()))
        passed &= check("...and is followed by the scene after it",
                        (s.stage, s.line.id) == (OUTRO, "c1o01"))
        passed &= check("ENTER skips a scene, into the next chapter's",
                        s.key(13) and (s.stage, s.chapter, s.line.id) == (INTRO, 1, "c2i01"))
        s.key(13)
        while s.stage == PLAY:
            s.update(s.now + 1 / 30, None)          # nobody at the shard
        passed &= check("a chapter lost offers another go, and unlocks nothing",
                        (s.stage, s.line.id) == (RETRY, "c2f") and s.progress.unlocked == 2)
        lost = s.level
        step(s, 2.0, middle(s.button_box()))
        passed &= check("...which starts it afresh",
                        s.stage == PLAY and s.level is not lost and s.level.lives == Siege.LIVES)

        s = story()
        passed &= check("the story is picked up where it was left",
                        s.progress.unlocked == 2 and s.progress.stars(0) == 3 and
                        s.progress.stars(1) == 0)
        # The whole of it, start to finish - and everything it puts on screen.
        s.key(ord("1"))
        frames, reddest, stages = 0, -255, set()
        canvas = overlay.blank(size)
        while s.state != "over" and s.now < 3000:
            if s.stage in (INTRO, OUTRO) and s.clock - s._line_at > 0.4:
                s.key(32)
            elif s.stage == RESULT and s.clock - s._stage_at > 0.5:
                s.key(32)
            s.update(s.now + 1 / 30, bot(s.level) if s.stage == PLAY else None)
            frames += 1
            if frames % 12 == 0:
                s.draw(canvas)
                stages.add(s.stage)
                reddest = max(reddest, int((canvas[..., 2].astype(np.int16)
                                            - canvas[..., 1]).max()))
        s.draw(canvas)
        s.draw_over(canvas, None, None)
        passed &= check("the story can be played from the first chapter to the end",
                        s.state == "over" and s.stage == END and s.progress.finished and
                        s.progress.unlocked == 5 and
                        all(s.progress.stars(i) >= 2 for i in range(5)) and s.total > 0,
                        f"{s.now / 60:.0f} minutes without its scenes, score {s.total}")
        passed &= check("nothing it draws has more red in it than green",
                        reddest <= 0 and stages == {INTRO, PLAY, RESULT, OUTRO},
                        f"red - green at most {reddest}")
        # Scenery is rendered when first shown, and each scene has its own.
        scenery = story_art.Backdrop(size)
        for name in scenery.SCENES:
            scenery.draw(canvas, name, 3.0)
            reddest = max(reddest, int((canvas[..., 2].astype(np.int16) - canvas[..., 1]).max()))
        used = {l.scene for l in lines if l.scene} | {cls.SCENE for cls in LEVELS}
        # The same story in another language: every screen of it, drawn.
        s = Story(size, seed=2, language="uk")
        s.start(0.0)
        s.draw(canvas)
        shown = {s.stage}
        while s.state != "over" and s.now < 3000:
            if s.stage == TITLE:
                s.key(ord("1"))
            elif s.stage in (INTRO, OUTRO, RESULT) and s.clock - s._stage_at > 0.3:
                s.key(13)
            s.update(s.now + 1 / 30, bot(s.level) if s.stage == PLAY else None)
            if s.stage not in shown or int(s.now * 30) % 45 == 0:
                shown.add(s.stage)
                s.draw(canvas)
        s.draw_over(canvas, None, None)
        passed &= check("...and in Ukrainian", s.state == "over" and
                        s.words.chapters[0].title == "Іскри за вітром" and
                        s.level.words is s.words and len(shown) == 6, ", ".join(sorted(shown)))
        passed &= check("every scene the script names can be drawn",
                        used <= set(scenery.SCENES) and reddest <= 0, ", ".join(sorted(used)))

    class Tape:
        """Stands in for the music player, and notes what it is asked to do."""
        def __init__(self):
            self.log, self.on = [], None
        def music(self, name):
            if name != self.on:
                self.log.append(name)
            self.on = name
        def hold(self):
            self.log.append("hold")
        def release(self):
            self.log.append("release")
        def stop(self):
            self.log.append("stop")
            self.on = None

    s = Story(size, seed=2)
    s.start(0.0)
    s.music = tape = Tape()
    s.key(ord("1"))
    s.update(s.now + 0.5, None)
    passed &= check("a scene is left to the voices", s.stage == INTRO and not tape.log)
    s.key(13)
    s.update(s.now + 0.1, None)
    s.toggle_pause(s.now)
    s.toggle_pause(s.now + 5)
    s.update(s.now + 5.1, None)
    passed &= check("a chapter plays its own music, and pausing holds it",
                    s.stage == PLAY and tape.log == ["c1", "hold", "release"], str(tape.log))
    s.level.state = "over"                              # lost, as it happens
    s.update(s.now + 0.1, None)
    s.close()
    passed &= check("...which stops when the chapter does",
                    s.stage == RETRY and tape.log[3:] == ["stop", "stop"], str(tape.log))
    pieces = audio.Player(ASSETS / "music")
    lengths = [pieces.length(f"c{n + 1}") for n in range(len(LEVELS))]
    if not any(lengths):
        print("  SKIP  no music (tools/story_music.py) - the chapters play without")
    else:
        passed &= check("every chapter has its piece of music",
                        all(l and l > 30 for l in lengths),
                        "  ".join(f"{l:.0f}s" if l else "none" for l in lengths))

    print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
