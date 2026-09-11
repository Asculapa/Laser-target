"""Offline sanity checks - no camera or display needed.

Renders synthetic frames with a fake laser dot, runs the detector on them,
solves a homography from the results and checks the round-trip accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from laserapp import overlay
from laserapp.calibration import grid_points, solve
from laserapp.lens import LensModel
from laserapp.autocal import AutoCalibration, find_dot
from laserapp.game import Game, HighScores, Stats, NORMAL, BONUS, DECOY
from laserapp.detector import LaserDetector


def synth_frame(size=(1280, 720), dot=None, noise=True, glare=0.0, bands=0.0,
                dot_strength=1.0, decoy=False):
    """A camera's view of a screen, with optional sunlight and display banding.

    `glare` washes one corner out the way sunlight on a screen does; `bands`
    adds the drifting rolling-shutter stripes a camera picks up when pointed
    straight at a display.
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
    if dot is not None:
        # A laser drives the red channel almost alone; only a strong spot
        # blows its core out to white.
        x, y = dot
        layer = np.zeros_like(img)
        cv2.circle(layer, (int(x), int(y)), 7,
                   (int(20 * dot_strength), int(20 * dot_strength), int(255 * dot_strength)), -1)
        if dot_strength > 0.8:
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
    g.update(g.now + 0.3, (100.0, 100.0))
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
        now += 1 / 60
        target = next((t for t in g.targets if t.dying is None and t.kind != DECOY), None)
        # off for a frame, then on the target: that is a flash
        g.update(now, None)
        now += 1 / 60
        g.update(now, (target.x, target.y) if target else None)
        g.draw(canvas)
    st = g.stats
    passed &= check("a full round plays out", st.hits > 5 and st.score > 0,
                    f"{st.hits} hits, score {st.score}, accuracy {st.accuracy:.0%}, "
                    f"best streak {st.best_combo}, lives left {g.lives}")
    passed &= check("targets stay on screen",
                    all(0 <= t.x <= 1920 and 0 <= t.y <= 1080 for t in g.targets))

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

    print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
