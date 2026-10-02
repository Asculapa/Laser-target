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
from laserapp.camera import FrameDecoder, jpeg_intact
from laserapp.game import Game, HighScores, Stats, NORMAL, BONUS, DECOY
from laserapp.mathgames import PACKS, BalloonMath, NumberHunt, make_task
from laserapp.menu import GAMES, GameMenu
from laserapp.silhouette import CENTRE_V, RING, Silhouette, zone
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
                    s.results == [0] and s.stats.escaped == 1 and t.label == "too slow")

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
        if text.startswith("multiples of"):
            return lambda v: v % int(w[2]) == 0
        if text.startswith("divisors of"):
            return lambda v: int(w[2]) % v == 0
        if text in ("even numbers", "odd numbers"):
            return lambda v: v % 2 == (0 if w[0] == "even" else 1)
        if text == "prime numbers":
            return lambda v: v in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)
        if text.startswith("equal to"):
            return lambda v: v == value(w[2])
        if text == "perfect squares":
            return lambda v: round(v ** 0.5) ** 2 == v
        if text.startswith("powers of"):
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
    now, picked = 0.0, None
    while now < 3 and menu.page is None:
        now += 0.05
        picked = menu.update(now, centre(boxes[1]))
    opened = now
    for _ in range(10):                           # the same hold, half a second on
        now += 0.05
        picked = picked or menu.update(now, centre(boxes[1]))
    passed &= check("a game with levels opens its level page first",
                    picked is None and menu.page == 1, f"after {opened:.1f}s")
    while now < opened + 3 and picked is None:
        now += 0.05
        picked = menu.update(now, centre(boxes[1]))
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
    made = [f((1920, 1080)) for e in GAMES for _, f in e.options]
    names = [g.scores_name for g in made]
    passed &= check("every menu entry builds its game",
                    [type(g).__name__ for g in made] ==
                    ["Game", "Silhouette"] + ["BalloonMath"] * 3 + ["NumberHunt"] * 2)
    passed &= check("each scored game keeps its own table",
                    names == ["highscores", "highscores-silhouette", None, None, None,
                              "highscores-hunt-5-6", "highscores-hunt-7-9"], str(names))
    passed &= check("ESC steps back a page, then out", menu.back() and not menu.back())
    menu.draw(overlay.blank((1920, 1080)))

    print("\n" + ("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
