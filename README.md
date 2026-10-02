# Laser Target

Points a webcam at your screen (or a projected copy of it), finds a red laser
dot in the camera image, and draws an animated reticle on the screen exactly
where the laser is. Includes [games](#games) - a shooting gallery, a
silhouette range, two maths games for school classes, a voiced story in
five chapters and a split-screen duel for two lasers - press **G**.

Alignment comes from a **homography** solved during calibration: the screen is
a plane, the camera sees it from some angle, so one 3x3 projective transform
maps camera pixels to screen pixels. Because the calibration points are
captured *with the laser itself*, the transform absorbs camera position, lens
scale, rotation and projector keystone all at once.

A homography maps straight lines to straight lines, which a **wide-angle lens
does not** - it bows them outwards ("barrel" distortion). So the distortion is
measured and removed first; see [Wide-angle cameras](#wide-angle-cameras).

## Install

Already done - `.venv/` in this directory has OpenCV and NumPy. To rebuild it
from scratch (this system's Python ships without `pip`/`ensurepip`, so pip is
bootstrapped rather than installed system-wide):

    cd ~/prj/laser-app
    python3 -m venv --without-pip .venv
    curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
    .venv/bin/pip install -r requirements.txt

## Windows

Build a portable bundle (on this Linux machine - no Wine needed):

    ./build_windows.sh            # -> dist/LaserTarget-windows-x64.zip

It pairs the official embeddable Python 3.12 with Windows builds of OpenCV,
NumPy and pygrabber (for camera names), so the target PC needs nothing
installed. Copy the zip over, unzip it anywhere writable (not `Program Files`),
and double-click **`Laser Target.bat`**. Command-line options work the same way:
`"Laser Target.bat" --windowed`. `List cameras.bat` prints what DirectShow sees.
Calibration, high scores and the chosen camera are saved next to the `.bat`.

Windows specifics:

* Cameras are opened through DirectShow; indices are DirectShow's order.
* Exposure is in DirectShow units - log2 seconds, so `-6` is 1/64 s and `-8`
  is darker. **E** / **R** step it by one (half / double the exposure time).
* The app is DPI-aware, so fullscreen covers the display's real pixels even
  with Windows display scaling at 125-150%.
* If a camera shows "cannot open", another program (Teams, Zoom, the Camera
  app) probably has it - close that first.

## Run

    ./run.sh                      # fullscreen, USB camera, preview on
    ./run.sh --list-cameras       # what is plugged in
    ./run.sh --camera genius      # pick by name fragment, or by index: --camera 2
    ./run.sh --windowed           # half-size window, handy while tuning

It opens **fullscreen** and shows a **live camera preview** in the top-right
corner straight away, so you can aim the camera before anything else. The
preview is greyscale on purpose - a colour preview of a screen the camera is
pointed at would feed its own red pixels back into the detector.

### Which camera

With more than one camera plugged in, the first start shows a **camera
chooser**: the cameras on the left, a live view of the highlighted one on the
right. Press **1-9** (or click) to preview a camera, then **ENTER** (or click
it again) to use it. The choice is remembered in `settings.json`, and later
starts go straight to that camera as long as it is still plugged in. Press
**V** at any time to bring the chooser back (**ESC** leaves it and keeps the
old camera), or **N** to just step to the next camera.

`--camera` skips all of that: `--camera 2` or `--camera genius` (index or name
fragment), `--camera pick` to always show the chooser, or `--camera auto` to
**prefer a plugged-in USB camera** over the laptop's built-in one; the
built-in is recognised by its name (`Integrated Webcam`, `FaceTime`, and
similar). `--list-cameras` shows what is available:

    capture devices:
      [0] Integrated_Webcam_HD  (built-in)
      [2] Genius WideCam F100 V2

On Linux, note the index gap: UVC cameras publish several `/dev/video*` nodes each, and
only some of them deliver frames. The listing already filters out the rest, so
use the indices it prints.

Each camera needs its own calibration (different position, lens and
resolution). After switching, the app tells you if the loaded calibration
belongs to the other camera.

**The first run calibrates itself.** With no calibration on file the app
lights a sequence of dots on the screen and finds each one in the camera - no
laser, no printed pattern, about 15 seconds. After that the preview is
straightened and the reticle is aligned. Press **K** to run it again (after
moving the camera, for instance), or **ESC** to skip it.

Then, if the laser is not being detected:

1. **F** - auto-tune the detector (point the laser *away* from the screen
   first; the current view becomes "background" and the thresholds are set
   just above it).
2. **D** - cycle the preview to camera + detector mask: the laser should be
   the only white blob in the mask.

## Calibration

Two ways to do it. Both produce the same thing: a lens model plus a camera ->
screen homography, written to `calibration.json` and reloaded next run.

### Automatic (K) - the default

The screen lights one bright dot at a time at a known position while the
camera watches. Each frame is differenced against a reference frame of the
black screen, which leaves the dot and nothing else - ambient light, glare,
reflections and the screen's own glow all cancel. 20 points, ~15 seconds,
nothing to hold.

That differencing is the reason this works where showing a *chessboard* does
not: a chessboard has to be recognised in a low-contrast, glare-streaked image
of a screen, and on the display here that succeeded on only about one frame in
three. A lit dot merely has to be brighter than the same screen was a moment
ago.

Two details it has to get right, both of which show up on real hardware:

* The window manager needs a moment to make the window fullscreen, and
  anything drawn before that lands somewhere else a fraction of a second
  later. Dots do not start until the window has stopped resizing.
* The camera lags the screen by several frames, so just after moving to a new
  target the *previous* dot is often still the brightest thing in view.
  A reading that has not moved since the last accepted point is that stale
  dot, and is ignored - pairing it with the new screen position would corrupt
  the fit (measured: 599 px mean error without this check, 0.3 px with it).

Measured here, on fresh positions that were not part of the fit: **mean 2.6 px,
median 1.9 px, worst 5.1 px** on a 1920x1080 screen.

### With the laser (C)

Useful when the camera cannot see the whole screen, or when you would rather
calibrate through the exact optical path the laser takes.

1. Press **C**. A bull's-eye appears at the first of 9 grid positions.
2. Aim the laser at its centre and hold still. The green ring fills; when it
   completes the point is captured automatically and the next marker appears.
   **SPACE** captures immediately if the auto-hold is being fussy, **B** goes
   back one point, **ESC** cancels.
3. After the last point the homography is solved (reusing the lens from the
   automatic step if there is one, which is more reliable than re-deriving it
   from a handful of laser samples), the mean error is shown and
   the result is written to `calibration.json` (loaded automatically next run).
   The 9 markers stay on screen for a few seconds so you can verify the
   reticle lands on them.

Recalibrate (**K**) whenever the camera, the screen or the projector is moved. The
mapping is to *window* pixels, so also recalibrate if you switch between
fullscreen and windowed.

## Games

Press **G** for the chooser. Hold the laser on a tile for a second to pick it,
or press its number; games with levels then ask which one. **ESC** steps back.

| | | |
| --- | --- | --- |
| **1** | Shooting Gallery | *targets*: they appear, you shoot them, the round lasts a minute |
| | | *silhouette*: a man-shaped range target with numbered rings, ten aimed shots |
| **2** | Balloon Math | grades 2-4: hold the laser on the balloon with the right answer |
| **3** | Number Hunt | grades 5-9: shoot only the numbers that fit the rule |
| **4** | Story | *The Last Light of Lantern Rock*: five chapters, five different games, told aloud - in English or Ukrainian |
| **5** | Range Duel | grades 8-9, two players: a half of the screen each, shoot the armed figures, spare the rest |

During a game, **G** starts the same game again, **P** pauses and **ESC** goes
back to tracking.

Both maths games show digits and symbols only, so they need no translating.

### Balloon Math (grades 2-4)

A sum is shown at the top and three or four balloons float up, each with a
number. Hold the laser on the right one for half a second - a white ring fills
while you hold. Only a hold counts here, never a flash, so a beam waved across
the screen picks nothing by accident.

Nothing is ever lost. A wrong balloon wobbles, greys out and cannot be picked
again; the sum stays until the right one is found. There is no clock and there
are no lives. A round is ten sums, and ends with one to three stars for how
many were right the first time (nine or ten: three stars, six to eight: two).

| Level | Sums |
| --- | --- |
| 1 | `+` and `-` up to 20, three balloons |
| 2 | `+` and `-` up to 100, four balloons |
| 3 | multiplication table (2-9), four balloons |

The wrong answers are the ones a child would actually arrive at - one or ten
off, or what the other operation would have given - so guessing does not work.

### Number Hunt (grades 5-9)

A rule is shown at the top and numbered targets drift across the screen. Shoot
the numbers that fit the rule; leave the others alone. Flash and dwell both
work, as in the gallery below, and so do streaks and the multiplier.

A round is 60 seconds with three lives. A life goes for shooting a number that
does not fit (which also costs 25 points) and for letting one that fits get
away. All targets look alike until they are shot: the burst is green for a
number that fitted and blue for one that did not. The rule changes every 15
seconds; numbers still on screen at that moment leave without costing
anything. Targets get smaller and quicker as the round goes on.

| Pack | Rules |
| --- | --- |
| grades 5-6 | multiples of *k*, divisors of *n*, even / odd, primes, fractions equal to 1/2, 1/3, 2/3 or 3/4 |
| grades 7-9 | `x > c` with negatives, linear inequalities such as `2x + 3 > 11`, `|x| < k`, perfect squares, powers of 2 or 3 |

Each pack keeps its own top five, in `highscores-hunt-5-6.json` and
`highscores-hunt-7-9.json`.

### The Last Light of Lantern Rock (story)

A story in five chapters, told between the games and spoken aloud, in
**English** or **Ukrainian** - picking Story in the chooser asks which. The lamp
of the lighthouse on Lantern Rock has been shattered by a storm, and its light
scattered; the old keeper, Maren, hands her apprentice - you - the shard at
the heart of the lens. That shard is the laser. With it you gather the light
again, hold off the Gloam, the fog that has waited a hundred years for the
lamp to go out, and bring the fishing fleet home, in the company of Pip, a
spark of the lamp that turns out to have opinions.

Each chapter is a different game:

| | | |
| --- | --- | --- |
| **1** | Sparks on the Wind | sparks blow across the sky: catch 20 before 8 are lost |
| **2** | What the Dark Wants | gloamlings creep towards the cage of sparks from every side, in three waves; the big ones split in two. Leave the green glow-moths alone |
| **3** | Ships in the Dark | boats sail for a reef they cannot see. *Hold* the light on a boat until its skipper turns for the channel, then find the next |
| **4** | The Keepers' Stars | a constellation lights up star by star; trace it back in the same order. Three of them, each longer |
| **5** | The Heart of the Storm | the Gloam itself: strike the bright knots as they open, keep its gloamlings off the lamp - and at the end, hold the light on its eye |

A chapter is won or lost rather than timed. The sparks in the top right are
what you have left to lose; run out and the chapter starts again, as often as
it takes. Winning one earns one to three stars, unlocks the next and is saved
in `story.json`, so the story can be put down and picked up later - **G**
during the story goes back to the list of chapters, and any chapter already
reached can be played again for a better result.

Everything can be done with the pointer alone:

* on the chapter list, hold the laser on a chapter (or press its number);
* a scene moves on by itself as each line ends. To read faster, flash the
  laser on the text box (or press **SPACE**); to skip the scene, hold the
  laser on *skip* in the corner (or press **ENTER**);
* after a chapter, hold on the button, press **SPACE**, or just wait.

A scene does not need the camera, so it carries on if the camera stutters;
only a chapter pauses itself then, as the other games do. **P** pauses
anywhere, and a line that was being spoken goes on from where it stopped (on
Windows it is said again from its beginning).

#### Voices

The lines are spoken - a narrator, Maren, Pip and the Gloam (Морок) - from
recordings in `laserapp/assets/story/en/` and `uk/` (six to seven minutes
of speech and 16 to 21 MB of WAV per language). They were made with text-to-speech models that
run locally, and whose licences allow the recordings to be passed on with
the app:

* English: [Kokoro](https://github.com/hexgrad/kokoro) (Apache-2.0)
* Ukrainian: [StyleTTS2 for Ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian)
  by patriotyk (MIT), which also works out where the stress falls in a word

Playback needs no extra Python package: it goes through PipeWire, PulseAudio
or ALSA on Linux, and `winsound` on Windows. `--no-sound` keeps it quiet, and
a machine with no sound simply shows the text, timed for reading.

#### Music

Each chapter has its own piece of music, which plays while the chapter is
being played and stops with it - the scenes between are left to the voices.
**P** holds it with everything else.

| | | |
| --- | --- | --- |
| **1** | Sparks on the Wind | light and hopeful: bells over a slow major round |
| **2** | What the Dark Wants | a heartbeat and a creeping bass, closing in |
| **3** | Ships in the Dark | a slow rocking in six-eight, with the sea under it |
| **4** | The Keepers' Stars | hardly there: single bells over a held chord, so as not to get in the way of remembering |
| **5** | The Heart of the Storm | drums, thunder, and a tune that fights back |

A chapter is not timed, so each piece is a loop of about forty seconds. They
are synthesised by `tools/story_music.py` (numpy only, run with the
project's Python) into `laserapp/assets/story/music/c1.wav` to `c5.wav`; a
chapter whose file is missing simply plays without. On Windows the music is
played through MCI, beside `winsound`, so that it does not cut the effects
off; if that does not work on a machine, there is everything but the music.

The words are kept apart from the code, one file per language:
`laserapp/story_words_en.py` and `story_words_uk.py` hold every spoken line
and every label on screen. After changing a line, record it again:

    ./build_story_audio.sh            # whatever changed, both languages
    ./build_story_audio.sh uk         # one language
    ./build_story_audio.sh uk --only c1i03,c2f

The first run sets the models up under `.build-cache/` (English about 0.5 GB,
Ukrainian about 2.5 GB). A line whose text no longer matches its recording is
shown without sound rather than with the wrong one, and the self-test says
which lines those are.

Where the Ukrainian model stresses a word wrongly, the word goes into `STRESS`
in `story_words_uk.py` with a `+` after the stressed vowel (`"років":
"рокі+в"`); `tools/story_audio.py --lang uk --stress`, run with the Ukrainian
venv's Python, prints every line as it will be stressed, without recording.

Ukrainian needs OpenCV 5, whose text drawing has Cyrillic letters in it; with
an older OpenCV the chooser offers the story in English only. To add a
language, copy a words file, add it to `LANGUAGES` in `story_script.py` and
give it a voice in `tools/story_audio.py`.

The pictures are drawn by the program (`story_art.py`), in the same palette
as the rest of the app and for the same reason - see
[A note on the colours](#a-note-on-the-colours). The self-test plays the
whole story and checks that no pixel of it has more red than green.

### Range Duel (grades 8-9, two players)

Two players, a laser each, and the screen split down the middle: **player 1
has the left half, player 2 the right**. Each half is a lane of a pop-up
range - figures come up from behind two walls, stay for a moment (the bar on
the wall is the time left) and go down again. Some are armed, some are not:

| | |
| --- | --- |
| **Shoot** | a masked man with a pistol, a masked man with a rifle, a man in a cap and dark glasses with a knife raised |
| **Spare** | a woman with a handbag, a man reading his phone, a man with his hands up, a medic with a bag |
| **Both** | a gunman behind a woman he is holding: only what shows of him counts, and a shot that lands on her is a shot at her |

They are told apart as on a real shoot / no-shoot range: by what is in the
hands - the one bright thing on each figure - and by the face, which on the
armed is masked or scowling.

Both lanes are given **the same figures at the same moments**, so the only
difference between the two scores is the two players. A round is 60 seconds;
figures come quicker and leave sooner as it goes on.

* An armed figure down is worth 60 to 100 points - more the sooner it is
  shot - about half as much again on the far wall, and double for the gunman
  with a hostage. Every hit in a row adds 0.1 to the multiplier, up to **x2**.
* Shooting someone unarmed costs **200** and the streak, so shooting at
  everything loses. A score can go below zero.
* A flash that hits nothing, or an armed figure left standing, costs the
  streak and nothing else.

Flash and dwell both shoot, as in the [gallery](#shooting). Both scores stay
at the top of the screen, the leader's underlined; at the end the round goes
to the higher one, with each player's hits, mistakes, accuracy and reaction
time. Holding either laser on the **REMATCH** button for a second starts the
next round, as **G** does, and the rounds won are counted until the duel is
left with **ESC**.

**Two lasers, one camera.** Nothing tells one laser from the other except
where it points: a dot counts for the half it is in, the strongest dot in
each half. So a player who points into the other half is shooting for the
opponent - at the opponent's civilians too - and keeping to one's own lane
is a rule for the class, not something the program can enforce. With the
mouse (**M**) there is one pointer, which plays for whichever half it is in.

**Music and sound.** A piece of music runs the length of the round and
builds as the clock runs down - a tune comes in at half time, and the last
twelve seconds are a tone higher - with a ring for a hit, a buzz for someone
unarmed and a fanfare at the end. All of it is synthesised by
`tools/duel_audio.py` (numpy only) into `laserapp/assets/duel/`; edit the
script and run it with the project's Python to change it. **P** pauses the
music with the round. `--no-sound` keeps it quiet.

The figures are drawn by the program (`duel_art.py`) rather than loaded from
photographs, for the reason given in
[A note on the colours](#a-note-on-the-colours): a photograph of a person is
full of red, and the camera would take it for the laser.

## Shooting game

The gallery: targets appear, you shoot them with the laser, the round lasts a
minute.

### Shooting

Two triggers, both read from the same stream of detections:

* **Flash** - the beam turns on while pointing at a target. Aim with the laser
  off, flash it on the target: an instant hit, and the hit is marked `SNAP`.
  The beam has to have been off for 150 ms, so a wobble does not count as a
  shot.
* **Dwell** - hold the beam on a target for 0.35 s. A white ring fills to show
  it. This exists because plenty of pointers are simply on all the time, and
  because once the beam is lit there is nothing to flash.

Turning the beam on while pointing at nothing counts as a missed shot: it
costs no points but breaks your streak and drags your accuracy down. Sweeping
a lit beam across empty screen costs nothing - that is just aiming.

### Targets and scoring

| | |
| --- | --- |
| **Cyan** | normal. Smaller and faster to shoot is worth more |
| **Lime yellow, marked `x3`** | bonus - small, quick to leave, triple points |
| **Blue, marked with an X** | decoy. Shooting it costs 25 points, your streak and a life |

Every hit in a row raises the multiplier by 0.1, up to **x2**. Shooting
quickly after a target appears is worth more than shooting it just before it
expires. You have **three lives**: one goes for every normal target that gets
away and for every decoy you shoot.

The round gets harder as it runs: targets spawn faster, get smaller, live for
less time and start moving, and decoys only appear after the first 15 seconds.
The level shown next to the clock is how far the ramp has gone.

At the end you get score, hits, missed shots, targets that got away, accuracy,
best streak and average reaction time, plus the top five scores, kept in
`highscores.json`.

**P** pauses (the clock stops properly - no free time while paused), **ESC**
goes back to tracking.

### Silhouette range

The second game under Shooting Gallery. A man-shaped target stands on the
target line with scoring rings numbered across its body: **10** in the middle
of the chest, then 9, 8, 7 and 6, and **5** for the rest of the figure, head
included. Here it matters where the shot lands, not just that it hits.

A series is ten figures, one at a time, **one shot each**, 100 points at best.
Each figure stands further away - smaller - than the one before and stays up
for less time (6 s down to 4 s; the bar on the ground under it is the time
left). The last few walk along the line. After the shot the figure stays up
for a moment with the hit marked and its score above the head.

* **Flash** shoots at the spot where the beam appears. Flashing beside the
  figure is a miss: it scores 0 and that figure is used up.
* **Dwell** is an aimed shot: the beam has to rest in one place on the figure
  for half a second. Sweeping a lit beam across it does not fire, so you can
  walk the beam to the middle and then hold.

A figure nobody shoots in time scores 0. The top five series are kept in
`highscores-silhouette.json`.

### Without a laser

**M** switches the pointer to the mouse: holding the left button stands in for
the beam being lit, so flash and dwell work exactly the same. Useful for
trying the game out, or for checking the app on a machine with no laser to
hand. `--mouse-pointer` starts in that mode.

### A note on the colours

Targets are cyan, lime yellow and blue, and never red. The camera is pointed
at this screen, and anything it sees as red reads as a second laser dot.

What matters is what the camera sees, not what the screen sends. Measured
through a webcam, small text in plain yellow or in magenta was taken for the
laser in every single frame, and white and grey text came out faintly red too.
So nothing on screen is plain yellow or magenta, and the white and greys are
held slightly cool. If you add graphics of your own, keep red well below
green: `(R, G, B) = (190, 255, 60)` is safe, `(255, 240, 60)` is not.

## Keys

| Key | Action |
| --- | --- |
| `G` | games (chooser; during a game, play it again) |
| `P` | pause (during the game) |
| `SPACE` / `ENTER` | next line / skip the scene (in the story) |
| `M` | use the mouse as the pointer |
| `K` | calibrate automatically (no laser) |
| `F` | auto-tune thresholds to the current scene |
| `D` | camera preview: wide → with mask → off |
| `V` | choose camera (live preview of each) |
| `N` | next camera |
| `U` | preview: lens-corrected or raw |
| `C` | calibrate with the laser |
| `SPACE` / `B` / `ESC` | capture now / back one point / cancel (during calibration) |
| `T` / `X` | trail on/off, crosshair on/off |
| `[` `]` | brightness threshold down/up |
| `,` `.` | redness threshold down/up |
| `E` / `R` | exposure darker/brighter |
| `A` | auto-exposure on/off |
| `S` | save calibration + detector settings |
| `H` | help overlay |
| `Q` | quit |

## Tuning the detector

Press **D** to see what the detector sees. The bottom half is the mask - you
want a single small white blob on the laser and nothing else.

The detector works on **local contrast in redness**, not absolute brightness.
A top-hat filter subtracts whatever the neighbourhood is doing, so anything
larger than the filter window disappears before anything is measured:

* **Sunlight on the screen** lifts a whole region - removed by the top-hat,
  and white glare has no redness to begin with.
* **The "waves"** a camera picks up pointing at a display (rolling-shutter
  banding, the sensor and the panel refreshing at different rates) are grey
  and arrive as long streaks. They lose their redness and then fail the shape
  test: a laser spot is compact and fills its bounding box, a band does not.
* **Red things in the room** - a wooden shelf, a standby LED, someone's
  jumper - are excluded outright once calibrated, because the screen outline
  in camera pixels is known and nothing outside it can be a laser on the
  screen. Measured here: without it, a red object in the room outscored the
  real spot in *every* frame; with it, zero false detections in 30 frames.

What is left is thresholded against the frame's own noise, so it re-tunes
itself as the light changes.

If the spot still is not found:

* **Press F** with the laser pointed away. Whatever the detector can see right
  now stops counting - this is the answer when the screen itself is showing
  something red.
* **`[` and `]`** change sensitivity, **`,` and `.`** the redness floor. The
  HUD shows `sens`, `floor` and the resulting threshold.
* **Overexposure is the one thing no threshold can fix.** A clipped sensor
  records white, and white has no redness. If more than 2% of the image is
  blown out and nothing is detected, the app says so - press **E** to darken
  the exposure until the screen is dim and the spot stands out.

## Options

    --camera SPEC       index (2), name fragment (genius), pick (show the
                        chooser) or auto (prefer a USB camera over the built-in
                        one). Default: last camera chosen, else the chooser
                        when several are plugged in
    --list-cameras      list capture devices and exit
    --width/--height    capture resolution (default 1280x720)
    --exposure V        start with manual exposure V (Linux: V4L2 units, try
                        50-200; Windows: log2 seconds, try -5 to -9)
    --screen W H        override detected screen resolution
    --windowed          run in a window instead of fullscreen
    --grid COLS ROWS    calibration grid, default 4 4 (3 3 is quicker, but
                        leaves less data for the lens fit)
    --no-lens           skip lens fitting, plain homography only
    --no-auto-calibrate do not calibrate automatically at startup
    --smoothing 0..1    reticle responsiveness (default 0.45, higher = snappier)
    --hold SECONDS      how long the reticle lingers after the laser vanishes
    --calibration PATH  calibration file (default ./calibration.json)
    --mouse             also drive the real mouse cursor (needs `pyautogui`)
    --no-preview        start without the camera preview
    --mouse-pointer     use the mouse as the pointer instead of the laser
    --no-sound          no voices, music or sound effects
    --no-trail          start with the trail off

## Wide-angle cameras

A 120-degree webcam bends straight lines noticeably, and no homography can
undo that. The calibration therefore also fits a radial distortion model
(OpenCV's `k1`/`k2`) - not from a printed chessboard, but from the same laser
samples: they are known to sit on a regular grid on a plane, so *the amount of
bowing that makes them fit a single homography best is the lens distortion*.
Points are straightened before the homography is applied, both when solving
and when tracking.

Measured on the Genius WideCam F100 V2 here, against a 1920x1080 screen:

| | mean error | worst |
| --- | --- | --- |
| homography alone | 10.8 px | ~25 px |
| with lens correction (`k1=-0.247`, `k2=+0.086`) | 1.5 px | 5.2 px |

The HUD shows `lens k1=...` once a lens has been fitted, and the preview is
straightened to match - press **U** to compare it against the raw camera image
(the raw view is also the wider one; straightening crops the edges).

Correction is only kept if it beats the plain homography by a clear margin, so
a rectilinear camera does not get a distortion invented for it. `--no-lens`
turns the fit off entirely.

A fit needs more points than a homography does, hence the 4x4 default grid.
16 points is enough: the same lens fitted from 16 points and from a 54-point
chessboard agreed to three decimal places here.

### If a capture goes wrong

If the laser slips off a marker, that one point is discarded rather than
smeared across the whole fit: the cut-off is a multiple of the *median*
residual, because a fixed pixel threshold cannot tell a slipped capture from
the large, legitimate residuals an uncorrected lens produces. Discarded points
are reported in the calibration message.

## A camera that sends damaged frames

USB webcams send each frame as a JPEG, and some of them - the Genius WideCam
among them - now and then send frames that are cut short, or get out of step
so that the end of every frame arrives glued to the start of the next. That
can last from a moment to a minute. OpenCV answers such a frame with the
previous picture and says nothing, so the app would be looking for the laser
in a picture that is seconds old: the reticle freezes and games stop reacting.

On Linux the app therefore takes the frames compressed and checks them itself.
A damaged frame is thrown away, frames that are merely out of step are put
back together, and if nothing usable arrives for 1.5 s the camera is reopened.
While there is no fresh picture there is no pointer, and a game in progress
pauses itself until the picture is back. When more than 5% of recent frames
are being lost, a yellow line above the status bar says so - at that point
look at the cable, try another USB port, or plug the camera straight into the
computer rather than a hub.

## Self-test

No camera or display required - renders synthetic frames, checks detection
accuracy, then solves and inverts a known homography:

    .venv/bin/python selftest.py

## Layout

    laserapp/camera.py       camera listing (V4L2 / DirectShow), threaded capture, exposure
    build_windows.sh         portable Windows bundle -> dist/
    laserapp/detector.py     red-dot detection (redness x brightness, sub-pixel centroid)
    laserapp/game.py         round engine (flash/dwell shots, pause), the shooting game, high scores
    laserapp/silhouette.py   silhouette range: the figure, its rings, ten aimed shots
    laserapp/mathgames.py    Balloon Math and Number Hunt
    laserapp/story.py        the story game: chapter list, scenes, progress
    laserapp/story_levels.py its five chapters
    laserapp/story_script.py its script: who speaks when, over which picture
    laserapp/story_words_*.py its words, one file per language
    laserapp/story_art.py    its pictures
    laserapp/duel.py         range duel: two lanes, two lasers, one score each
    laserapp/duel_art.py     its figures, armed and unarmed
    laserapp/assets/duel/    its music and sound effects (tools/duel_audio.py)
    laserapp/sound.py        WAV playback, no dependencies
    laserapp/assets/story/   the recorded voices, the sound effects, the music
    tools/story_music.py     makes the music again
    build_story_audio.sh     records the voices again (tools/story_audio.py)
    laserapp/menu.py         game chooser
    laserapp/autocal.py      automatic calibration: lit dots, differenced
    laserapp/lens.py         radial distortion model and its fit
    laserapp/calibration.py  homography solve/save/load + calibration state machine
    laserapp/overlay.py      reticle, markers, HUD, debug panel
    laserapp/app.py          main loop, modes, keybindings
    laserapp/screen.py       desktop resolution detection
