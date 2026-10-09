"""Fetch the Wild West game's pictures from OpenGameArt.org.

    .venv/bin/python tools/west_assets.py

Every file comes from OpenGameArt.org under CC0 (public domain), so it can
be shipped with the app; CREDITS.md, written beside them, says whose each
one is anyway. They are kept as they were published - redder and brighter
than anything this app may show - and recoloured when the game loads them
(west_art.py). The backdrops are photographs of real Old West towns, taken
out of the SVG files they were published in and made smaller.
"""
from __future__ import annotations

import base64
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "laserapp" / "assets" / "west"
FILES = "https://opengameart.org/sites/default/files/"

# name in the app: (SVG on OpenGameArt, where the photo was taken)
BACKDROPS = {
    "stelmo": ("backdrop-4.svg", "St. Elmo, Colorado"),
    "store": ("backdrop-2.svg", "a row of old shops and a blacksmith's"),
    "inn": ("backdrop-9.svg", "the Fairweather Inn, Virginia City, Montana"),
    "boardwalk": ("backdrop-1.svg", "a Western street with a boardwalk"),
    "hotel": ("backdrop-3.svg", "Hank's Hotel, Calico, California"),
    "calico": ("backdrop-8.svg", "Calico Ghost Town, California"),
}
# The townsfolk: name in the app, and the folder of its idle frames in the
# Gothicvania Town pack.
TOWNSFOLK = {"oldman": "oldman-idle", "gentleman": "hat-man-idle",
             "butcher": "bearded-idle", "lady": "woman-idle"}
TOWN_ZIP = "gothicvania-town-files.zip"
# The bandits - name in the app: (zip, prefix of its frames)
COWBOYS = {
    "cowboy2": ("Cowboy%202%20HiRes_0.zip", "Cowboy2_"),
    "cowboy4": ("Cowboy%204%20HiRes_0.zip", "Cowboy4_"),
}
POSES = {"idle with gun": "gun", "shoot": "shoot", "idle without gun": "idle",
         "walk with gun": "walk"}

CREDITS = """\
# Wild West: where the pictures come from

All from [OpenGameArt.org](https://opengameart.org), all **CC0** (public
domain). Fetched by `tools/west_assets.py`; recoloured by the game at load
time so that nothing on screen is red (see `laserapp/west_art.py`).

| Files | Work | Author |
| --- | --- | --- |
| `stelmo.jpg`, `store.jpg`, `inn.jpg`, `boardwalk.jpg`, `hotel.jpg`, `calico.jpg` | [Old West Backdrops](https://opengameart.org/content/old-west-backdrops) - photographs from Wikimedia Commons (CC0): St. Elmo, a row of shops, the Fairweather Inn, a boardwalk street, Hank's Hotel and the main street of Calico | Technopeasant |
| `cowboy2/` | [Gangster](https://opengameart.org/content/gangster) ("Cowboy 2 HiRes") | software_atelier |
| `cowboy4/` | [Cowboy](https://opengameart.org/content/cowboy) ("Cowboy 4 HiRes") | software_atelier |
| `townsfolk/` | [Gothicvania Town](https://opengameart.org/content/gothicvania-town) - the four townsfolk only (the pack's music, which is not CC0, is not used) | Luis Zuno (ansimuz) |
"""


def fetch(name: str) -> bytes:
    req = urllib.request.Request(FILES + name, headers={"User-Agent": "laser-app"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def backdrop(svg: bytes) -> np.ndarray:
    """The photograph inside one of the backdrop SVGs."""
    m = re.search(rb"data:image/(?:png|jpe?g);base64,([A-Za-z0-9+/=\s]+)", svg)
    data = base64.b64decode(re.sub(rb"\s", b"", m.group(1)))
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img.shape[1] > 1400:
        img = cv2.resize(img, (1400, int(img.shape[0] * 1400 / img.shape[1])),
                         interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (svg, _) in BACKDROPS.items():
        img = backdrop(fetch(svg))
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{name}.jpg  {img.shape[1]}x{img.shape[0]}")
    folk = OUT / "townsfolk"
    folk.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(fetch(TOWN_ZIP))) as z:
        for entry in z.namelist():
            parts = Path(entry).parts
            for name, folder in TOWNSFOLK.items():
                m = re.fullmatch(re.escape(folder) + r"-(\d+)\.png", parts[-1])
                if m and "sprites" in parts and folder in parts:
                    (folk / f"{name}_{int(m.group(1)) - 1}.png").write_bytes(z.read(entry))
    print(f"townsfolk: {len(list(folk.glob('*.png')))} frames")
    for name, (zipname, prefix) in COWBOYS.items():
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(fetch(zipname))) as z:
            for entry in z.namelist():
                stem = Path(entry).name
                for pose, short in POSES.items():
                    m = re.fullmatch(re.escape(prefix + pose) + r"_(\d)\.png", stem)
                    if m:
                        (folder / f"{short}_{m.group(1)}.png").write_bytes(z.read(entry))
        print(f"{name}: {len(list(folder.glob('*.png')))} frames")
    (OUT / "CREDITS.md").write_text(CREDITS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
