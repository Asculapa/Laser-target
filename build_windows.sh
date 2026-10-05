#!/usr/bin/env bash
# Build a portable Windows (x64) bundle from Linux - no Wine, no installer.
#
# It pairs the official embeddable Python with Windows wheels of the
# dependencies, so the result runs on a Windows machine with nothing
# installed: unzip, double-click "Laser Target.bat".
#
#   ./build_windows.sh            -> dist/LaserTarget-windows-x64.zip
set -euo pipefail
cd "$(dirname "$0")"

PYVER=3.12.10                      # last 3.12 release with an embeddable zip
PYTAG=312
OUT=dist/LaserTarget
CACHE=.build-cache
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

rm -rf "$OUT" "$OUT"-windows-x64.zip
mkdir -p "$OUT/python" "$CACHE"

embed="$CACHE/python-$PYVER-embed-amd64.zip"
[ -f "$embed" ] || curl -fsSL -o "$embed" \
    "https://www.python.org/ftp/python/$PYVER/python-$PYVER-embed-amd64.zip"
unzip -q "$embed" -d "$OUT/python"

# Windows wheels, whatever platform this runs on. pygrabber gives the
# camera chooser real device names; comtypes is what it talks DirectShow with.
# sounddevice (with PortAudio inside its wheel) mixes the effects, so that
# shots overlap instead of cutting each other off in winsound.
wheels="$CACHE/wheels-$PYTAG"
"$PY" -m pip download -q --only-binary=:all: --platform win_amd64 \
    --python-version "$PYTAG" --implementation cp -d "$wheels" \
    "opencv-python>=5.0" "numpy>=1.24" pygrabber comtypes sounddevice
site="$OUT/python/Lib/site-packages"
mkdir -p "$site"
for w in "$wheels"/*.whl; do unzip -q -o "$w" -d "$site"; done

# The embeddable build ignores PYTHONPATH and site-packages unless its ._pth
# file names them; add both, plus the app folder one level up.
cat > "$OUT/python/python$PYTAG._pth" <<EOF
python$PYTAG.zip
.
Lib\\site-packages
..
import site
EOF

cp -r laserapp "$OUT/"
find "$OUT/laserapp" -name __pycache__ -prune -exec rm -rf {} +
cp README.md "$OUT/"

# CRLF: cmd.exe misparses some LF-only batch files.
printf '%s\r\n' \
    '@echo off' \
    'rem Laser Target - calibration and settings are saved in this folder.' \
    'cd /d "%~dp0"' \
    'python\python.exe -m laserapp %*' \
    'if errorlevel 1 pause' \
    > "$OUT/Laser Target.bat"
printf '%s\r\n' \
    '@echo off' \
    'cd /d "%~dp0"' \
    'python\python.exe -m laserapp --list-cameras' \
    'pause' \
    > "$OUT/List cameras.bat"

(cd dist && zip -qr LaserTarget-windows-x64.zip LaserTarget)
echo "built dist/LaserTarget-windows-x64.zip ($(du -h dist/LaserTarget-windows-x64.zip | cut -f1))"
