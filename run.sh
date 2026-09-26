#!/usr/bin/env bash
# Convenience launcher: uses the project venv if present.
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
exec "$PY" -m laserapp "$@"
