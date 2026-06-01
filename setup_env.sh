#!/usr/bin/env bash
#
# Create the cuvslam_runner virtualenv and install dependencies + the cuVSLAM
# wheel. The venv PERSISTS after this script (use cleanup_env.sh to remove it).
#
# Usage:
#   ./setup_env.sh
#
# Environment overrides:
#   PYBIN=python3.10     interpreter used to create the venv (must match the wheel)
#   VENV=/path           venv location (default: ./cuvslam_venv)
#   WHEEL=/path/...whl   explicit wheel (default: newest ../dist/cuvslam-*.whl)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYBIN="${PYBIN:-python3.10}"
VENV="${VENV:-$HERE/cuvslam_venv}"
WHEEL="${WHEEL:-$(ls -t "$HERE"/../dist/cuvslam-*.whl 2>/dev/null | head -1 || true)}"

# Echo each command before running it (so every executed command is visible).
run() { echo "+ $*"; "$@"; }

if [ -d "$VENV" ]; then
    echo "[setup] reusing existing venv at $VENV"
    echo "        (delete it or run ./cleanup_env.sh first to recreate)"
else
    echo "[setup] creating venv with $PYBIN ..."
    run "$PYBIN" -m venv "$VENV"
fi

PY="$VENV/bin/python"
run "$PY" -m pip install --upgrade pip || true

echo "[setup] installing requirements ..."
run "$PY" -m pip install -r requirements.txt

if [ -n "$WHEEL" ] && [ -f "$WHEEL" ]; then
    echo "[setup] installing cuVSLAM wheel: $WHEEL"
    run "$PY" -m pip install "$WHEEL"
else
    echo "[setup] WARNING: no cuVSLAM wheel found in ../dist (set WHEEL=...);" \
         "tracking will be unavailable, only --check validation will work." >&2
fi

if "$PY" -c "import cuvslam; print('[setup] cuvslam', cuvslam.get_version()[0], 'OK')" 2>/dev/null; then
    :
else
    echo "[setup] note: cuvslam is installed but not importable in this venv" >&2
    echo "        (commonly a CUDA runtime mismatch — verify the wheel matches your CUDA)." >&2
fi

echo
echo "[setup] done. venv at: $VENV"
echo "        activate:        source \"$VENV/bin/activate\""
echo "        run a list:      python run_list.py runlist.txt"
echo "        tear down:       ./cleanup_env.sh"
