#!/usr/bin/env bash
#
# Remove the cuvslam_runner virtualenv created by setup_env.sh, and (optionally)
# the generated outputs and __pycache__.
#
# Usage:
#   ./cleanup_env.sh                # remove the venv + __pycache__
#   ./cleanup_env.sh --outputs      # also remove the out/ directory
#
# Environment overrides:
#   VENV=/path     venv location (default: ./cuvslam_venv)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/cuvslam_venv}"

CLEAN_OUTPUTS=0
for arg in "$@"; do
    case "$arg" in
        --outputs) CLEAN_OUTPUTS=1 ;;
        -h|--help) sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "cleanup_env.sh: unknown arg '$arg'" >&2; exit 2 ;;
    esac
done

# Echo each command before running it (so every executed command is visible).
run() { echo "+ $*"; "$@"; }

if [ -n "${VIRTUAL_ENV:-}" ] && [ "${VIRTUAL_ENV}" = "$VENV" ]; then
    echo "[cleanup] note: this venv looks active — run 'deactivate' in your shell afterwards."
fi

if [ -d "$VENV" ]; then
    run rm -rf "$VENV"
    echo "[cleanup] removed $VENV"
else
    echo "[cleanup] no venv at $VENV (nothing to remove)"
fi

if [ "$CLEAN_OUTPUTS" = "1" ] && [ -d "$HERE/out" ]; then
    run rm -rf "$HERE/out"
    echo "[cleanup] removed $HERE/out"
fi

echo "+ find $HERE -name __pycache__ -type d -prune -exec rm -rf {} +"
find "$HERE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "[cleanup] done"
