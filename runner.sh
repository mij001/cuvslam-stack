#!/usr/bin/env bash
#
# Convenience launcher for cuvslam_runner.
#
# Picks a Python interpreter that has the `cuvslam` module, sets PYTHONPATH so
# the cuvslam_runner package is importable, and dispatches to the right entry
# point — so you don't have to manage the venv or PYTHONPATH by hand.
#
# Usage:
#   ./runner.sh <config.toml> [extra args]    run one config        (-> run.py)
#   ./runner.sh check <config.toml>           validate one config   (-> run.py --check)
#   ./runner.sh all [args...]                 run every config      (-> run_all.py)
#   ./runner.sh eval <est> <gt> [args...]     evaluate a trajectory (-> evaluate.py)
#   ./runner.sh python [args...]              the chosen interpreter (debug)
#
# Interpreter selection (first that can `import cuvslam` wins; otherwise the
# first that simply exists, so `check` still works without the wheel):
#   $CUVSLAM_PYTHON                       explicit override
#   $HERE/.venv/bin/python                local virtualenv
#   $HERE/../../wheel/cuvslam_env/...     sibling wheel env (this repo layout)
#   python3 / python                      whatever is on PATH
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_can_import_cuvslam() { "$1" -c 'import cuvslam' >/dev/null 2>&1; }

pick_python() {
    local candidates=()
    [ -n "${CUVSLAM_PYTHON:-}" ] && candidates+=("$CUVSLAM_PYTHON")
    candidates+=(
        "$HERE/.venv/bin/python"
        "$HERE/../../wheel/cuvslam_env/bin/python"
    )
    command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
    command -v python  >/dev/null 2>&1 && candidates+=("$(command -v python)")

    local fallback=""
    for c in "${candidates[@]}"; do
        [ -x "$c" ] || continue
        [ -z "$fallback" ] && fallback="$c"
        if _can_import_cuvslam "$c"; then
            echo "$c"
            return 0
        fi
    done
    if [ -n "$fallback" ]; then
        echo "$fallback"
        return 0
    fi
    echo "runner.sh: no usable Python interpreter found" >&2
    return 1
}

usage() { sed -n '3,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

[ $# -eq 0 ] && { usage; exit 1; }

PY="$(pick_python)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

if ! _can_import_cuvslam "$PY"; then
    echo "runner.sh: note: '$PY' cannot import cuvslam — only 'check' will work." >&2
    echo "           set CUVSLAM_PYTHON=/path/to/python with the wheel installed." >&2
fi

# Echo the exact command, then replace this shell with it.
runexec() { echo "+ $*" >&2; exec "$@"; }

cmd="$1"; shift || true
case "$cmd" in
    all)    runexec "$PY" "$HERE/run_all.py"  "$@" ;;
    eval)   runexec "$PY" "$HERE/evaluate.py" "$@" ;;
    check)  runexec "$PY" "$HERE/run.py"      "$@" --check ;;
    python) runexec "$PY" "$@" ;;
    -h|--help|help) usage ;;
    *)      runexec "$PY" "$HERE/run.py" "$cmd" "$@" ;;
esac
