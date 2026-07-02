#!/usr/bin/env bash
# check_env.sh — headless preflight for the profiling harness on ANY machine.
# Prints a PASS/FAIL line per requirement; exit code 0 only if the required
# pieces are in place. Run this first on a new host (laptop, workstation, CI).
#
#   profiling/env/check_env.sh [--venv PYTHON]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV_PY="${2:-$REPO_ROOT/cuvslam_venv/bin/python}"
[ "${1:-}" = "--venv" ] && VENV_PY="$2"

fail=0
pass() { printf ' PASS  %s\n' "$1"; }
warn() { printf ' WARN  %s\n' "$1"; }
die()  { printf ' FAIL  %s\n' "$1"; fail=1; }

echo "== cuvslam-stack profiling preflight =="

# GPU + driver
if command -v nvidia-smi >/dev/null; then
    gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
    pass "GPU: $gpu"
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "${drv:-0}" -le 575 ]; then
        pass "driver <= 575: NVBit (Slice-3 data-movement track) can run here"
    else
        warn "driver > 575: NVBit blocked; NCU/Nsys track only (see blocked/README)"
    fi
else
    die "nvidia-smi not found (no NVIDIA driver?)"
fi

# profilers
for tool in nsys ncu; do
    if command -v $tool >/dev/null; then
        pass "$tool: $($tool --version 2>&1 | grep -m1 -o 'Version [0-9.]*\|version [0-9.]*' | head -1)"
    else
        die "$tool not on PATH (install nsight-systems / nsight-compute)"
    fi
done

# counter permissions (ncu)
if [ -r /proc/driver/nvidia/params ] && grep -q "RmProfilingAdminOnly: 0" /proc/driver/nvidia/params; then
    pass "GPU counters unrestricted (RmProfilingAdminOnly=0)"
else
    warn "GPU counters admin-only — run: sudo profiling/env/setup_perms.sh (then reboot)"
fi

# workload venv
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import cuvslam" 2>/dev/null; then
    pass "workload venv imports cuvslam ($("$VENV_PY" -c 'import cuvslam;print(getattr(cuvslam,"__version__","?"))'))"
else
    die "no python with cuvslam at $VENV_PY — run ./setup_env.sh (needs the wheel; see README 'make wheel')"
fi

# analysis layer needs only python3 stdlib
if command -v python3 >/dev/null; then
    pass "python3 for the analysis layer: $(python3 --version)"
else
    die "python3 not found"
fi

# datasets root
DATASETS="${CUVSLAM_DATASETS:-$HOME/Projects/cuvslam_datasets}"
if [ -d "$DATASETS" ]; then
    pass "dataset root: $DATASETS ($(ls "$DATASETS" | tr '\n' ' '))"
else
    warn "dataset root $DATASETS missing — profiling/env/fetch_datasets.sh, or export CUVSLAM_DATASETS"
fi

# disk space (nsys reports can reach GBs)
avail_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc 0-9)
if [ "${avail_gb:-0}" -ge 10 ]; then
    pass "disk: ${avail_gb} GB free"
else
    warn "disk: only ${avail_gb} GB free — nsys/ncu reports may not fit; clean profiling/results/"
fi

# clock locking policy is per-hw-descriptor; just surface the state
if nvidia-smi -q -d CLOCK 2>/dev/null | grep -q "Applications Clocks"; then
    pass "clock query OK (lock policy comes from the --hw descriptor via env/lock_clocks.sh)"
fi

echo "== preflight $( [ $fail -eq 0 ] && echo OK || echo FAILED ) =="
exit $fail
