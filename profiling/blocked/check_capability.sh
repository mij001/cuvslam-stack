#!/usr/bin/env bash
# check_capability.sh — gate for the Slice-3 DAMOV data-movement track.
#
# The NVBit → locality → Accel-Sim pipeline only runs where the toolchain can
# actually load. This check fails FAST with the exact reason and the unblock
# instructions, so the gated scripts never half-run and leave misleading output.
#
# Unblock conditions (either):
#   * CUDA driver <= 575.xx (NVBit release cap as of nvbit 1.7.7), or
#   * a newer NVBit release that supports this driver (update NVBIT_MAX_DRIVER).
set -u
NVBIT_MAX_DRIVER=575          # bump when a newer NVBit release raises the cap
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
NVBIT_DIR="${NVBIT_DIR:-$REPO_ROOT/external_repos/nvbit_release_x86_64}"

fail=0
say() { printf '%s\n' "$*"; }

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
major=${drv%%.*}
if [ -z "${drv:-}" ]; then
    say "BLOCKED: no NVIDIA driver found"
    fail=1
elif [ "${major:-999}" -gt "$NVBIT_MAX_DRIVER" ]; then
    say "BLOCKED: driver $drv > NVBit cap ($NVBIT_MAX_DRIVER.xx)."
    say "  unblock: run on a host with driver <= $NVBIT_MAX_DRIVER (e.g. the RTX 2000 Ada"
    say "  workstation if it runs the production-branch driver), OR download a newer"
    say "  NVBit from https://github.com/NVlabs/NVBit/releases and update"
    say "  NVBIT_MAX_DRIVER in $0."
    fail=1
else
    say "OK: driver $drv <= $NVBIT_MAX_DRIVER"
fi

if [ -d "$NVBIT_DIR" ] && [ -f "$NVBIT_DIR/core/nvbit.h" ]; then
    say "OK: NVBit release at $NVBIT_DIR"
else
    say "BLOCKED: NVBit not found at $NVBIT_DIR (set NVBIT_DIR or place the release there)"
    fail=1
fi

if [ -d "$REPO_ROOT/external_repos/DAMOV-main/simulator/src" ]; then
    say "OK: DAMOV sources (locality.cpp) at external_repos/DAMOV-main"
else
    say "WARN: DAMOV sources missing — clone https://github.com/CMU-SAFARI/DAMOV to external_repos/DAMOV-main"
fi

if [ $fail -ne 0 ]; then
    say "-- Slice-3 data-movement track is GATED on this host; the NCU/Nsys spine is unaffected."
fi
exit $fail
