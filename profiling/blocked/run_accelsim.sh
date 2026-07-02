#!/usr/bin/env bash
# run_accelsim.sh — Slice-3: steady-state cache simulation via Accel-Sim.
#
# GATED: needs NVBit traces (check_capability.sh) AND an Accel-Sim checkout.
#
#   profiling/blocked/run_accelsim.sh <traces-dir>
#
# Methodology (PROFILING_PLAN §6 Slice 3):
#   * base config from the hw descriptor's [accelsim].base_config
#     (sm_75 = stock SM75_RTX2060; sm_89 must be ADAPTED from Turing and
#     calibrated against NCU microbenchmarks to ~5% before use)
#   * report DELTAS between configurations, never absolute simulated numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

"$HERE/check_capability.sh" || exit 1

ACCELSIM_DIR="${ACCELSIM_DIR:-$REPO_ROOT/external_repos/accel-sim-framework}"
if [ ! -d "$ACCELSIM_DIR" ]; then
    cat <<EOF
BLOCKED: Accel-Sim not found at $ACCELSIM_DIR
  unblock:
    git clone https://github.com/accel-sim/accel-sim-framework "$ACCELSIM_DIR"
    cd "$ACCELSIM_DIR" && . environment_setup/setup_environment.sh && make -j
  then re-run. (Also set the base config per profiling/hw/<gpu>.toml [accelsim].)
EOF
    exit 1
fi

TRACES="${1:?usage: $0 <traces-dir produced by accel-sim tracer>}"
echo "[accel-sim] simulating traces in $TRACES"
echo "REMINDER: report deltas, not absolutes (see PROFILING_PLAN.md §6)."
"$ACCELSIM_DIR/gpu-simulator/bin/release/accel-sim.out" -trace "$TRACES" \
    -config "${GPGPUSIM_CONFIG:?set GPGPUSIM_CONFIG to the calibrated gpgpusim.config}"
