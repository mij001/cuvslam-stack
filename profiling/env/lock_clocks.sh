#!/usr/bin/env bash
# Lock GPU/CPU clocks for measurement reproducibility, driven by a hardware
# descriptor (profiling/hw/*.toml). On laptop GPUs (enforce_clock_locks = false)
# this is a no-op: clocks can't be pinned, so we just record them per run instead.
#
#   sudo profiling/env/lock_clocks.sh [--hw profiling/hw/<gpu>.toml]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HW="$HERE/../hw/mx450_sm75.toml"
while [[ $# -gt 0 ]]; do case "$1" in --hw) HW="$2"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

val() { grep -E "^\s*$1\s*=" "$HW" 2>/dev/null | head -1 | sed 's/.*=\s*//; s/#.*//; s/[" ]//g'; }
ENFORCE="$(val enforce_clock_locks)"; GFX="$(val graphics_clock_mhz)"; MEM="$(val memory_clock_mhz)"
GOV="$(val cpu_governor)"; GOV="${GOV:-performance}"

if [[ "${ENFORCE,,}" != "true" ]]; then
  echo "[i] $HW has enforce_clock_locks != true — clocks not pinned (expected on laptops)."
  echo "    Current clocks are recorded in each run's metadata.json instead."
  nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,clocks.sm --format=csv 2>/dev/null || true
  exit 0
fi

[[ $EUID -eq 0 ]] || exec sudo bash "$0" --hw "$HW"
echo "[*] persistence on; locking gfx=${GFX} MHz mem=${MEM} MHz; cpu governor=${GOV}"
nvidia-smi -pm 1 >/dev/null
[[ -n "$GFX" ]] && nvidia-smi -lgc "$GFX" >/dev/null || echo "[!] no graphics_clock_mhz in $HW"
[[ -n "$MEM" ]] && nvidia-smi -lmc "$MEM" >/dev/null || echo "[!] no memory_clock_mhz in $HW"
command -v cpupower >/dev/null && cpupower frequency-set -g "$GOV" >/dev/null || echo "[!] cpupower not available"
nvidia-smi --query-gpu=name,clocks.gr,clocks.mem,clocks.sm,persistence_mode --format=csv
echo "[✓] clocks locked. (nvidia-smi -rgc / -rmc to reset)"
