#!/usr/bin/env bash
# run_nvbit_memtrace.sh — Slice-3: per-warp memory address traces via NVBit.
#
# GATED: exits immediately with the reason if check_capability.sh fails.
#
#   profiling/blocked/run_nvbit_memtrace.sh <runner-config.toml> [out.zst]
#
# Streams NVBit mem_trace output (kernel_id, warp_id, pc, addr, size, R/W)
# through zstd so uncompressed tuples never touch the (small) disk. The trace
# feeds locality analysis (DAMOV locality.cpp, GPU-adapted) and Accel-Sim.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

"$HERE/check_capability.sh" || exit 1

CONFIG="${1:?usage: $0 <runner-config.toml> [out.zst]}"
OUT="${2:-$REPO_ROOT/profiling/results/$(date +%Y-%m-%d_%H%M%S)_memtrace.zst}"
NVBIT_DIR="${NVBIT_DIR:-$REPO_ROOT/external_repos/nvbit_release_x86_64}"
VENV_PY="${CUVSLAM_PYTHON:-$REPO_ROOT/cuvslam_venv/bin/python}"
TOOL="$NVBIT_DIR/tools/mem_trace/mem_trace.so"

if [ ! -f "$TOOL" ]; then
    echo "[build] mem_trace tool"
    make -C "$NVBIT_DIR/tools/mem_trace"
fi

command -v zstd >/dev/null || { echo "zstd required (pacman/apt install zstd)"; exit 1; }

echo "[trace] $CONFIG -> $OUT  (expect 5-50x slowdown; NEVER use for timing)"
CUDA_INJECTION64_PATH="$TOOL" \
    "$VENV_PY" "$REPO_ROOT/run.py" "$CONFIG" 2>/dev/null | zstd -1 -o "$OUT"
echo "[done] $(du -h "$OUT" | cut -f1) compressed"
