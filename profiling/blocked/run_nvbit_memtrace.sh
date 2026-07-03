#!/usr/bin/env bash
# run_nvbit_memtrace.sh — Slice-3: per-warp memory address traces via NVBit.
#
# GATED: exits immediately with the reason if check_capability.sh fails.
#
#   LAUNCH_BEGIN=N LAUNCH_END=M profiling/blocked/run_nvbit_memtrace.sh \
#       <runner-config.toml> [out.zst]
#
# LAUNCH_BEGIN/LAUNCH_END (grid-launch-id window; requires
# mem_trace_launch_window.patch, applied by this script) bound the trace to a
# steady-state or loop-closure window — full-run traces are TB-scale at
# 100-1000x slowdown. Align window ids with a prior nsys run's launch indices
# (analysis.build_dag gives kernels/frame). Output streams through zstd; feed
# the result to `python3 -m analysis.locality <out.zst>` for footprint /
# reuse-distance-vs-capacity / coalescing / inter-launch-overlap tables.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

"$HERE/check_capability.sh" || exit 1

CONFIG="${1:?usage: $0 <runner-config.toml> [out.zst]}"
OUT="${2:-$REPO_ROOT/profiling/results/$(date +%Y-%m-%d_%H%M%S)_memtrace.zst}"
NVBIT_DIR="${NVBIT_DIR:-$REPO_ROOT/external_repos/nvbit_release_x86_64}"
VENV_PY="${CUVSLAM_PYTHON:-$REPO_ROOT/cuvslam_venv/bin/python}"
TOOL="$NVBIT_DIR/tools/mem_trace/mem_trace.so"

# ensure the launch-window patch is applied (idempotent), then build
if ! grep -q LAUNCH_BEGIN "$NVBIT_DIR/tools/mem_trace/mem_trace.cu"; then
    echo "[patch] launch-window support"
    (cd "$NVBIT_DIR/tools/mem_trace" && patch -N -p0 < "$HERE/mem_trace_launch_window.patch")
fi
if [ ! -f "$TOOL" ] || [ "$NVBIT_DIR/tools/mem_trace/mem_trace.cu" -nt "$TOOL" ]; then
    echo "[build] mem_trace tool"
    make -C "$NVBIT_DIR/tools/mem_trace"
fi

command -v zstd >/dev/null || { echo "zstd required (pacman/apt install zstd)"; exit 1; }

echo "[trace] $CONFIG -> $OUT  window=[${LAUNCH_BEGIN:-0}, ${LAUNCH_END:-inf})"
echo "        (expect 5-50x slowdown in-window; NEVER use traces for timing)"
LAUNCH_BEGIN="${LAUNCH_BEGIN:-0}" LAUNCH_END="${LAUNCH_END:-18446744073709551615}" \
CUDA_INJECTION64_PATH="$TOOL" \
    "$VENV_PY" "$REPO_ROOT/run.py" "$CONFIG" 2>/dev/null | zstd -3 -T0 -o "$OUT"
echo "[done] $(du -h "$OUT" | cut -f1) compressed"
