#!/usr/bin/env bash
# run_campaign.sh — the full-scale characterization campaign, headless.
#
# NOTE: on the downgraded driver-575 workstation, set NCU_BIN to the CUDA-12.9
# ncu 2025.2 (the system ncu 2026.2 / 2025.3 are CUDA-13, driver-incompatible):
export NCU_BIN="${NCU_BIN:-$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu}"
#
# For every configs/campaign/<name>_odom.toml (+ matching _slam.toml):
#   odom:  3x warmed nsys repeats + 1 ncu characterize (steady window 200:300)
#   slam:  1 warmed nsys (full sequence) + 1 ncu on the st_* SLAM kernels
# Failures are logged and skipped — one bad sequence never kills the campaign.
#
#   profiling/campaign/run_campaign.sh --hw profiling/hw/<gpu>.toml [--only REGEX]
#
# Run under `setsid nohup ... &` (survives ssh); progress in the campaign log.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

HW="" ONLY="."
while [ $# -gt 0 ]; do
    case "$1" in
        --hw) HW="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done
[ -n "$HW" ] || { echo "--hw required"; exit 2; }

P=python3
step() { echo "=== [$(date +%H:%M:%S)] $*"; }
run() { "$@" >/dev/null 2>&1 || echo "    [!] FAILED: $*"; }

configs=$(ls "$REPO_ROOT"/configs/campaign/*_odom.toml 2>/dev/null | grep -E "$ONLY" || true)
n=$(echo "$configs" | grep -c . || true)
step "campaign start: $n sequences, hw=$HW"
i=0
done_tag() { ls -d profiling/results/*"$1"* >/dev/null 2>&1; }
for odom in $configs; do
    i=$((i+1))
    name=$(basename "$odom" _odom.toml)
    slam="${odom%_odom.toml}_slam.toml"
    if done_tag "camp-$name-steady" && { [ ! -f "$slam" ] || done_tag "camp-$name-slamncu"; }; then
        step "[$i/$n] $name — already captured, skipping"; continue
    fi
    step "[$i/$n] $name — odometry (3x nsys + ncu characterize)"
    for r in 1 2 3; do
        run $P profiling/harness/profile.py --config "$odom" --profiler nsys \
            --hw "$HW" --tag "camp-$name-r$r" --gpu-warmup 8
    done
    NSYS_DIR=$(ls -d profiling/results/*_nsys_* 2>/dev/null | tail -1)
    run $P profiling/harness/profile.py --config "$odom" --profiler ncu --hw "$HW" \
        --metrics characterize --auto-window "$NSYS_DIR:200:300" \
        --gpu-warmup 8 --timeout 5400 --tag "camp-$name-steady"
    if [ -f "$slam" ]; then
        step "[$i/$n] $name — SLAM (nsys + st_* ncu)"
        run $P profiling/harness/profile.py --config "$slam" --profiler nsys \
            --hw "$HW" --tag "camp-$name-slam" --gpu-warmup 8 --timeout 3600
        run $P profiling/harness/profile.py --config "$slam" --profiler ncu --hw "$HW" \
            --metrics characterize --kernel-filter "st_" \
            --launch-skip 20 --launch-count 120 --gpu-warmup 8 --timeout 5400 \
            --tag "camp-$name-slamncu"
    fi
done
step "campaign done — $(ls -d profiling/results/* | wc -l) total results dirs"
df -h "$REPO_ROOT" | tail -1
