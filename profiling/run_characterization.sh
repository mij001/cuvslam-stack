#!/usr/bin/env bash
# run_characterization.sh — the one-command Slice-2 pipeline, headless anywhere.
#
# preflight → nsys baseline → nsys SLAM (loop closure) → steady-state ncu
# (characterize metric set, window derived from the nsys run) → report.
#
#   profiling/run_characterization.sh                          # defaults (TUM office workloads)
#   profiling/run_characterization.sh --hw profiling/hw/rtx2000ada_sm89.toml
#   profiling/run_characterization.sh --config <odom.toml> --slam-config <slam.toml> \
#       --warm 200 --launches 300 --skip-slam
#
# If --hw is omitted and no descriptor matches this host, one is auto-generated
# (structural values exact; ceilings flagged for ERT verification).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"

HW=""
CONFIG="profiling/configs/tum_office_profile.toml"
SLAM_CONFIG="profiling/configs/tum_office_slam_profile.toml"
WARM=200 LAUNCHES=300 SKIP_SLAM=0 TAG="char" REPEATS=5
while [ $# -gt 0 ]; do
    case "$1" in
        --hw) HW="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --slam-config) SLAM_CONFIG="$2"; shift 2 ;;
        --warm) WARM="$2"; shift 2 ;;
        --launches) LAUNCHES="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --skip-slam) SKIP_SLAM=1; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

echo "### 1/5 preflight"
profiling/env/check_env.sh || { echo "preflight failed"; exit 1; }

if [ -z "$HW" ]; then
    echo "### auto-selecting hardware descriptor"
    python3 profiling/env/gen_hw_descriptor.py
    slug=$(python3 -c "import os,re;print(re.sub(r'[^a-z0-9]+','',os.uname().nodename.split('.')[0].lower()) or 'host')")
    HW=$(ls profiling/hw/${slug}_sm*.toml | head -1)
fi
echo "hw descriptor: $HW"

echo "### 2/5 nsys baseline (odometry, x$REPEATS repeats): $CONFIG"
NSYS_REPEAT_DIRS=()
for i in $(seq 1 "$REPEATS"); do
    python3 profiling/harness/profile.py --config "$CONFIG" --profiler nsys --hw "$HW" \
        --tag "$TAG-r$i" --gpu-warmup 8
    NSYS_REPEAT_DIRS+=("$(ls -d profiling/results/*_nsys_* | tail -1)")
done
NSYS_DIR="${NSYS_REPEAT_DIRS[0]}"
echo "baseline nsys: $NSYS_DIR (${#NSYS_REPEAT_DIRS[@]} repeats)"

NSYS_SLAM_ARG=()
if [ "$SKIP_SLAM" -eq 0 ]; then
    echo "### 3/5 nsys SLAM (loop closure): $SLAM_CONFIG"
    python3 profiling/harness/profile.py --config "$SLAM_CONFIG" --profiler nsys --hw "$HW" --tag "${TAG}-slam" --gpu-warmup 8
    NSYS_SLAM_DIR=$(ls -d profiling/results/*_nsys_* | tail -1)
    NSYS_SLAM_ARG=(--nsys-slam "$NSYS_SLAM_DIR")
    echo "slam nsys: $NSYS_SLAM_DIR"
else
    echo "### 3/5 skipped (--skip-slam)"
fi

echo "### 4/5 ncu steady-state (characterize set, warm=$WARM launches=$LAUNCHES; cold+warm cache bracket)"
python3 profiling/harness/profile.py --config "$CONFIG" --profiler ncu --hw "$HW" \
    --metrics characterize --auto-window "$NSYS_DIR:$WARM:$LAUNCHES" \
    --gpu-warmup 8 --timeout 5400 --tag "${TAG}-steady"
NCU_DIR=$(ls -d profiling/results/*_ncu_* | tail -1)
python3 profiling/harness/profile.py --config "$CONFIG" --profiler ncu --hw "$HW" \
    --metrics characterize --auto-window "$NSYS_DIR:$WARM:$LAUNCHES" \
    --gpu-warmup 8 --cache-control none --timeout 5400 --tag "${TAG}-steady-warmcache"
NCU_WARM_DIR=$(ls -d profiling/results/*_ncu_* | tail -1)
echo "steady-state ncu: $NCU_DIR (cold) / $NCU_WARM_DIR (warm-cache bracket)"

echo "### 5/5 report"
python3 -m analysis.make_report --hw "$HW" --nsys "$NSYS_DIR" --ncu "$NCU_DIR" "${NSYS_SLAM_ARG[@]}"
REPORT_DIR=$(ls -dt profiling/reports/*/ | head -1)
if [ "${#NSYS_REPEAT_DIRS[@]}" -gt 1 ]; then
    echo "### variance across ${#NSYS_REPEAT_DIRS[@]} nsys repeats"
    (cd profiling && python3 -m analysis.variance \
        $(printf '../%s ' "${NSYS_REPEAT_DIRS[@]}") --out "../$REPORT_DIR/data")
fi
echo "### done"
