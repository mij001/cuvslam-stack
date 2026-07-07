#!/usr/bin/env bash
# ws_attribution_capture.sh — TaggedAllocator/NVTX attribution captures
# (workstation). Produces everything analysis.attribution needs, from THREE
# runs of the instrumented wheel (patches/0002-tagged-allocator-nvtx.patch,
# installed in cuvslam_venv_tagged) on the TUM fr3 long_office SLAM workload:
#
#   pass 1  NVBit injected, empty window (LAUNCH_END=0): near-native full-run
#           LAUNCH map + driver alloc sidecar + Layer-1 journal. Windows for
#           pass 2 are read off this map (cast_image_kernel_rgb = frame start).
#           Also: the nsys NVTX capture for the kernel→stage table.
#   pass 2a all-kernel steady-state window (~3 frames mid-orbit, padded for
#           launch-id drift between runs).
#   pass 2b KERNEL_FILTER=st_ — every keyframe-database scan
#           (st_track_with_cache / st_build_cache) across the whole sequence.
#
# Each capture keeps its own journal+sidecar: the join is per-process (ASLR).
# Traces are NEVER used for timing, so clocks are not locked here.
#
# Run:  setsid nohup profiling/campaign/ws_attribution_capture.sh \
#           > ~/attribution/capture.log 2>&1 &
set -uo pipefail
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH
OUT=${OUT:-~/attribution}
TOOL=$PWD/external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
PY=./cuvslam_venv_tagged/bin/python
WINDOW_BEGIN=${WINDOW_BEGIN:-99200}
WINDOW_END=${WINDOW_END:-99960}
mkdir -p "$OUT"
log() { echo "=== [$(date +%F_%H:%M:%S)] $*"; }

sed -e "s|\${CUVSLAM_DATASETS}/tum|/mnt/data/TUM_RGBD/extracted|" \
    configs/profiling/tum_office_slam_profile.toml > "$OUT/tum_office_slam_full.toml"

if [ ! -f "$OUT/pass1_launchmap.txt.zst" ]; then
    log "pass 1: launch map + journals (uninstrumented)"
    LAUNCH_BEGIN=0 LAUNCH_END=0 \
    CUDA_INJECTION64_PATH=$TOOL \
    MEM_TRACE_ALLOC_LOG=$OUT/pass1_nvbit_allocs.csv \
    CUVSLAM_ALLOC_LOG=$OUT/pass1_cuvslam_allocs.csv \
        $PY run.py "$OUT/tum_office_slam_full.toml" 2>"$OUT/pass1_stderr.txt" \
        | zstd -3 -T0 -f -o "$OUT/pass1_launchmap.txt.zst"
fi

if [ ! -f "$OUT/nvtx_stage_map.nsys-rep" ]; then
    log "nsys NVTX capture (600 frames) for the kernel->stage table"
    sed "s/max_frames  = 0/max_frames  = 600/" "$OUT/tum_office_slam_full.toml" \
        > "$OUT/tum_office_slam_600.toml"
    nsys profile -t nvtx,cuda -o "$OUT/nvtx_stage_map" --force-overwrite true \
        $PY run.py "$OUT/tum_office_slam_600.toml" > "$OUT/nsys_stdout.txt" 2>&1
    nsys stats --report nvtx_kern_sum --format csv --output "$OUT/nvtx_stage_map" \
        "$OUT/nvtx_stage_map.nsys-rep" >/dev/null 2>&1 || true
fi

if [ ! -f "$OUT/pass2a_trace.txt.zst" ]; then
    log "pass 2a: all-kernel steady-state window [$WINDOW_BEGIN,$WINDOW_END)"
    LAUNCH_BEGIN=$WINDOW_BEGIN LAUNCH_END=$WINDOW_END \
    CUDA_INJECTION64_PATH=$TOOL \
    MEM_TRACE_ALLOC_LOG=$OUT/pass2a_nvbit_allocs.csv \
    CUVSLAM_ALLOC_LOG=$OUT/pass2a_cuvslam_allocs.csv \
        $PY run.py "$OUT/tum_office_slam_full.toml" 2>"$OUT/pass2a_stderr.txt" \
        | zstd -3 -T0 -f -o "$OUT/pass2a_trace.txt.zst"
fi

if [ ! -f "$OUT/pass2b_trace.txt.zst" ]; then
    log "pass 2b: st_ keyframe-database scans (whole sequence)"
    KERNEL_FILTER=st_ \
    CUDA_INJECTION64_PATH=$TOOL \
    MEM_TRACE_ALLOC_LOG=$OUT/pass2b_nvbit_allocs.csv \
    CUVSLAM_ALLOC_LOG=$OUT/pass2b_cuvslam_allocs.csv \
        $PY run.py "$OUT/tum_office_slam_full.toml" 2>"$OUT/pass2b_stderr.txt" \
        | zstd -3 -T0 -f -o "$OUT/pass2b_trace.txt.zst"
fi

log "resolve + join (workstation: needs addr2line + the exact .so of the runs)"
cd profiling
for p in pass1 pass2a pass2b; do
    python3 -m analysis.attribution resolve "$OUT/${p}_cuvslam_allocs.csv" \
        --out "$OUT/${p}_alloc_table.csv"
done
python3 -m analysis.attribution join "$OUT/pass2a_trace.txt.zst" \
    "$OUT/pass2a_alloc_table.csv" "$OUT/pass2a_nvbit_allocs.csv" \
    --out "$OUT/join_steady_state"
python3 -m analysis.attribution join "$OUT/pass2b_trace.txt.zst" \
    "$OUT/pass2b_alloc_table.csv" "$OUT/pass2b_nvbit_allocs.csv" \
    --out "$OUT/join_st_scans"
log "done — commit the small CSVs (alloc tables, joins, nvtx_kern_sum);"
log "       traces stay on this host"
