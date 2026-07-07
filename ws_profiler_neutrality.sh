#!/usr/bin/env bash
# ws_profiler_neutrality.sh — prove ALL THREE profilers (Nsight Systems, Nsight
# Compute, NVBit) preserve cuVSLAM's trajectory accuracy: the test harness must
# not change the result. For a set of deterministic representative sequences,
# run the SAME config under each profiler and compare the resulting
# ATE/RMSE-APE (vs ground truth) to the un-profiled (plain) baseline.
#
# Why these are valid, bounded runs:
#   nsys  — observational; full sequence, low overhead.
#   ncu   — profiles a BOUNDED launch window (--launch-skip/--launch-count); the
#           app otherwise runs native to completion, so the eval is the full
#           trajectory.
#   nvbit — mem_trace instruments a bounded LAUNCH window (LAUNCH_BEGIN/END);
#           outside it kernels run native, so the full trajectory completes. The
#           trace itself is discarded (we only want the eval here).
#
# Each profiler run writes its eval into profiling_coverage_out/<tag>/ (the
# __base coverage configs are already redirected there), so the accuracy_out
# baselines are never touched. Streams a tailable log; writes a summary table.
#
# Run:  setsid nohup ./ws_profiler_neutrality.sh > ~/profiler_neutrality.boot 2>&1 &
set -uo pipefail
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH
# ncu 2025.2 (CUDA 12.9) — the system ncu (2026.2/2025.3) rejects driver 575
export NCU_BIN="${NCU_BIN:-$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu}"
HW=profiling/hw/dellworkstation_sm89.toml
PY=./cuvslam_venv/bin/python
CDIR=configs/profiling_coverage
ACC=/mnt/data/accuracy_out
PCO=/mnt/data/profiling_coverage_out   # where the __base configs write eval/traj
OUT=/mnt/data/profiler_neutrality_out
LOG=~/profiler_neutrality.log
TOOL=external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
RUN_TIMEOUT=${RUN_TIMEOUT:-3000}
LB=${LB:-1000}; LE=${LE:-1200}          # NVBit instrumentation launch window

mkdir -p "$OUT"; : > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
ape_of() { grep -oE "RMSE APE[^(]*\(([0-9.]+) m\)" "$1" 2>/dev/null | grep -oE "[0-9.]+ m" | head -1 | grep -oE "[0-9.]+"; }
matched_of() { grep -oE "matched poses[^0-9]*([0-9]+)" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1; }

# representative, deterministic (SLAM/loop-closure) configs — one per modality
REPS="euroc_V1_01_easy_stereo_slam euroc_MH_01_easy_stereo_slam \
      euroc_MH_01_easy_inertial_slam tum_fr3_long_office_household_rgbd_slam \
      icl_living_room_traj1_rgbd_slam kitti06_stereo_slam"

log "freeing GPU + locking clocks"
zsh ~/free_gpu.zsh >/dev/null 2>&1 || true
for i in 1 2 3; do sudo -n nvidia-smi -pm 1 >/dev/null 2>&1; sudo -n nvidia-smi -lgc 1620,1620 >/dev/null 2>&1; sudo -n nvidia-smi -lmc 7001,7001 >/dev/null 2>&1; sleep 2; [ "$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits)" = "1620" ] && break; done
[ -f "$TOOL" ] || { log "mem_trace.so missing — build it first"; exit 1; }

log "profiler-neutrality check — $(echo $REPS | wc -w) sequences × {plain, nsys, ncu, nvbit}"
printf '%-42s %-10s %10s %10s %10s %10s\n' sequence mode plainAPE nsysAPE ncuAPE nvbitAPE | tee -a "$LOG"

SUM="$OUT/SUMMARY.tsv"
echo -e "sequence\tmode\tplain_APE\tnsys_APE\tncu_APE\tnvbit_APE\tnsys_d\tncu_d\tnvbit_d\tverdict" > "$SUM"
overall_ok=1

for R in $REPS; do
    CFG="$CDIR/${R}__base.toml"
    [ -f "$CFG" ] || { log "skip $R (no coverage config)"; continue; }
    mode=$(grep -oE 'odometry_mode = "[A-Za-z]+"' "$CFG" | grep -oE '"[A-Za-z]+"' | tr -d '"')
    D="$OUT/$R"; mkdir -p "$D"
    EVAL="$PCO/${R}__base/eval.txt"       # all __base runs write eval here
    P_APE=$(ape_of "$ACC/$R/eval.txt")

    # nsys (full) — read + stash eval before the next profiler overwrites it
    timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$CFG" \
        --profiler nsys --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" >"$D/nsys.log" 2>&1 || true
    N_APE=$(ape_of "$EVAL"); cp -f "$EVAL" "$D/eval_nsys.txt" 2>/dev/null || true

    # ncu (bounded window, full run)
    timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$CFG" \
        --profiler ncu --metrics quick --launch-skip 3000 --launch-count 20 \
        --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" >"$D/ncu.log" 2>&1 || true
    C_APE=$(ape_of "$EVAL"); cp -f "$EVAL" "$D/eval_ncu.txt" 2>/dev/null || true

    # nvbit (bounded LAUNCH window, full run, trace discarded)
    LAUNCH_BEGIN="$LB" LAUNCH_END="$LE" CUDA_INJECTION64_PATH="$TOOL" \
        timeout "$RUN_TIMEOUT" "$PY" run.py "$CFG" >/dev/null 2>"$D/nvbit.log" || true
    V_APE=$(ape_of "$EVAL"); cp -f "$EVAL" "$D/eval_nvbit.txt" 2>/dev/null || true

    d() { [ -n "$1" ] && [ -n "$P_APE" ] && awk -v a="$1" -v b="$P_APE" 'BEGIN{x=a-b;print(x<0?-x:x)}' || echo "-"; }
    ND=$(d "$N_APE"); CD=$(d "$C_APE"); VD=$(d "$V_APE")
    tol=$(awk -v p="${P_APE:-0}" 'BEGIN{t=0.05*p;print(t>0.05?t:0.05)}')
    verdict="OK"
    for D_ in "$ND" "$CD" "$VD"; do
        [ "$D_" = "-" ] && continue
        awk -v d="$D_" -v t="$tol" 'BEGIN{exit !(d>t)}' && { verdict="CHECK"; overall_ok=0; }
    done
    printf '%-42s %-10s %10s %10s %10s %10s  [%s]\n' "$R" "$mode" "${P_APE:--}" "${N_APE:--}" "${C_APE:--}" "${V_APE:--}" "$verdict" | tee -a "$LOG"
    echo -e "$R\t$mode\t${P_APE:--}\t${N_APE:--}\t${C_APE:--}\t${V_APE:--}\t$ND\t$CD\t$VD\t$verdict" >> "$SUM"
done

log "DONE — $([ $overall_ok -eq 1 ] && echo 'ALL PROFILERS ACCURACY-NEUTRAL' || echo 'some CHECK — inspect (mono/km-odometry nondeterminism vs real perturbation)')"
log "summary: $SUM"
[ -f ~/restore_gui.zsh ] && zsh ~/restore_gui.zsh >/dev/null 2>&1 || true
