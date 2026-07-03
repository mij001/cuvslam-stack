#!/usr/bin/env bash
# ws_slice3_campaign.sh — RESUMABLE Slice-3 traces + full campaign (workstation).
#
# Hardened after the first overnight run died (power event + an over-wide
# launch-window that instrumented ~6000 kernels to catch a dozen st_track
# launches). Fixes:
#   * KERNEL_FILTER traces only the target kernel — no aim pass, tiny traces;
#   * frees the GPU (kills the KDE compositor) and re-locks clocks first, so a
#     post-power-cut autologin desktop doesn't pollute traces or reset clocks;
#   * every step skips if its output already exists (rerun-safe);
#   * bounded locality (--max-launches / --max-accesses) so the parser can't
#     blow up.
#
# Run:  setsid nohup profiling/campaign/ws_slice3_campaign.sh > ~/campaign.log 2>&1 &
set -uo pipefail
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH
TOOL=external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
PY=./cuvslam_venv/bin/python
HW=profiling/hw/dellworkstation_sm89.toml
OUT=~/slice3
mkdir -p "$OUT"
log() { echo "=== [$(date +%F_%H:%M:%S)] $*"; }

# ── 0. free the GPU + lock clocks (idempotent; survives power-cut autologin) ──
# KDE doesn't pollute NVBit traces (injection is per-process), but its
# compositor competes for SMs during the ncu/nsys campaign captures — so free
# the GPU before both. Prefer a user free-GPU script if present, else stop the
# display-manager (keeps sshd/network up; reversible with `start`).
log "freeing GPU and locking clocks"
if [ -f ~/free_gpu.zsh ]; then
    log "running ~/free_gpu.zsh"; zsh ~/free_gpu.zsh || true
else
    for s in ~/free_gpu.sh ~/kill_kde.sh ~/stop_kde.sh; do
        [ -x "$s" ] && { log "running $s"; "$s" || true; break; }
    done
    sudo -n systemctl stop display-manager 2>/dev/null && log "stopped display-manager" || true
fi
# free_gpu.zsh resets the GPU (nukes /dev/nvidia0), which can drop the clock
# lock — retry until graphics actually pins to 1620.
for attempt in 1 2 3 4 5; do
    sudo -n nvidia-smi -pm 1 >/dev/null 2>&1
    sudo -n nvidia-smi -lgc 1620,1620 >/dev/null 2>&1
    sudo -n nvidia-smi -lmc 7001,7001 >/dev/null 2>&1
    sleep 2
    g=$(nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits)
    [ "${g:-0}" -ge 1600 ] 2>/dev/null && break
    log "clock-lock attempt $attempt: applications.graphics=$g, retrying"
done
log "clocks: $(nvidia-smi --query-gpu=clocks.applications.graphics,clocks.current.memory --format=csv,noheader) (target 1620/7001)"
log "GPU procs now: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) compute, display=$(nvidia-smi --query-gpu=display_active --format=csv,noheader)"

log "generating campaign configs (the trace section references them too)"
python3 profiling/campaign/gen_configs.py --root /mnt/data --tumvi-extracted ~/tumvi_extracted || true

expand() { sed "s|\${CUVSLAM_DATASETS}|$HOME/Projects/cuvslam_datasets|g; s|\${CUVSLAM_DATA2}|/mnt/data|g" "$1"; }

# trace_kernel <name> <config> <KERNEL_FILTER> [max_frames]
trace_kernel() {
    local name="$1" cfg="$2" filt="$3" frames="${4:-}"
    local zst="$OUT/$name.zst"
    if [ -s "$zst" ]; then log "skip trace $name (exists)"; return; fi
    local tmp=/tmp/s3_$name.toml
    expand "$cfg" > "$tmp"
    [ -n "$frames" ] && sed -i "s/^max_frames.*/max_frames = $frames/" "$tmp"
    log "trace $name (KERNEL_FILTER=$filt)"
    KERNEL_FILTER="$filt" CUDA_INJECTION64_PATH=$TOOL \
        timeout 7200 $PY run.py "$tmp" 2>/dev/null | zstd -3 -T0 -f -o "$zst" \
        || { log "trace $name FAILED/timeout"; return 1; }
    log "trace $name done: $(du -h "$zst" | cut -f1)"
}

locality() {  # name [extra locality args...]
    local name="$1"; shift
    [ -s "$OUT/$name.zst" ] || { log "locality $name: no trace"; return; }
    [ -s "$OUT/loc_$name/locality.csv" ] && { log "skip locality $name"; return; }
    log "locality $name"
    (cd profiling && python3 -m analysis.locality "$OUT/$name.zst" \
        --out "$OUT/loc_$name" --max-launches 40 --max-accesses 4000000 "$@" 2>&1 | tail -12)
}

# ── 1. front-end steady state (all per-frame kernels, one frame's worth) ─────
#     preprocess+feature+tracking scanning: KERNEL_FILTER="" would be huge, so
#     trace a single frame's launches via a narrow LAUNCH window instead.
if [ ! -s "$OUT/tum_frontend.zst" ]; then
    expand profiling/configs/tum_office_profile.toml > /tmp/s3_fe.toml
    sed -i "s/^max_frames.*/max_frames = 205/" /tmp/s3_fe.toml
    log "trace tum_frontend (1 steady frame, launch window)"
    LAUNCH_BEGIN=14400 LAUNCH_END=14480 CUDA_INJECTION64_PATH=$TOOL \
        timeout 3600 $PY run.py /tmp/s3_fe.toml 2>/dev/null | zstd -3 -T0 -f -o "$OUT/tum_frontend.zst" || true
    log "tum_frontend: $(du -h "$OUT/tum_frontend.zst" 2>/dev/null | cut -f1)"
fi
locality tum_frontend

# ── 2. loop-closure scan: st_track_with_cache across three map scales ────────
trace_kernel tum_sttrack   profiling/configs/tum_office_slam_profile.toml       st_track_with_cache
trace_kernel kitti00_sttrack profiling/configs/campaign/kitti00_slam.toml       st_track_with_cache
trace_kernel kitti06_sttrack profiling/configs/campaign/kitti06_slam.toml       st_track_with_cache
locality tum_sttrack     --kernel st_track_with_cache
locality kitti00_sttrack --kernel st_track_with_cache
locality kitti06_sttrack --kernel st_track_with_cache

# ── 3. bundle-adjust build kernel (hot-persistent, per-frame) ────────────────
trace_kernel tum_ba profiling/configs/tum_office_profile.toml "build_full_system_1" 210
locality tum_ba --kernel build_full_system_1

log "Slice-3 traces complete: $(ls "$OUT"/*.zst 2>/dev/null | wc -l) traces"

# ── 4. the 29-sequence campaign (its own resumability via result-dir names) ──
log "campaign start"
profiling/campaign/run_campaign.sh --hw "$HW"
log "ALL DONE — restoring GUI"
[ -f ~/restore_gui.zsh ] && zsh ~/restore_gui.zsh || sudo -n systemctl start display-manager 2>/dev/null || true
