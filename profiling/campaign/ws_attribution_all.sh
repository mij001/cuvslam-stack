#!/usr/bin/env bash
# ws_attribution_all.sh — TaggedAllocator/NVTX attribution across the FULL
# dataset matrix (all 27 SLAM campaign sequences: KITTI 00-10, EuRoC ×11,
# TUM fr3 ×4, TUM-VI). Multi-hour/day, RESUMABLE, power-cut-safe.
#
# Per sequence (skips any step whose output already exists):
#   pass1   NVBit injected, empty window → full LAUNCH map + Layer-1 journal
#           + driver alloc sidecar (near-native speed)
#   window  auto-picked from pass1: ~3 frames of launches at mid-sequence
#   nsys    NVTX capture (≤600 frames) → kernel→stage table (nvtx_kern_sum)
#   pass2a  all-kernel steady-state window trace
#   pass2b  KERNEL_FILTER=st_ → every keyframe-database scan, whole sequence
#   resolve+join → per-kernel × data-structure attribution CSVs
#
# ALL outputs (raw traces + derived + logs) go to $OUT on sda
# (/mnt/data/attribution_out). The volume is fstab-mounted ro and the NTFS
# dirty bit survives power cuts, so the preamble re-asserts a rw,force mount.
#
# Run (workstation):
#   setsid nohup profiling/campaign/ws_attribution_all.sh &
# Watch:
#   tail -f ~/attribution_campaign.log        (mirrored to $OUT at each step)
#   cat /mnt/data/attribution_out/PROGRESS
set -uo pipefail
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH
TOOL=$PWD/external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
PY=./cuvslam_venv_tagged/bin/python
CFGDIR=$PWD/profiling/configs/campaign
OUT=/mnt/data/attribution_out
LOG=~/attribution_campaign.log
STEP_TIMEOUT=${STEP_TIMEOUT:-10800}         # 3 h per capture step, then move on

log() { echo "=== [$(date +%F_%H:%M:%S)] $*" | tee -a "$LOG"; }
progress() {
    echo "$(date +%F_%H:%M:%S) $*" > "$OUT/PROGRESS" 2>/dev/null || true
    cp "$LOG" "$OUT/campaign.log" 2>/dev/null || true
}

# ── sda rw (fstab says ro; NTFS dirty bit blocks plain rw after power cuts) ──
if ! touch "$OUT/.w" 2>/dev/null; then
    sudo -n umount /mnt/data 2>/dev/null
    sudo -n mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data \
        || { echo "FATAL: cannot mount /mnt/data rw" | tee -a "$LOG"; exit 1; }
fi
rm -f "$OUT/.w"; mkdir -p "$OUT"
log "campaign start — output root $OUT ($(df -h /mnt/data | awk 'NR==2{print $4}') free)"

expand() { sed "s|\${CUVSLAM_DATASETS}|$HOME/Projects/cuvslam_datasets|g; s|\${CUVSLAM_DATA2}|/mnt/data|g" "$1"; }

# capture <outfile.zst> <journal> <sidecar> <env...> — one instrumented run
capture() {
    local trace="$1" journal="$2" sidecar="$3" cfg="$4"; shift 4
    timeout "$STEP_TIMEOUT" env "$@" \
        CUDA_INJECTION64_PATH=$TOOL \
        MEM_TRACE_ALLOC_LOG="$sidecar" \
        CUVSLAM_ALLOC_LOG="$journal" \
        $PY run.py "$cfg" 2>"${trace%.txt.zst}_stderr.txt" \
        | zstd -3 -T0 -f -o "$trace"
    local rc=$?
    [ $rc -eq 124 ] && log "  WARNING: step hit ${STEP_TIMEOUT}s timeout (partial trace kept)"
    return 0
}

SEQS=$(ls "$CFGDIR"/*_slam.toml | sort)
TOTAL=$(echo "$SEQS" | wc -l)
N=0
for CFG in $SEQS; do
    N=$((N+1))
    NAME=$(basename "$CFG" _slam.toml)
    D="$OUT/$NAME"
    mkdir -p "$D"
    log "[$N/$TOTAL] $NAME"
    progress "[$N/$TOTAL] $NAME starting"
    expand "$CFG" > "$D/config.toml"

    # 1. pass 1 — launch map + journals
    if [ ! -s "$D/pass1_launchmap.txt.zst" ]; then
        log "  pass1: launch map + journals"
        progress "[$N/$TOTAL] $NAME pass1 (launch map)"
        capture "$D/pass1_launchmap.txt.zst" "$D/pass1_cuvslam_allocs.csv" \
                "$D/pass1_nvbit_allocs.csv" "$D/config.toml" LAUNCH_BEGIN=0 LAUNCH_END=0
    fi

    # 2. windows: pass2a = ~3 frames mid-sequence (all kernels); pass2b = a
    #    LATE window (~100 frames at 55%, loop-closure-likely) for st_ scans.
    #    A whole-sequence st_ trace is TB-scale (the loop-closure scan reads the
    #    growing keyframe DB every keyframe) — 40 GB for one EuRoC sequence —
    #    so pass2b is windowed here and the join is access-capped as a backstop.
    read -r W_BEGIN W_END WB_BEGIN WB_END <<< "$(zstdcat "$D/pass1_launchmap.txt.zst" 2>/dev/null | awk '
        match($0, /grid launch id [0-9]+/) {
            id = substr($0, RSTART+15, RLENGTH-15) + 0
            if (id > L) L = id
        }
        /frames_tracked/ { if (match($0, /frames_tracked.: [0-9]+/))
            F = substr($0, RSTART+17, RLENGTH-17) + 0 }
        END {
            if (L == 0) { print 0, 0, 0, 0; exit }
            if (F == 0) F = 1000
            lpf = L / F
            mid = int(L / 2); w = int(3 * lpf); if (w < 200) w = 200
            lb = int(0.55 * L); lw = int(100 * lpf); if (lw < 2000) lw = 2000
            print mid, mid + w, lb, lb + lw
        }')"
    if [ "${W_END:-0}" -eq 0 ]; then
        log "  ERROR: pass1 empty for $NAME — skipping sequence"
        progress "[$N/$TOTAL] $NAME FAILED (empty pass1)"
        continue
    fi
    log "  pass2a window [$W_BEGIN,$W_END)  pass2b st_ window [$WB_BEGIN,$WB_END)"

    # 3. nsys NVTX → kernel→stage table
    if [ ! -s "$D/nvtx_stage_map.nsys-rep" ]; then
        log "  nsys NVTX capture"
        progress "[$N/$TOTAL] $NAME nsys NVTX"
        sed "s/^max_frames.*/max_frames  = 600/" "$D/config.toml" > "$D/config_600.toml"
        timeout "$STEP_TIMEOUT" nsys profile -t nvtx,cuda -o "$D/nvtx_stage_map" \
            --force-overwrite true $PY run.py "$D/config_600.toml" \
            > "$D/nsys_stdout.txt" 2>&1 || log "  WARNING: nsys step failed/timed out"
        nsys stats --report nvtx_kern_sum --format csv --output "$D/nvtx_stage_map" \
            "$D/nvtx_stage_map.nsys-rep" >/dev/null 2>&1 || true
    fi

    # 4. pass 2a — all-kernel steady-state window
    if [ ! -s "$D/pass2a_trace.txt.zst" ]; then
        log "  pass2a: all-kernel window trace"
        progress "[$N/$TOTAL] $NAME pass2a (window trace)"
        capture "$D/pass2a_trace.txt.zst" "$D/pass2a_cuvslam_allocs.csv" \
                "$D/pass2a_nvbit_allocs.csv" "$D/config.toml" \
                LAUNCH_BEGIN=$W_BEGIN LAUNCH_END=$W_END
        log "  pass2a done ($(du -h "$D/pass2a_trace.txt.zst" | cut -f1))"
    fi

    # 5. pass 2b — st_ keyframe-database scans in the late window
    if [ ! -s "$D/pass2b_trace.txt.zst" ]; then
        log "  pass2b: st_ scans (late window)"
        progress "[$N/$TOTAL] $NAME pass2b (st_ scans)"
        capture "$D/pass2b_trace.txt.zst" "$D/pass2b_cuvslam_allocs.csv" \
                "$D/pass2b_nvbit_allocs.csv" "$D/config.toml" \
                LAUNCH_BEGIN=$WB_BEGIN LAUNCH_END=$WB_END KERNEL_FILTER=st_
        log "  pass2b done ($(du -h "$D/pass2b_trace.txt.zst" | cut -f1))"
    fi

    # 6. resolve + join
    if [ ! -s "$D/join_steady_state/attribution.csv" ] || [ ! -s "$D/join_st_scans/attribution.csv" ]; then
        log "  resolve + join"
        progress "[$N/$TOTAL] $NAME resolve+join"
        (cd profiling
         for p in pass1 pass2a pass2b; do
             [ -s "$D/${p}_cuvslam_allocs.csv" ] && \
                 python3 -m analysis.attribution resolve "$D/${p}_cuvslam_allocs.csv" \
                     --out "$D/${p}_alloc_table.csv" >> "$LOG" 2>&1
         done
         python3 -m analysis.attribution join "$D/pass2a_trace.txt.zst" \
             "$D/pass2a_alloc_table.csv" "$D/pass2a_nvbit_allocs.csv" \
             --max-accesses-per-kernel 5000000 \
             --out "$D/join_steady_state" >> "$LOG" 2>&1
         python3 -m analysis.attribution join "$D/pass2b_trace.txt.zst" \
             "$D/pass2b_alloc_table.csv" "$D/pass2b_nvbit_allocs.csv" \
             --max-accesses-per-kernel 5000000 \
             --out "$D/join_st_scans" >> "$LOG" 2>&1)
    fi
    log "[$N/$TOTAL] $NAME COMPLETE"
    progress "[$N/$TOTAL] $NAME COMPLETE"
done

log "ALL SEQUENCES DONE — summary:"
for CFG in $SEQS; do
    NAME=$(basename "$CFG" _slam.toml)
    if [ -s "$OUT/$NAME/join_steady_state/attribution.csv" ]; then
        echo "  ✓ $NAME" | tee -a "$LOG"
    else
        echo "  ✗ $NAME (incomplete)" | tee -a "$LOG"
    fi
done
progress "CAMPAIGN DONE"
cp "$LOG" "$OUT/campaign.log" 2>/dev/null || true
log "campaign finished — derived CSVs per sequence under $OUT/<sequence>/"
