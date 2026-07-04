#!/usr/bin/env bash
# ws_attribution_gapfill.sh — close per-sequence kernel-coverage gaps left by
# the windowed campaign (ws_attribution_all.sh). For each sequence,
# plan_gapfill.py compares the pass1 launch map against the joins already on
# disk and plans up to 4 launch windows covering every cuVSLAM kernel that has
# no attribution rows yet; each window is captured (all kernels), resolved and
# joined into join_gapfill_<i>/. Sequences with no gaps are skipped, so the
# script is idempotent: rerun until the audit is clean.
#
# Run:  setsid nohup profiling/campaign/ws_attribution_gapfill.sh &
# Watch: tail -f ~/attribution_campaign.log
set -uo pipefail
cd ~/Projects/cuvslam-stack
export PATH=/opt/cuda/bin:$PATH
TOOL=$PWD/external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
PY=./cuvslam_venv_tagged/bin/python
OUT=/mnt/data/attribution_out
LOG=~/attribution_campaign.log
STEP_TIMEOUT=${STEP_TIMEOUT:-10800}

log() { echo "=== [$(date +%F_%H:%M:%S)] $*" | tee -a "$LOG"; }
progress() {
    echo "$(date +%F_%H:%M:%S) $*" > "$OUT/PROGRESS" 2>/dev/null || true
    cp "$LOG" "$OUT/campaign.log" 2>/dev/null || true
}

if ! touch "$OUT/.w" 2>/dev/null; then
    sudo -n umount /mnt/data 2>/dev/null
    sudo -n mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data \
        || { echo "FATAL: cannot mount /mnt/data rw" | tee -a "$LOG"; exit 1; }
fi
rm -f "$OUT/.w"

log "gapfill start"
N=0
for D in "$OUT"/*/; do
    NAME=$(basename "$D")
    [ -f "$D/pass1_launchmap.txt.zst" ] || continue
    N=$((N+1))
    PLAN=$(cd profiling && python3 campaign/plan_gapfill.py "$D" 2>>"$LOG")
    if [ -z "$PLAN" ]; then
        log "[$NAME] no gaps"
        continue
    fi
    while read -r B E COVER; do
        # outputs keyed by the window itself: a rerun that plans DIFFERENT
        # windows (launch-id drift between runs moves sparse kernels) captures
        # them instead of matching a stale pass2c_<n> filename and skipping
        W="w${B}_${E}"
        TRACE="$D/pass2c_${W}_trace.txt.zst"
        if [ ! -s "$TRACE" ]; then
            log "[$NAME] gapfill window [$B,$E) covers $COVER kernel(s)"
            progress "$NAME gapfill [$B,$E)"
            timeout "$STEP_TIMEOUT" env \
                LAUNCH_BEGIN=$B LAUNCH_END=$E \
                CUDA_INJECTION64_PATH=$TOOL \
                MEM_TRACE_ALLOC_LOG="$D/pass2c_${W}_nvbit_allocs.csv" \
                CUVSLAM_ALLOC_LOG="$D/pass2c_${W}_cuvslam_allocs.csv" \
                $PY run.py "$D/config.toml" 2>"$D/pass2c_${W}_stderr.txt" \
                | zstd -3 -T0 -f -o "$TRACE"
            log "[$NAME] gapfill [$B,$E) done ($(du -h "$TRACE" | cut -f1))"
        fi
        if [ ! -s "$D/join_gapfill_${W}/attribution.csv" ]; then
            (cd profiling
             python3 -m analysis.attribution resolve "$D/pass2c_${W}_cuvslam_allocs.csv" \
                 --out "$D/pass2c_${W}_alloc_table.csv" >> "$LOG" 2>&1
             python3 -m analysis.attribution join "$TRACE" \
                 "$D/pass2c_${W}_alloc_table.csv" "$D/pass2c_${W}_nvbit_allocs.csv" \
                 --max-accesses-per-kernel 5000000 \
                 --out "$D/join_gapfill_${W}" >> "$LOG" 2>&1)
        fi
    done <<< "$PLAN"
    log "[$NAME] gapfill complete"
done
log "gapfill DONE ($N sequences checked)"
progress "GAPFILL DONE"
