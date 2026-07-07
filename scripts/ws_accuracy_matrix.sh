#!/usr/bin/env bash
# ws_accuracy_matrix.sh — run the full accuracy matrix (workstation).
#
# Phase 1: every configs/accuracy_matrix/*.toml on the BASELINE wheel
#          (cuvslam_venv) — trajectories + [eval] reports vs ground truth
#          land in /mnt/data/accuracy_out/<run_name>/.
# Phase 2: QoR-neutrality check — a representative subset re-run on the
#          INSTRUMENTED wheel (cuvslam_venv_tagged, journal off) into
#          /mnt/data/accuracy_out/qor_tagged/<run_name>/ so the two wheels'
#          metrics can be diffed.
#
# Resumable: a run is skipped when its eval.txt exists and is non-empty.
# Run:   setsid nohup scripts/ws_accuracy_matrix.sh &
# Watch: tail -f ~/accuracy_matrix.log ; cat /mnt/data/accuracy_out/PROGRESS
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"      # cd's to the repo root
OUT=/mnt/data/accuracy_out
LOG=~/accuracy_matrix.log
PY=./cuvslam_venv/bin/python
PY_TAGGED=./cuvslam_venv_tagged/bin/python
RUN_TIMEOUT=${RUN_TIMEOUT:-2400}

log_attach "$LOG"          # append across resumes
progress() {
    echo "$(date +%F_%H:%M:%S) $*" > "$OUT/PROGRESS" 2>/dev/null || true
    cp "$LOG" "$OUT/matrix.log" 2>/dev/null || true
}

ensure_data_rw /mnt/data /dev/sda2 || exit 1

python3 scripts/gen_accuracy_configs.py --root /mnt/data --tumvi-extracted "$HOME/tumvi_extracted" \
    --out configs/accuracy_matrix 2>&1 | tee -a "$LOG"

CFGS=$(ls configs/accuracy_matrix/*.toml | sort)
TOTAL=$(echo "$CFGS" | wc -l)
log "accuracy matrix start — $TOTAL runs (baseline wheel)"

N=0
for CFG in $CFGS; do
    N=$((N+1))
    NAME=$(basename "$CFG" .toml)
    D="$OUT/$NAME"
    mkdir -p "$D"
    if [ -s "$D/eval.txt" ]; then continue; fi
    log "[$N/$TOTAL] $NAME"
    progress "[$N/$TOTAL] $NAME"
    timeout "$RUN_TIMEOUT" $PY run.py "$CFG" > "$D/stdout.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -s "$D/eval.txt" ]; then
        log "[$N/$TOTAL] $NAME FAILED rc=$rc (see $D/stdout.txt)"
        mv "$D/eval.txt" "$D/eval_partial.txt" 2>/dev/null
    fi
done

# ── Phase 2: instrumented-wheel QoR pairs ────────────────────────────────────
QOR="kitti06_stereo_slam euroc_V1_01_easy_stereo_slam euroc_MH_01_easy_inertial_slam \
     tum_fr3_long_office_household_rgbd_slam kitti00_stereo_odom euroc_V2_02_medium_stereo_odom"
log "QoR phase — instrumented wheel on: $QOR"
for NAME in $QOR; do
    CFG="configs/accuracy_matrix/$NAME.toml"
    [ -f "$CFG" ] || { log "QoR skip $NAME (no config)"; continue; }
    D="$OUT/qor_tagged/$NAME"
    mkdir -p "$D"
    if [ -s "$D/eval.txt" ]; then continue; fi
    progress "QoR $NAME"
    sed "s|$OUT/$NAME|$D|g" "$CFG" > "$D/config.toml"
    timeout "$RUN_TIMEOUT" $PY_TAGGED run.py "$D/config.toml" > "$D/stdout.txt" 2>&1 \
        || log "QoR $NAME FAILED (see $D/stdout.txt)"
done

log "matrix DONE — summary:"
OK=0; BAD=0
for CFG in $CFGS; do
    NAME=$(basename "$CFG" .toml)
    if [ -s "$OUT/$NAME/eval.txt" ]; then OK=$((OK+1)); else BAD=$((BAD+1)); echo "  ✗ $NAME" | tee -a "$LOG"; fi
done
log "$OK ok, $BAD failed of $TOTAL"
progress "MATRIX DONE ($OK/$TOTAL)"
