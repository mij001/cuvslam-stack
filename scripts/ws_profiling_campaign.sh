#!/usr/bin/env bash
# ws_profiling_campaign.sh — profile cuVSLAM across many feature-toggle variants
# of each dataset, and confirm accuracy-vs-ground-truth does not deviate under
# profiling (no bug introduced by nsys instrumentation).
#
# Per config (configs/profiling_coverage/*.toml):
#   1. PLAIN   run.py <cfg>              -> eval.txt  -> plain APE  (reference)
#   2. PROFILE profile.py --profiler nsys -> .nsys-rep (the profile) + the same
#              run re-computes eval.txt  -> nsys APE
#   3. compare: |nsys - plain| within tolerance => OK (profiling is neutral),
#              else CHECK (flagged; note km-scale odometry is nondeterministic
#              run-to-run per F13, so a large Δ there is not necessarily a bug).
#
# Progress is streamed to $LOG — `tail -f ~/profiling_coverage.log`.
# Resumable: a variant with a non-empty results line in $DONE is skipped.
# Run:  setsid nohup scripts/ws_profiling_campaign.sh > ~/profiling_coverage.boot 2>&1 &
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"      # cd's to the repo root
HW=profiling/hw/dellworkstation_sm89.toml
PY=./cuvslam_venv/bin/python
OUT=/mnt/data/profiling_coverage_out
LOG=~/profiling_coverage.log
DONE="$OUT/DONE.tsv"
export PATH=/opt/cuda/bin:$PATH
RUN_TIMEOUT=${RUN_TIMEOUT:-2400}

mkdir -p "$OUT"
log_init "$LOG"      # fresh log for tail -f
[ -f "$DONE" ] || echo -e "variant\tmode\tplain_APE_m\tnsys_APE_m\tdelta_m\tmatched\tstatus" > "$DONE"

# free the GPU and lock clocks once (KDE compositor perturbs captures)
log "freeing GPU + locking clocks"
free_gpu
lock_gpu_clocks 1620 7001

log "generating feature-toggle configs"
python3 scripts/gen_profiling_coverage.py --matrix configs/accuracy_matrix --out configs/profiling_coverage 2>&1 | tee -a "$LOG"

CFGS=$(ls configs/profiling_coverage/*.toml | sort)
TOTAL=$(echo "$CFGS" | wc -l)
log "profiling-coverage campaign start — $TOTAL variants"
log "  legend: each variant runs PLAIN then under NSYS; Δ = |nsysAPE - plainAPE|"
printf '%-52s %-9s %10s %10s %9s %7s %s\n' variant mode plainAPE nsysAPE delta matched status | tee -a "$LOG"

N=0 okc=0 checkc=0
for CFG in $CFGS; do
    N=$((N+1))
    TAG=$(basename "$CFG" .toml)
    if grep -qP "^${TAG}\t" "$DONE" 2>/dev/null; then continue; fi
    D="$OUT/$TAG"; mkdir -p "$D"
    mode=$(grep -oE 'odometry_mode = "[A-Za-z]+"' "$CFG" | grep -oE '"[A-Za-z]+"' | tr -d '"')
    if grep -q '^\[slam\]' "$CFG"; then
        if grep -q 'sync_mode = false' "$CFG"; then mode="$mode/async"; else mode="$mode/slam"; fi
    else
        mode="$mode/odom"
    fi

    # 1) PLAIN accuracy (reference). For a __base breadth variant the plain run
    #    is exactly the accuracy-matrix run, whose eval already exists — reuse it
    #    (no re-run). Only the finer-toggle variants (new configs) run plain here.
    ACC_BASE=""; case "$TAG" in *__base) ACC_BASE="${TAG%__base}";; esac
    if [ -n "$ACC_BASE" ] && [ -s "/mnt/data/accuracy_out/$ACC_BASE/eval.txt" ]; then
        cp -f "/mnt/data/accuracy_out/$ACC_BASE/eval.txt" "$D/eval_plain.txt"
    else
        timeout "$RUN_TIMEOUT" "$PY" run.py "$CFG" > "$D/plain.log" 2>&1 || true
        cp -f "$D/eval.txt" "$D/eval_plain.txt" 2>/dev/null || true
    fi
    P_APE=$(ape_of "$D/eval_plain.txt"); P_MATCH=$(matched_of "$D/eval_plain.txt")

    # 2) PROFILE under nsys (full sequence) — the same run re-writes eval.txt
    timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$CFG" \
        --profiler nsys --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" \
        > "$D/nsys.log" 2>&1 || true
    N_APE=$(ape_of "$D/eval.txt"); N_MATCH=$(matched_of "$D/eval.txt")
    REP=$(ls -dt profiling/results/*_nsys_* 2>/dev/null | head -1)

    # 3) compare — tolerance: 5 cm or 5% of plain APE, whichever larger
    status="OK"; delta="-"
    if [ -n "${P_APE:-}" ] && [ -n "${N_APE:-}" ]; then
        delta=$(awk -v a="$P_APE" -v b="$N_APE" 'BEGIN{d=a-b; print (d<0?-d:d)}')
        tol=$(awk -v a="$P_APE" 'BEGIN{t=0.05*a; print (t>0.05?t:0.05)}')
        status=$(awk -v d="$delta" -v t="$tol" 'BEGIN{print (d<=t?"OK":"CHECK")}')
    else
        status="NO_APE"
    fi
    [ "$status" = "OK" ] && okc=$((okc+1)) || checkc=$((checkc+1))
    printf '%-52s %-9s %10s %10s %9s %7s %s\n' \
        "[$N/$TOTAL] $TAG" "${mode:-?}" "${P_APE:--}" "${N_APE:--}" "$delta" "${N_MATCH:--}" "$status" | tee -a "$LOG"
    echo -e "$TAG\t${mode:-?}\t${P_APE:--}\t${N_APE:--}\t$delta\t${N_MATCH:--}\t$status" >> "$DONE"
    cp -f "$LOG" "$OUT/campaign.log" 2>/dev/null || true
done

log "DONE — $TOTAL variants profiled; $okc accuracy-neutral (OK), $checkc flagged (CHECK — inspect; may be km-scale odometry nondeterminism)"
log "profiles: profiling/results/*_nsys_* ; per-variant evals: $OUT/<variant>/eval_plain.txt"
restore_gui
