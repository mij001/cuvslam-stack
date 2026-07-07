#!/usr/bin/env bash
# validation_regime.sh — ONE campaign: {base + mutated configs} × {plain, nsys,
# ncu, nvbit}, each cell validated against ground truth (and, via the paper
# report, against the cuVSLAM paper's own numbers).
#
# Supersedes ws_profiling_campaign.sh (plain-vs-nsys over the coverage set) and
# ws_profiler_neutrality.sh (3-profiler check on 6 reps): both are single cells
# of this regime's matrix.
#
# The run vehicles are the coverage-set configs (configs/generated/coverage):
# every accuracy config exists there as <name>__base with its outputs
# redirected to $PCO, so profiled runs NEVER overwrite the plain accuracy_out
# baselines; toggle variants (__<toggle>) extend the matrix. Each profiled run
# goes through profiling/harness/profile.py — the single capture entrypoint —
# and completes the FULL trajectory (ncu/nvbit are launch-windowed), so every
# cell yields a comparable eval.txt.
#
# Scope tiers (cells = configs × profilers):
#   reps      6 representatives × {nsys, ncu, nvbit}            (~30 min)
#   accuracy  141 __base × nsys  +  reps × {ncu, nvbit}         (~1 day)
#   coverage  192 (all) × nsys   +  reps × {ncu, nvbit}         (~1.5 days)
#   full      192 × {nsys, ncu, nvbit}                          (weeks — think first)
# Plain cells come free: reused from accuracy_out when present, else run once.
#
# Resumable per cell via $LEDGER. Watch: tail -f ~/validation_regime.log
# Run:  setsid nohup scripts/validation_regime.sh [scope] > ~/validation_regime.boot 2>&1 &
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"      # cd's to the repo root

SCOPE="${1:-accuracy}"
HW=${HW:-profiling/hw/dellworkstation_sm89.toml}
PY=./cuvslam_venv/bin/python
ACC=/mnt/data/accuracy_out
PCO=/mnt/data/profiling_coverage_out
OUT=/mnt/data/validation_regime_out
LOG=~/validation_regime.log
LEDGER="$OUT/REGIME.tsv"
CDIR=configs/generated/coverage
RUN_TIMEOUT=${RUN_TIMEOUT:-3000}
export PATH=/opt/cuda/bin:$PATH
export NCU_BIN="${NCU_BIN:-$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu}"

REPS="euroc_V1_01_easy_stereo_slam euroc_MH_01_easy_stereo_slam \
      euroc_MH_01_easy_inertial_slam tum_fr3_long_office_household_rgbd_slam \
      icl_living_room_traj1_rgbd_slam kitti06_stereo_slam"

mkdir -p "$OUT"
log_attach "$LOG"
[ -f "$LEDGER" ] || echo -e "config\tprofiler\tplain_APE_m\tprof_APE_m\tdelta_m\tmatched\tstatus" > "$LEDGER"

ensure_data_rw /mnt/data /dev/sda2 || exit 1
log "regenerating configs (bases -> mutations)"
python3 scripts/mutate_configs.py --select coverage 2>&1 | tee -a "$LOG"

log "freeing GPU + locking clocks"
free_gpu
lock_gpu_clocks 1620 7001

# ── cell helpers ─────────────────────────────────────────────────────────────
plain_ape() {  # <tag> — reuse accuracy_out for __base tags, else run plain once
    local tag="$1" acc_base="" d="$PCO/$1"
    case "$tag" in *__base) acc_base="${tag%__base}";; esac
    if [ -n "$acc_base" ] && [ -s "$ACC/$acc_base/eval.txt" ]; then
        ape_of "$ACC/$acc_base/eval.txt"; return
    fi
    if [ ! -s "$d/eval_plain.txt" ]; then
        timeout "$RUN_TIMEOUT" "$PY" run.py "$CDIR/$tag.toml" >/dev/null 2>&1 || true
        cp -f "$d/eval.txt" "$d/eval_plain.txt" 2>/dev/null || true
    fi
    ape_of "$d/eval_plain.txt"
}

run_cell() {  # <tag> <profiler>  — one profiled run through the ONE entrypoint
    local tag="$1" prof="$2" cfg="$CDIR/$1.toml" d="$PCO/$1"
    grep -qP "^${tag}\t${prof}\t" "$LEDGER" && return 0     # resumable
    [ -f "$cfg" ] || { log "skip $tag/$prof (no config)"; return 0; }
    local pape; pape=$(plain_ape "$tag")
    case "$prof" in
        nsys)  timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$cfg" \
                   --profiler nsys --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" \
                   > "$d.$prof.log" 2>&1 || true ;;
        ncu)   timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$cfg" \
                   --profiler ncu --metrics quick --launch-skip 3000 --launch-count 20 \
                   --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" \
                   > "$d.$prof.log" 2>&1 || true ;;
        nvbit) timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$cfg" \
                   --profiler nvbit --launch-skip 1000 --launch-count 200 \
                   --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" \
                   > "$d.$prof.log" 2>&1 || true ;;
    esac
    mkdir -p "$d"
    local ape match delta="-" status="NO_APE"
    ape=$(ape_of "$d/eval.txt"); match=$(matched_of "$d/eval.txt")
    cp -f "$d/eval.txt" "$d/eval_$prof.txt" 2>/dev/null || true
    if [ -n "${pape:-}" ] && [ -n "${ape:-}" ]; then
        delta=$(awk -v a="$pape" -v b="$ape" 'BEGIN{d=a-b;print(d<0?-d:d)}')
        local tol; tol=$(awk -v a="$pape" 'BEGIN{t=0.05*a;print(t>0.05?t:0.05)}')
        status=$(awk -v d="$delta" -v t="$tol" 'BEGIN{print(d<=t?"OK":"CHECK")}')
    fi
    printf '%-52s %-6s plain=%-9s prof=%-9s Δ=%-9s %s\n' \
        "$tag" "$prof" "${pape:--}" "${ape:--}" "$delta" "$status" | tee -a "$LOG"
    echo -e "$tag\t$prof\t${pape:--}\t${ape:--}\t$delta\t${match:--}\t$status" >> "$LEDGER"
}

# ── the matrix, by scope ─────────────────────────────────────────────────────
ALL=$(ls "$CDIR"/*.toml 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/\.toml$//' | sort)
BASE_TAGS=$(echo "$ALL" | grep "__base$" || true)
case "$SCOPE" in
    reps)     N_CFG=$(echo $REPS | wc -w); log "scope=reps: $N_CFG reps × {nsys,ncu,nvbit}"
              for r in $REPS; do for p in nsys ncu nvbit; do run_cell "${r}__base" "$p"; done; done ;;
    accuracy) log "scope=accuracy: $(echo "$BASE_TAGS" | wc -l) __base × nsys + reps × {ncu,nvbit}"
              for t in $BASE_TAGS; do run_cell "$t" nsys; done
              for r in $REPS; do for p in ncu nvbit; do run_cell "${r}__base" "$p"; done; done ;;
    coverage) log "scope=coverage: $(echo "$ALL" | wc -l) configs × nsys + reps × {ncu,nvbit}"
              for t in $ALL; do run_cell "$t" nsys; done
              for r in $REPS; do for p in ncu nvbit; do run_cell "${r}__base" "$p"; done; done ;;
    full)     log "scope=full: $(echo "$ALL" | wc -l) configs × {nsys,ncu,nvbit} — long haul"
              for t in $ALL; do for p in nsys ncu nvbit; do run_cell "$t" "$p"; done; done ;;
    *) log "unknown scope '$SCOPE' (reps|accuracy|coverage|full)"; exit 2 ;;
esac

# ── verdict + paper comparison ───────────────────────────────────────────────
okc=$(tail -n +2 "$LEDGER" | awk -F'\t' '$NF=="OK"' | wc -l)
ckc=$(tail -n +2 "$LEDGER" | awk -F'\t' '$NF=="CHECK"' | wc -l)
log "REGIME DONE — $okc OK, $ckc CHECK (ledger: $LEDGER)"
log "paper-metric report (plain runs vs arXiv:2506.04359 tables):"
python3 scripts/accuracy_report.py 2>&1 | tail -5 | tee -a "$LOG" || true
restore_gui
