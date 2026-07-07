#!/usr/bin/env bash
# validation_regime.sh — THE campaign: every config in the matrix (bases +
# mutations) gets ACCURACY (plain) and PROFILING (nsys, ncu, and — where the
# config carries `[profiling] nvbit = true` — the deep NVBit memory-trace leg),
# every cell validated against ground truth and, via the paper report, against
# the cuVSLAM paper's own numbers.
#
#   matrix   configs/base/*.toml + configs/generated/*.toml   (~192)
#   modes    plain (reused from accuracy_out when present), nsys, ncu,
#            nvbit (marked configs only — the human's per-config knob)
#
# No frame windowing: full sequences by default. The ncu/nvbit LAUNCH windows
# below are capture bounding (the app still runs the whole trajectory), and
# are optional/overridable:
#   NCU_WINDOW="skip:count"    default 3000:20   ("" = full replay — very slow)
#   NVBIT_WINDOW="begin:count" default 1000:200  ("" = trace everything — huge)
#
# Profiled cells run from a RETARGETED copy (outputs -> $OUT), so the plain
# accuracy_out baselines are never overwritten. Every profiled run completes
# the full trajectory -> comparable eval.txt per cell.
#
# After the capture loop the JOIN stage (scripts/join_regime.py) merges all
# cells + per-kernel ncu metrics in parallel (--jobs = all cores).
#
# Resumable per cell via $LEDGER. Watch: tail -f ~/validation_regime.log
# Run:  setsid nohup scripts/validation_regime.sh > ~/validation_regime.boot 2>&1 &
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"      # cd's to the repo root

HW=${HW:-profiling/hw/dellworkstation_sm89.toml}
PY=./cuvslam_venv/bin/python
ACC=/mnt/data/accuracy_out
OUT=/mnt/data/validation_regime_out
LOG=~/validation_regime.log
LEDGER="$OUT/REGIME.tsv"
RUN_TIMEOUT=${RUN_TIMEOUT:-3000}
NCU_WINDOW=${NCU_WINDOW-3000:20}
NVBIT_WINDOW=${NVBIT_WINDOW-1000:200}
ONLY=${ONLY:-}                      # optional regex filter over config tags
export PATH=/opt/cuda/bin:$PATH
export NCU_BIN="${NCU_BIN:-$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu}"

mkdir -p "$OUT/cfg"
log_attach "$LOG"
[ -f "$LEDGER" ] || echo -e "config\tmode\tplain_APE_m\tmode_APE_m\tdelta_m\tmatched\tstatus\tresults_dir" > "$LEDGER"

ensure_data_rw /mnt/data /dev/sda2 || exit 1
log "mutating configs (bases -> generated)"
python3 scripts/mutate_configs.py 2>&1 | tee -a "$LOG"

log "freeing GPU + locking clocks"
free_gpu
lock_gpu_clocks 1620 7001

# ── helpers ──────────────────────────────────────────────────────────────────
cfg_path() {  # tag -> its config file (base or generated)
    for d in configs/base configs/generated; do
        [ -f "$d/$1.toml" ] && { echo "$d/$1.toml"; return; }
    done
}

retargeted() {  # tag -> path of the output-retargeted copy (made once)
    local tag="$1" src; src=$(cfg_path "$tag")
    local dst="$OUT/cfg/$tag.toml"
    [ -s "$dst" ] || sed "s|/accuracy_out/$tag/|/validation_regime_out/$tag/|g" "$src" > "$dst"
    echo "$dst"
}

plain_cell() {  # accuracy leg: reuse accuracy_out eval, else run once
    local tag="$1" src; src=$(cfg_path "$tag")
    grep -qP "^${tag}\tplain\t" "$LEDGER" && return 0
    if [ ! -s "$ACC/$tag/eval.txt" ]; then
        mkdir -p "$ACC/$tag"
        timeout "$RUN_TIMEOUT" "$PY" run.py "$src" > "$ACC/$tag/stdout.txt" 2>&1 || true
    fi
    local ape match; ape=$(ape_of "$ACC/$tag/eval.txt"); match=$(matched_of "$ACC/$tag/eval.txt")
    printf '%-56s %-6s APE=%-10s matched=%s\n' "$tag" plain "${ape:--}" "${match:--}" | tee -a "$LOG"
    echo -e "$tag\tplain\t${ape:--}\t${ape:--}\t0\t${match:--}\t$([ -n "$ape" ] && echo OK || echo NO_APE)\t$ACC/$tag" >> "$LEDGER"
}

prof_cell() {  # one profiled cell through the ONE capture entrypoint
    local tag="$1" prof="$2"
    grep -qP "^${tag}\t${prof}\t" "$LEDGER" && return 0
    local cfg d="$OUT/$tag"; cfg=$(retargeted "$tag"); mkdir -p "$d"
    local extra=()
    case "$prof" in
        ncu)   extra=(--metrics quick)
               [ -n "$NCU_WINDOW" ] && extra+=(--launch-skip "${NCU_WINDOW%%:*}" --launch-count "${NCU_WINDOW##*:}") ;;
        nvbit) [ -n "$NVBIT_WINDOW" ] && extra=(--launch-skip "${NVBIT_WINDOW%%:*}" --launch-count "${NVBIT_WINDOW##*:}") ;;
    esac
    timeout "$RUN_TIMEOUT" "$PY" profiling/harness/profile.py --config "$cfg" \
        --profiler "$prof" --hw "$HW" --gpu-warmup 0 --timeout "$RUN_TIMEOUT" \
        "${extra[@]}" > "$d/$prof.log" 2>&1 || true
    local rdir; rdir=$(grep -oE "Results: .*" "$d/$prof.log" | tail -1 | cut -d" " -f2)
    local pape ape match delta="-" status="NO_APE"
    pape=$(ape_of "$ACC/$tag/eval.txt")
    ape=$(ape_of "$d/eval.txt"); match=$(matched_of "$d/eval.txt")
    cp -f "$d/eval.txt" "$d/eval_$prof.txt" 2>/dev/null || true
    if [ -n "${pape:-}" ] && [ -n "${ape:-}" ]; then
        delta=$(awk -v a="$pape" -v b="$ape" 'BEGIN{d=a-b;print(d<0?-d:d)}')
        local tol; tol=$(awk -v a="$pape" 'BEGIN{t=0.05*a;print(t>0.05?t:0.05)}')
        status=$(awk -v d="$delta" -v t="$tol" 'BEGIN{print(d<=t?"OK":"CHECK")}')
    fi
    printf '%-56s %-6s plain=%-10s prof=%-10s Δ=%-10s %s\n' \
        "$tag" "$prof" "${pape:--}" "${ape:--}" "$delta" "$status" | tee -a "$LOG"
    echo -e "$tag\t$prof\t${pape:--}\t${ape:--}\t$delta\t${match:--}\t$status\t${rdir:--}" >> "$LEDGER"
}

# ── the matrix ───────────────────────────────────────────────────────────────
TAGS=$( (ls configs/base/*.toml configs/generated/*.toml 2>/dev/null) \
        | xargs -n1 basename | sed 's/\.toml$//' | sort -u )
[ -n "$ONLY" ] && TAGS=$(echo "$TAGS" | grep -E "$ONLY" || true)
TOTAL=$(echo "$TAGS" | wc -l)
NVB=$(for t in $TAGS; do grep -q "^nvbit = true" "$(cfg_path "$t")" 2>/dev/null && echo "$t"; done | wc -l)
log "REGIME start — $TOTAL configs × {plain,nsys,ncu} + $NVB nvbit-marked"

N=0
for t in $TAGS; do
    N=$((N+1)); log "[$N/$TOTAL] $t"
    plain_cell "$t"
    prof_cell "$t" nsys
    prof_cell "$t" ncu
    if grep -q "^nvbit = true" "$(cfg_path "$t")"; then prof_cell "$t" nvbit; fi
done

# ── JOIN (parallel) + verdict + paper comparison ─────────────────────────────
log "joining cells + kernel metrics (parallel)"
"$PY" scripts/join_regime.py --ledger "$LEDGER" --out "$OUT" --jobs 0 2>&1 | tee -a "$LOG" || true
okc=$(tail -n +2 "$LEDGER" | awk -F'\t' '$7=="OK"' | wc -l)
ckc=$(tail -n +2 "$LEDGER" | awk -F'\t' '$7=="CHECK"' | wc -l)
log "REGIME DONE — $okc OK, $ckc CHECK (ledger: $LEDGER)"
log "paper-metric report (plain runs vs arXiv:2506.04359 tables):"
python3 scripts/accuracy_report.py 2>&1 | tail -5 | tee -a "$LOG" || true
restore_gui
