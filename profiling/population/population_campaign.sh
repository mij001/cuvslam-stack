#!/usr/bin/env bash
# population_campaign.sh — the DAMOV two-phase validation, at population scale,
# on REAL codebases (run on the target after fetch_and_build.sh).
#
# Per app in ~/gpu_workloads/manifest.tsv:
#   1) ncu `characterize` capture at locked base clocks (via the SAME harness +
#      command adapter that profiles cuVSLAM) -> blind classification with the
#      frozen thresholds (analysis.classify — untouched).
#   2) three plain nsys passes at the intervention clock points
#      (base 1620/7001, half-core 810/7001, low-mem 1620/5001) ->
#      per-kernel time via cuda_gpu_kern_sum -> S_core, S_mem per kernel.
#
# Phase-2 correctness (paper §3.5): a kernel is CORRECT iff its blind class's
# clock-response signature matches its measured (S_core, S_mem). Analysis is
# done repo-side by analysis/population.py over the CSVs this emits.
#
# Windowing policy (documented design decision): iterative apps (rodinia,
# babelstream) skip warm-up launches; single-launch apps (polybench, most
# samples) must use skip=0 — their kernels launch exactly once, so cold-start
# effects are part of the measurement (as they were for DAMOV's functions).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export PATH=/opt/cuda/bin:$PATH
NCU_DEFAULT="$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu"
[ -z "${NCU_BIN:-}" ] && [ -x "$NCU_DEFAULT" ] && export NCU_BIN="$NCU_DEFAULT"

MAN=~/gpu_workloads/manifest.tsv
OUT=${1:-/mnt/data/population_out}
HW="profiling/hw/dellworkstation_sm89.toml"
mkdir -p "$OUT/cls" "$OUT/sweep" configs/population
INDEX="$OUT/population_index.tsv"
[ -s "$INDEX" ] || printf 'app\tstatus\tresults_dir\tcls_csv\n' > "$INDEX"

BASE_GFX=1620; BASE_MEM=7001; LOW_GFX=810; LOW_MEM=5001
# 4th point, OFF both swept axes (1005/5001): held-out validation of the
# per-kernel linear clock model t = a/f_core + b/f_mem + c that the first
# three points determine exactly — the NDP evaluator's fit is only trusted
# where this point is predicted well (see analysis/ndp_eval.py)
HOLD_GFX=1005; HOLD_MEM=5001
lock() { sudo -n nvidia-smi -lgc "$1,$1" >/dev/null; sudo -n nvidia-smi -lmc "$2,$2" >/dev/null; sleep 2; }
trap 'lock $BASE_GFX $BASE_MEM' EXIT

window() {  # app -> "skip count"
    case "$1" in
        rod_*)        echo "8 48" ;;
        babelstream)  echo "20 50" ;;
        *)            echo "0 64" ;;   # single-launch apps: cold start included
    esac
}

while IFS=$'\t' read -r app dir cmd; do
    [ -n "$app" ] || continue
    grep -qP "^${app}\t(ncu-ok|done)" "$INDEX" && { echo "[skip] $app (done)"; continue; }
    echo "== $app =="

    # ── workload config (command adapter) ───────────────────────────────────
    cfg="configs/population/${app}.toml"
    printf '# population workload: %s\n[workload]\ncmd = "%s"\ncwd = "%s"\n' \
        "$app" "$cmd" "$dir" > "$cfg"

    # ── 1 · ncu characterize at base clocks -> classify ─────────────────────
    lock $BASE_GFX $BASE_MEM
    read -r SKIP COUNT <<< "$(window "$app")"
    ./cuvslam_venv/bin/python profiling/harness/profile.py \
        --config "$cfg" --profiler ncu --hw "$HW" \
        --metrics characterize --launch-skip "$SKIP" --launch-count "$COUNT" \
        --timeout 900 > "$OUT/cls/${app}.profile.log" 2>&1
    rdir=$(ls -dt profiling/results/*_ncu_* 2>/dev/null | head -1)
    if [ -z "$rdir" ] || [ ! -s "$rdir/derived/ncu_metrics.csv" ]; then
        printf '%s\tncu-FAILED\t-\t-\n' "$app" >> "$INDEX"; continue
    fi
    # NB: $OUT is absolute — pass it bare (a ../ prefix would silently re-root
    # an absolute path under the repo, which is exactly the bug that ate run 1)
    ( cd profiling && ../cuvslam_venv/bin/python -m analysis.classify "../$rdir" \
        --hw "../$HW" --out "$OUT/cls/${app}" ) >> "$OUT/cls/${app}.profile.log" 2>&1
    if [ ! -s "$OUT/cls/${app}/classification.csv" ]; then
        printf '%s\tclassify-FAILED\t%s\t-\n' "$app" "$rdir" >> "$INDEX"; continue
    fi
    # raw counters travel home too — metric discovery mines them directly
    cp "$rdir/derived/ncu_metrics.csv" "$OUT/cls/${app}/ncu_metrics.csv" 2>/dev/null

    # ── 2 · clock-point sweep: nsys kern_sum at 3 points ────────────────────
    ok=1
    for point in "base:$BASE_GFX:$BASE_MEM" "lowcore:$LOW_GFX:$BASE_MEM" \
                 "lowmem:$BASE_GFX:$LOW_MEM" "holdout:$HOLD_GFX:$HOLD_MEM"; do
        IFS=: read -r tag gfx mem <<< "$point"
        lock "$gfx" "$mem"
        rep="$OUT/sweep/${app}_${tag}"
        ( cd "$dir" && timeout 300 nsys profile --force-overwrite true -t cuda \
              -o "$rep" $cmd ) > /dev/null 2>&1 || { ok=0; break; }
        nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
            --output "$rep" "$rep.nsys-rep" > /dev/null 2>&1
        [ -s "${rep}_cuda_gpu_kern_sum.csv" ] || { ok=0; break; }
        rm -f "$rep.nsys-rep" "$rep.sqlite"
    done
    if [ "$ok" = 1 ]; then
        printf '%s\tdone\t%s\t%s\n' "$app" "$rdir" "$OUT/cls/${app}/classification.csv" >> "$INDEX"
        echo "[done] $app"
    else
        printf '%s\tsweep-FAILED\t%s\t%s\n' "$app" "$rdir" "$OUT/cls/${app}/classification.csv" >> "$INDEX"
        echo "[partial] $app (classified; sweep failed)"
    fi
done < "$MAN"

lock $BASE_GFX $BASE_MEM
echo
echo "== population campaign index =="
column -t -s$'\t' "$INDEX"
