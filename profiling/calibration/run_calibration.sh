#!/usr/bin/env bash
# run_calibration.sh — the DAMOV §3.5-style held-out validation, GPU edition.
#
# Builds the ground-truth archetype kernels (archetypes.cu — each DESIGNED to be
# one G-class), profiles each through the SAME harness + classifier that judges
# cuVSLAM (command adapter -> ncu characterize -> analysis.classify with the
# frozen thresholds), and emits a confusion matrix: designed class vs classified
# class. The classifier never sees the design intent — it only sees counters.
#
#   profiling/calibration/run_calibration.sh [outdir]
#
# Needs: nvcc on PATH (/opt/cuda/bin), NCU_BIN (CUDA-12.9 ncu on driver-575
# hosts), a free GPU. ~10 min total.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
OUT="${1:-reports/2026-07-09_damov_validation}"
HW="profiling/hw/dellworkstation_sm89.toml"
mkdir -p "$OUT" configs/calibration

export PATH=/opt/cuda/bin:$PATH
NCU_DEFAULT="$HOME/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu"
[ -z "${NCU_BIN:-}" ] && [ -x "$NCU_DEFAULT" ] && export NCU_BIN="$NCU_DEFAULT"

# ── build ────────────────────────────────────────────────────────────────────
# the host default gcc may be newer than the CUDA release supports (the doctor's
# nvcc-vs-gcc trap) — pick the newest CUDA-compatible host compiler present
CCBIN=""
for g in g++-14 g++-13 g++-12; do
    command -v "$g" >/dev/null && { CCBIN="-ccbin $g"; break; }
done
if [ ! -x "$HERE/archetypes" ] || [ "$HERE/archetypes.cu" -nt "$HERE/archetypes" ]; then
    echo "[build] nvcc $CCBIN archetypes.cu"
    # shellcheck disable=SC2086
    nvcc $CCBIN -O3 -arch=native -o "$HERE/archetypes" "$HERE/archetypes.cu" || exit 1
fi

# ── the ground truth: archetype -> designed class ───────────────────────────
ARCHES=(g1_triad g2_gather g3_l2 g4_chase g5_fma g6_shared g7_dep g0_tiny)
declare -A EXPECT=(
    [g1_triad]=G1-bandwidth [g2_gather]=G2-coalescing [g3_l2]=G3-l2-reuse
    [g4_chase]=G4-latency   [g5_fma]=G5-compute       [g6_shared]=G6-onchip
    [g7_dep]=G7-dependency  [g0_tiny]=screened-Step1
)

RESULTS="$OUT/calibration_results.csv"
echo "archetype,designed_class,classified_class,match,confidence,rationale" > "$RESULTS"

for arch in "${ARCHES[@]}"; do
    cfg="configs/calibration/${arch}.toml"
    cat > "$cfg" <<EOF
# calibration archetype: ${arch} (designed ground truth: ${EXPECT[$arch]})
[workload]
cmd = "profiling/calibration/archetypes ${arch} 8"
[qor]
stdout_regex = "checksum=([0-9.eE+-]+)"
EOF
    echo "[cal] $arch — profiling (ncu characterize, launches 2-4)"
    ./cuvslam_venv/bin/python profiling/harness/profile.py \
        --config "$cfg" --profiler ncu --hw "$HW" \
        --metrics characterize --launch-skip 2 --launch-count 3 \
        --timeout 900 > "$OUT/${arch}.profile.log" 2>&1
    rdir=$(ls -dt profiling/results/*_ncu_* 2>/dev/null | head -1)
    if [ -z "$rdir" ] || [ ! -s "$rdir/derived/ncu_metrics.csv" ]; then
        echo "$arch,${EXPECT[$arch]},CAPTURE-FAILED,no,," >> "$RESULTS"
        continue
    fi
    ( cd profiling && ../cuvslam_venv/bin/python -m analysis.classify "../$rdir" \
        --hw "../$HW" --out "../$OUT/${arch}_cls" ) > "$OUT/${arch}.classify.log" 2>&1
    ./cuvslam_venv/bin/python - "$arch" "${EXPECT[$arch]}" "$OUT/${arch}_cls/classification.csv" "$RESULTS" <<'PY'
import csv, sys
arch, expect, cls_csv, results = sys.argv[1:5]
rows = list(csv.DictReader(open(cls_csv)))
# the archetype kernel is the one matching its own name (harness may also see memsets)
r = next((x for x in rows if arch.split("_")[0] in x["kernel"] or arch in x["kernel"]), None)
if r is None and rows:
    r = max(rows, key=lambda x: float(x.get("time_ms") or 0))
got = r["class"] if r else "NOT-FOUND"
# DAMOV applies the Step-1 SCREEN before classification: sub-floor kernels are
# never classified. g0 is designed to fail the screen — correct behavior is
# "screened out", whatever the tree would have said.
t_ms = float(r.get("time_ms") or 0) if r else 0.0
if arch == "g0_tiny":
    screened = t_ms < 0.05
    got = f"screened({t_ms:.4f}ms<0.05)" if screened else got
    ok = "yes" if screened else "no"
else:
    ok = "yes" if got == expect else "no"
with open(results, "a", newline="") as fh:
    csv.writer(fh).writerow([arch, expect, got, ok,
                             r.get("confidence", "") if r else "",
                             (r.get("rationale", "") if r else "").replace(",", ";")])
print(f"  -> designed {expect}  classified {got}  [{'OK' if ok=='yes' else 'MISS'}]")
PY
done

echo
echo "== confusion summary =="
column -t -s, "$RESULTS"
ok=$(grep -c ",yes," "$RESULTS" || true)
echo "correct: $ok / ${#ARCHES[@]}"
