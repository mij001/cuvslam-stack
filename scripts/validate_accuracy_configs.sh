#!/usr/bin/env bash
# validate_accuracy_configs.sh — prove every accuracy-matrix config (and every
# derived sub-config: odom/slam/slam_async/slam_cpu/mono/inertial) is correct
# under BOTH harnesses:
#   1. the TOML runner (`run.py --check`)   — config loads, data resolves, rig valid
#   2. the profiling flow (`profile.py`)    — profile.py can wrap it under nsys
#
# Run on the workstation (data + GPU present). GPU is only touched by the single
# profiling smoke; --check is CPU-only. Prints a per-config PASS/FAIL table and
# exits non-zero if any config fails.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"      # cd's to the repo root
CFGDIR="${1:-configs/accuracy_matrix}"
PY="${CUVSLAM_PYTHON:-./cuvslam_venv/bin/python}"
HW="${HW:-profiling/hw/dellworkstation_sm89.toml}"

pass=0 fail=0; FAILED=()
echo "== Phase 1: run.py --check on every config =="
for cfg in "$CFGDIR"/*.toml; do
    if out=$("$PY" run.py "$cfg" --check 2>&1) && echo "$out" | grep -q "configuration is valid"; then
        # Only image_folder/tum sources report a numeric frame count; euroc is a
        # streaming source and reports "(live / unbounded)" — that is valid.
        # Flag a NUMERIC count < 2 (catches data/association bugs, e.g. ICL).
        fc=$(echo "$out" | grep -oE "frame count *= *[0-9]+")
        frames=$(echo "$fc" | grep -oE "[0-9]+" | head -1)
        if [ -n "$fc" ] && [ "${frames:-2}" -lt 2 ]; then
            echo " FAIL  $(basename "$cfg")  (only ${frames:-0} frames — data/association)"
            fail=$((fail+1)); FAILED+=("$(basename "$cfg"):frames=${frames:-0}")
        else
            pass=$((pass+1))
        fi
    else
        echo " FAIL  $(basename "$cfg")"
        echo "$out" | grep -iE "error|not found|missing|traceback" | head -1 | sed 's/^/        /'
        fail=$((fail+1)); FAILED+=("$(basename "$cfg")")
    fi
done
echo "  check: $pass pass, $fail fail"

echo "== Phase 2: profiling-flow compatibility (profile.py wraps one config per dataset family) =="
# one representative per family; nsys with a tiny frame window (fast), just to
# prove profile.py consumes an accuracy config end to end.
# one representative per input-source/mode family (image_folder stereo, euroc
# stereo/inertial/mono, tum RGBD, ICL RGBD) — proves profile.py handles each.
reps=$(for fam in kitti00_stereo euroc_MH_01_easy_stereo euroc_MH_01_easy_inertial \
                  euroc_MH_01_easy_mono tum_fr3_long_office_household_rgbd \
                  icl_traj0_rgbd icl_living_room_traj0_rgbd; do
         ls "$CFGDIR"/${fam}_odom.toml 2>/dev/null | head -1; done)
pfp=0 pff=0
for cfg in $reps; do
    [ -f "$cfg" ] || continue
    if "$PY" profiling/harness/profile.py --config "$cfg" --profiler nsys --hw "$HW" \
         --frames 0:20 --gpu-warmup 0 --timeout 300 >/tmp/pf_$$.log 2>&1 \
       && ls profiling/results/*/raw/*.nsys-rep >/dev/null 2>&1; then
        echo " PASS  profile.py <- $(basename "$cfg")"; pfp=$((pfp+1))
    else
        echo " FAIL  profile.py <- $(basename "$cfg")"; tail -2 /tmp/pf_$$.log | sed 's/^/        /'; pff=$((pff+1))
    fi
done
echo "  profile-flow: $pfp pass, $pff fail"

echo "== SUMMARY =="
echo "  --check: $pass/$((pass+fail)) valid; profiling-flow: $pfp/$((pfp+pff))"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  failed configs:"; printf '   - %s\n' "${FAILED[@]}"
fi
[ $fail -eq 0 ] && [ $pff -eq 0 ] && echo "  ALL CONFIGS CORRECT UNDER BOTH HARNESSES" || exit 1
