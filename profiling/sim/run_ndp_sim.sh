#!/usr/bin/env bash
# run_ndp_sim.sh — the actual NDP evaluation: cycle-level Accel-Sim runs over
# the archetypes' real SASS traces, baseline vs NDP overlays.
#
# This is the ZSim+Ramulator role in DAMOV, realized: trace-driven simulation
# of {baseline, NDP-conservative, NDP-moderate} per archetype, reporting cycle
# DELTAS. The baseline config approximates the traced GPU (SM86_RTX3070 base
# with clusters set to 22 SMs — deltas are the claim, not absolutes; standing
# rule 5). NDP overlays come from gen_ndp_config.py --accelsim-base (DRAM clk
# x k, core clk x c_r, L2 sets/8).
#
#   profiling/sim/run_ndp_sim.sh [traces_root=~/ndp_traces] [jobs=4]
#
# Resumable: skips (arch, config) pairs whose log already ends in the cycle
# line. Results: ~/ndp_sim/<arch>.<config>.log + ndp_sim_results.csv
set -uo pipefail
AS=~/Projects/cuvslam-stack/external_repos/accel-sim-framework
# the simulator resolves libgpgpusim symbols at RUN time via the env the build
# used — source it here or face `undefined symbol: shd_warp_t`. The setup
# script touches unset vars, so suspend `set -u` around it (with -u active a
# sourced unbound-variable kills this whole script SILENTLY).
export CUDA_INSTALL_PATH=/opt/cuda
set +u
cd "$AS/gpu-simulator" && source ./setup_environment.sh >/dev/null 2>&1
cd - >/dev/null
set -u
BIN=$AS/gpu-simulator/bin/release/accel-sim.out
# gpgpusim.config lives under gpgpu-sim/configs; trace.config under the
# accel-sim-level configs — two different trees for the same GPU name
BASE_CFG_DIR=$AS/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTX3070
TRACE_CFG=$AS/gpu-simulator/configs/tested-cfgs/SM86_RTX3070/trace.config
TROOT="${1:-$HOME/ndp_traces}"
JOBS="${2:-4}"
OUT=~/ndp_sim
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "accel-sim.out not built"; exit 1; }

# baseline localized to 22 SMs (our part) — one knob, documented
cat > "$OUT/base_local.config" <<EOF
-gpgpu_n_clusters 22
EOF

# NDP overlays (idempotent regeneration against the real base config)
python3 ~/Projects/cuvslam-stack/profiling/sim/gen_ndp_config.py \
    --accelsim-base "$BASE_CFG_DIR/gpgpusim.config" \
    --out ~/Projects/cuvslam-stack/profiling/sim/configs >/dev/null 2>&1 || true
CONS=~/Projects/cuvslam-stack/profiling/sim/configs/conservative.accelsim.config
MODE=~/Projects/cuvslam-stack/profiling/sim/configs/moderate.accelsim.config

run_one() {  # arch cfgname extra_config
    local arch=$1 cfg=$2 extra=$3
    local log="$OUT/${arch}.${cfg}.log"
    grep -q "gpu_tot_sim_cycle" "$log" 2>/dev/null && return 0
    local tr="$TROOT/$arch/traces/kernelslist.g"
    [ -f "$tr" ] || { echo "no trace for $arch"; return 1; }
    local args=(-trace "$tr" -config "$BASE_CFG_DIR/gpgpusim.config"
                -config "$TRACE_CFG" -config "$OUT/base_local.config")
    [ -n "$extra" ] && args+=(-config "$extra")
    ( cd "$OUT" && timeout 5400 "$BIN" "${args[@]}" > "$log" 2>&1 )
    local cyc
    cyc=$(grep -oE "gpu_tot_sim_cycle\s*=\s*[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    echo "[sim] $arch/$cfg: cycles=${cyc:-FAILED}"
}
export -f run_one; export OUT TROOT BIN BASE_CFG_DIR TRACE_CFG LD_LIBRARY_PATH GPGPUSIM_ROOT GPGPUSIM_CONFIG

ARCHES=(g1_triad g2_gather g3_l2 g4_chase g5_fma g6_shared g7_dep g0_tiny)
jobs_list=()
for arch in "${ARCHES[@]}"; do
    jobs_list+=("$arch baseline ''" "$arch ndp_cons $CONS" "$arch ndp_mod $MODE")
done
printf '%s\n' "${jobs_list[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'

# collect
CSV="$OUT/ndp_sim_results.csv"
echo "archetype,config,cycles" > "$CSV"
for arch in "${ARCHES[@]}"; do
    for cfg in baseline ndp_cons ndp_mod; do
        cyc=$(grep -oE "gpu_tot_sim_cycle\s*=\s*[0-9]+" "$OUT/${arch}.${cfg}.log" 2>/dev/null \
              | tail -1 | grep -oE "[0-9]+")
        echo "$arch,$cfg,${cyc:-}" >> "$CSV"
    done
done
echo; column -t -s, "$CSV"
