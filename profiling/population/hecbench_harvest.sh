#!/usr/bin/env bash
# hecbench_harvest.sh — scale the population the DAMOV way (they screened 77K
# functions from 345 applications; we harvest a benchmark super-suite).
#
# HeCBench (zjin-lcf/HeCBench) carries ~400 REAL CUDA benchmark apps in a
# uniform layout (src/<name>-cuda/{Makefile,*.cu}) maintained for modern CUDA —
# the closest GPU analog to DAMOV's multi-suite population. This harvester:
#
#   * clones once (shallow), walks src/*-cuda/
#   * builds each with a time budget (nvcc + g++-14 host compiler forced)
#   * extracts the app's own canonical run command from its Makefile `run:`
#     target (HeCBench convention: `./$(program) <args>`)
#   * smoke-runs it (30 s budget); only build+run survivors join the manifest
#   * APPENDS to ~/gpu_workloads/manifest.tsv as hb_<name> rows
#
# Expect a large failure fraction (CUDA-12.9 strictness, missing data files,
# GPU-arch asserts): that is fine — DAMOV also kept only what screened in.
#   profiling/population/hecbench_harvest.sh [max_apps]
set -uo pipefail
export PATH=/opt/cuda/bin:$PATH
CCBIN=g++-14
W=~/gpu_workloads
MAN="$W/manifest.tsv"
MAX=${1:-160}
mkdir -p "$W"

if [ ! -d "$W/HeCBench" ]; then
    echo "[clone] HeCBench (shallow)"
    git clone -q --depth 1 https://github.com/zjin-lcf/HeCBench "$W/HeCBench" || exit 1
fi

built=0; tried=0
for d in "$W"/HeCBench/src/*-cuda; do
    [ -d "$d" ] || continue
    name=$(basename "$d"); name=${name%-cuda}
    app="hb_${name//[^a-zA-Z0-9]/_}"
    grep -qP "^${app}\t" "$MAN" 2>/dev/null && continue          # already harvested
    [ "$built" -ge "$MAX" ] && break
    [ -f "$d/Makefile" ] || continue
    tried=$((tried + 1))

    # ── extract the canonical run command from the Makefile `run:` target ──
    runcmd=$(awk '/^run:/{flag=1;next} flag && /^\t/{sub(/^\t+/,""); print; exit}' "$d/Makefile")
    runcmd=${runcmd//\$(LAUNCHER)/}
    # resolve $(program) from the Makefile
    prog=$(awk -F'= *' '/^program[ \t]*[:+]?=/{print $2; exit}' "$d/Makefile")
    runcmd=${runcmd//\$(program)/$prog}
    case "$runcmd" in ""|*'$('*) continue ;; esac                # unresolved vars -> skip

    # ── build (forced host compiler; their Makefiles honor CC/NVCC vars) ──
    ( cd "$d" && timeout 240 make -s CC="nvcc -ccbin $CCBIN" NVCC="nvcc -ccbin $CCBIN" \
        CUDA_ARCH="sm_89" ARCH="sm_89" -j4 ) >/dev/null 2>&1
    binfile="$d/${prog:-none}"
    [ -x "$binfile" ] || { rm -f "$d"/*.o 2>/dev/null; continue; }

    # ── smoke (their run args; 30 s budget; must exit 0) ──
    ( cd "$d" && timeout 30 $runcmd ) >/dev/null 2>&1 || { continue; }

    printf '%s\t%s\t%s\n' "$app" "$d" "$runcmd" >> "$MAN"
    built=$((built + 1))
    echo "[OK $built] $app"
done

echo
echo "harvested: $built new apps (tried $tried) — manifest now $(wc -l < "$MAN") rows"
