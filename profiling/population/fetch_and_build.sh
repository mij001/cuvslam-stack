#!/usr/bin/env bash
# fetch_and_build.sh — acquire + build the REAL-CODEBASE population for the
# DAMOV-style two-phase validation (run on a profiling target).
#
# Suite selection (design decision, justified):
#   * Polybench-GPU and Rodinia appear in DAMOV's OWN application population
#     (its Table-8 sources include Polybench and Rodinia) — using their GPU
#     implementations gives cross-ISA continuity with the original study.
#   * BabelStream is the community-standard bandwidth kernel: a known-truth
#     G1 anchor from a codebase we did not write.
#   * CUDA samples add access-pattern diversity (transpose, histogram
#     atomics, scan) from a fourth independent codebase.
#
# Everything lands in ~/gpu_workloads (NOT in the repo; suites are external).
# Emits ~/gpu_workloads/manifest.tsv:  app <TAB> workdir <TAB> command
# Only apps that BUILD AND RUN (3s smoke) enter the manifest.
set -uo pipefail
export PATH=/opt/cuda/bin:$PATH
CCBIN=g++-14
ARCH=native
W=~/gpu_workloads
mkdir -p "$W"
MAN="$W/manifest.tsv"
: > "$MAN"

note() { printf '%s\n' "$*"; }
have() { [ -e "$1" ]; }

add() {  # app workdir cmd...  -> smoke-test then manifest
    local app="$1" dir="$2"; shift 2
    ( cd "$dir" && timeout 60 "$@" >/dev/null 2>&1 )
    if [ $? -eq 0 ]; then
        printf '%s\t%s\t%s\n' "$app" "$dir" "$*" >> "$MAN"
        note "[OK ] $app"
    else
        note "[SKIP] $app (build ok but smoke run failed)"
    fi
}

# ── 1 · BabelStream — SKIPPED (documented decision) ─────────────────────────
# Its CUDA model fought this stack three ways (custom CUDA_ARCH flag, explicit
# CMAKE_CUDA_COMPILER requirement, then loop-style #error under direct nvcc
# compile on CUDA 12.9 + g++-14). The population already carries pure-streaming
# G1 anchors from foreign code (Polybench 2D/3D conv stencils, Rodinia
# pathfinder/hotspot) and the g1_triad archetype IS the STREAM triad pattern —
# BabelStream's marginal value was branding, not information. Revisit only if
# a reviewer asks for it by name.

# ── 2 · Polybench-GPU (15 single-kernel apps; overrides their sm_20 flags) ──
if ! have "$W/polybenchGpu"; then
    git clone -q --depth 1 https://github.com/sgrauerg/polybenchGpu "$W/polybenchGpu"
fi
PB="$W/polybenchGpu/CUDA"
for d in 2DCONV 3DCONV 2MM 3MM ATAX BICG CORR COVAR FDTD-2D GEMM GESUMMV \
         GRAMSCHM MVT SYR2K SYRK; do
    [ -d "$PB/$d" ] || continue
    src=$(ls "$PB/$d"/*.cu 2>/dev/null | head -1); [ -n "$src" ] || continue
    bin="$PB/$d/$(basename "${src%.cu}").exe"
    if ! have "$bin"; then
        nvcc -O3 -arch=$ARCH -ccbin $CCBIN -o "$bin" "$src" >/dev/null 2>&1
    fi
    have "$bin" && add "pb_$(echo "$d" | tr 'A-Z-' 'a-z_')" "$PB/$d" "$bin" \
        || note "[FAIL] polybench $d build"
done

# ── 3 · Rodinia (CUDA-12-compatible subset; DAMOV-overlap suite) ────────────
if ! have "$W/gpu-rodinia"; then
    git clone -q --depth 1 https://github.com/yuhc/gpu-rodinia "$W/gpu-rodinia"
fi
R="$W/gpu-rodinia"
RC="$R/cuda"
NVXX="nvcc -O3 -arch=$ARCH -ccbin $CCBIN"

rod() {  # name builddir buildcmd... ; runs later via per-app 'add'
    local name="$1" dir="$2"; shift 2
    ( cd "$dir" && eval "$*" ) >/dev/null 2>&1 && return 0 || return 1
}

# backprop
rod backprop "$RC/backprop" "$NVXX -o backprop backprop.cu backprop_cuda.cu facetrain.c imagenet.c -I." \
  && add rod_backprop "$RC/backprop" ./backprop 1048576
# bfs
rod bfs "$RC/bfs" "$NVXX -o bfs bfs.cu" \
  && add rod_bfs "$RC/bfs" ./bfs "$R/data/bfs/graph1MW_6.txt"
# gaussian
rod gaussian "$RC/gaussian" "$NVXX -o gaussian gaussian.cu" \
  && add rod_gaussian "$RC/gaussian" ./gaussian -s 2048
# hotspot
rod hotspot "$RC/hotspot" "$NVXX -o hotspot hotspot.cu" \
  && add rod_hotspot "$RC/hotspot" ./hotspot 1024 2 1000 \
        "$R/data/hotspot/temp_1024" "$R/data/hotspot/power_1024" out.txt
# hotspot3D
rod hotspot3D "$RC/hotspot3D" "$NVXX -o 3D 3D.cu opt1.cu -I." \
  && add rod_hotspot3d "$RC/hotspot3D" ./3D 512 8 500 \
        "$R/data/hotspot3D/power_512x8" "$R/data/hotspot3D/temp_512x8" out.txt
# lud
rod lud "$RC/lud" "$NVXX -o cuda/lud_cuda cuda/lud.cu cuda/lud_kernel.cu common/common.c -Icommon -Icuda" \
  && add rod_lud "$RC/lud" ./cuda/lud_cuda -s 4096
# nw (needleman-wunsch)
rod nw "$RC/nw" "$NVXX -o needle needle.cu" \
  && add rod_nw "$RC/nw" ./needle 8192 10
# pathfinder
rod pathfinder "$RC/pathfinder" "$NVXX -o pathfinder pathfinder.cu" \
  && add rod_pathfinder "$RC/pathfinder" ./pathfinder 200000 200 20
# srad_v2
rod srad2 "$RC/srad/srad_v2" "$NVXX -o srad srad.cu" \
  && add rod_srad2 "$RC/srad/srad_v2" ./srad 4096 4096 0 127 0 127 0.5 4
# lavaMD
rod lavaMD "$RC/lavaMD" "make -s CUDA_DIR=/opt/cuda 'NVCC=nvcc -ccbin $CCBIN' 2>/dev/null || $NVXX -o lavaMD main.cu kernel/kernel_gpu_cuda_wrapper.cu util/timer/timer.c util/num/num.c -I." \
  && add rod_lavamd "$RC/lavaMD" ./lavaMD -boxes1d 20
# particlefilter (naive float version)
rod particlefilter "$RC/particlefilter" "$NVXX -o particlefilter_float ex_particle_CUDA_float_seq.cu" \
  && add rod_particlefilter "$RC/particlefilter" ./particlefilter_float -x 128 -y 128 -z 10 -np 100000
# streamcluster
rod streamcluster "$RC/streamcluster" "$NVXX -o sc_gpu streamcluster_cuda_cpu.cpp streamcluster_cuda.cu streamcluster_header.cu -I." \
  && add rod_streamcluster "$RC/streamcluster" ./sc_gpu 10 20 256 16384 16384 1000 none out.txt 1
# b+tree
rod btree "$RC/b+tree" "$NVXX -o b+tree.out ./main.c ./kernel/kernel_gpu_cuda_wrapper.cu ./kernel/kernel_gpu_cuda_wrapper_2.cu ./util/timer/timer.c ./util/num/num.c ./util/cuda/cuda.cu -I. -DTIMER" \
  && add rod_btree "$RC/b+tree" ./b+tree.out file "$R/data/b+tree/mil.txt" command "$R/data/b+tree/command.txt"
# nn (nearest neighbor)
rod nn "$RC/nn" "$NVXX -o nn nn_cuda.cu" \
  && add rod_nn "$RC/nn" ./nn filelist_4 -r 5 -lat 30 -lng 90

# ── 4 · CUDA samples (small diverse subset) ─────────────────────────────────
if ! have "$W/cuda-samples"; then
    git clone -q --depth 1 https://github.com/NVIDIA/cuda-samples "$W/cuda-samples"
fi
CS="$W/cuda-samples/Samples"
for pair in \
    "0_Introduction/matrixMul" \
    "6_Performance/transpose" \
    "2_Concepts_and_Techniques/reduction" \
    "2_Concepts_and_Techniques/histogram" \
    "5_Domain_Specific/BlackScholes"; do
    name=$(basename "$pair")
    d="$CS/$pair"
    [ -d "$d" ] || continue
    if ! have "$d/build/$name"; then
        ( cd "$d" && cmake -Bbuild -DCMAKE_CUDA_ARCHITECTURES=89 \
              -DCMAKE_CUDA_HOST_COMPILER=$CCBIN >/dev/null 2>&1 &&
          cmake --build build -j >/dev/null 2>&1 )
    fi
    have "$d/build/$name" && add "cs_$(echo "$name" | tr 'A-Z' 'a-z')" "$d/build" "./$name" \
        || note "[FAIL] sample $name"
done

echo
echo "== population manifest =="
column -t -s$'\t' "$MAN" | cut -c1-110
echo "apps: $(wc -l < "$MAN")"
