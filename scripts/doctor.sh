#!/usr/bin/env bash
# doctor.sh — is this machine ready to run + profile GPU workloads?
#
# Checks every layer the harness depends on and prints PASS/WARN/FAIL with a
# concrete fix for each finding. The hints encode real failures this stack has
# hit and solved (driver/CUDA/profiler/kernel incompatibilities), so the error
# you would have spent an evening on becomes a one-line instruction.
#
# Run locally:            scripts/doctor.sh
# Run on a profiling target (from the controller):
#                         ssh <target> 'cd <repo> && scripts/doctor.sh'
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

P() { printf 'PASS  %-22s %s\n' "$1" "$2"; }
W() { printf 'WARN  %-22s %s\n      fix: %s\n' "$1" "$2" "$3"; }
F() { printf 'FAIL  %-22s %s\n      fix: %s\n' "$1" "$2" "$3"; FAILS=$((FAILS+1)); }
FAILS=0

echo "== cuvslam-stack doctor — $(hostname) ($(uname -m), kernel $(uname -r)) =="

# ── GPU driver ────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    F driver "nvidia-smi not found" \
      "install the NVIDIA driver (Jetson: part of L4T/JetPack). Nothing GPU works without it."
else
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$DRV" ]; then
        F driver "nvidia-smi present but not answering" \
          "driver/kernel-module mismatch — reboot after driver updates; check 'dmesg | grep -i nvidia' (a kernel upgrade without dkms rebuild causes exactly this)."
    else
        P driver "$GPU, driver $DRV"
    fi
fi
DRV_MAJOR=${DRV%%.*}; DRV_MAJOR=${DRV_MAJOR:-0}

# ── CUDA toolkit ──────────────────────────────────────────────────────────────
if command -v nvcc >/dev/null 2>&1; then
    CUDA=$(nvcc --version | grep -oE "release [0-9.]+" | cut -d" " -f2)
    P cuda "toolkit $CUDA ($(command -v nvcc))"
else
    W cuda "nvcc not on PATH" \
      "profiling analyses don't need it, but NVBit tool builds do: export PATH=/opt/cuda/bin:\$PATH (or install cuda). On this stack, mem_trace builds inside the podman image 'cuvslam-wheel-builder' because a host gcc newer than the CUDA release breaks nvcc."
fi

# ── runner venv + cuvslam import ─────────────────────────────────────────────
VPY=./cuvslam_venv/bin/python
if [ ! -x "$VPY" ]; then
    F venv "cuvslam_venv missing" "./setup_env.sh (builds the venv and installs the wheel from cuvslam_src/dist — 'make wheel' first if no wheel exists)."
else
    IMP=$("$VPY" -c "import cuvslam" 2>&1 >/dev/null)
    if [ -z "$IMP" ]; then
        P cuvslam "wheel imports ($($VPY -c 'import cuvslam;print(getattr(cuvslam,"__version__","?"))' 2>/dev/null))"
    elif echo "$IMP" | grep -q "invalid ELF header"; then
        F cuvslam "corrupt CUDA runtime library ($(echo "$IMP" | grep -oE '[^ ]*\.so[^ :]*' | head -1))" \
          "a system CUDA lib is damaged. Quick unblock: export LD_LIBRARY_PATH=\$HOME/.local/cuda-repair/lib (a known-good copy); real fix: reinstall the cuda package (e.g. 'sudo pacman -S cuda')."
    elif echo "$IMP" | grep -qE "libcu[a-z]+\.so"; then
        F cuvslam "CUDA runtime mismatch: $(echo "$IMP" | tail -1 | cut -c1-90)" \
          "the wheel's CUDA major must match the installed runtime: a cu12 wheel needs CUDA 12 libs, cu13 needs 13. Rebuild the wheel for this host ('make wheel') or install the matching cuda runtime."
    else
        F cuvslam "import fails: $(echo "$IMP" | tail -1 | cut -c1-90)" \
          "./setup_env.sh with a wheel built for this CUDA/driver (see Makefile: make wheel)."
    fi
fi

# ── Nsight Systems ────────────────────────────────────────────────────────────
if command -v nsys >/dev/null 2>&1; then
    P nsys "$(nsys --version 2>/dev/null | head -1)"
else
    F nsys "nsys not found" "install nsight-systems (pacman/apt/JetPack). The timeline+DAG analyses need it."
fi

# ── Nsight Compute vs driver (the classic trap) ─────────────────────────────
NCU_EFF="${NCU_BIN:-ncu}"
if ! command -v "$NCU_EFF" >/dev/null 2>&1 && [ ! -x "$NCU_EFF" ]; then
    F ncu "ncu not found" "install nsight-compute, or point NCU_BIN at one (this stack keeps a CUDA-12.9 ncu 2025.2 at ~/ncu2025/.../ncu for driver-575 hosts)."
else
    NCUV=$("$NCU_EFF" --version 2>/dev/null | grep -oE "Version [0-9.]+" | head -1)
    # ncu 2025.3+/2026.x are CUDA-13 builds and REFUSE drivers < 580; a
    # downgraded 575 host (required for NVBit) must use ncu <= 2025.2.
    NCU_MINOR=$(echo "$NCUV" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [ "$DRV_MAJOR" -le 575 ] 2>/dev/null && \
       { [ "${NCU_MINOR%%.*}" -ge 2026 ] 2>/dev/null || [ "$NCU_MINOR" = "2025.3" ]; }; then
        F ncu "$NCUV rejects driver $DRV" \
          "use a CUDA-12.x ncu: export NCU_BIN=~/ncu2025/nsight_compute-linux-x86_64-2025.2.1.3-archive/ncu (NVIDIA public redist). The harness honours NCU_BIN everywhere."
    else
        P ncu "$NCUV (NCU_BIN=${NCU_BIN:-system})"
    fi
    # GPU perf counter permission (ERR_NVGPUCTRPERM)
    if [ -f /proc/driver/nvidia/params ] && grep -q "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params 2>/dev/null; then
        W ncu-perms "GPU perf counters restricted to root" \
          "echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf && reboot — or run ncu with sudo."
    fi
fi

# ── NVBit (deep memory traces) ───────────────────────────────────────────────
TOOL=external_repos/nvbit_release_x86_64/tools/mem_trace/mem_trace.so
if [ "$DRV_MAJOR" -gt 575 ] 2>/dev/null; then
    W nvbit "driver $DRV > 575 — NVBit cannot attach" \
      "NVBit (all releases through 1.8) caps at driver 575. Either skip the nvbit legs on this target, or downgrade: driver 575.64 + matching CUDA 12.9 + an LTS kernel with dkms (documented in PROJECT_STATUS 2026-07-03)."
elif [ ! -f "$TOOL" ]; then
    W nvbit "mem_trace.so not built" \
      "build it in the podman image (host gcc is usually too new for nvcc): see profiling/blocked/README + patches; then re-run doctor."
else
    P nvbit "mem_trace.so present (driver $DRV <= 575 OK)"
fi

# ── clocks + data volume ─────────────────────────────────────────────────────
if sudo -n true 2>/dev/null; then
    P sudo "passwordless sudo (clock locking will work)"
else
    W sudo "no passwordless sudo" \
      "clock locking (nvidia-smi -lgc/-lmc) keeps captures reproducible; grant NOPASSWD for nvidia-smi or expect noisier numbers."
fi
DATA="${CUVSLAM_DATA2:-/mnt/data}"
if [ -d "$DATA" ] && [ -n "$(ls "$DATA" 2>/dev/null | head -1)" ]; then
    P datasets "$DATA mounted ($(ls "$DATA" | wc -l) entries)"
else
    W datasets "$DATA missing/empty" \
      "mount the dataset volume or set CUVSLAM_DATA2; fetch benchmarks with scripts/fetch_paper_datasets.sh (see PAPER_DATASETS.md). --check validation works without data."
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "READY — this machine can run the profiling regime."
else
    echo "NOT READY — $FAILS blocking issue(s) above; each has its fix."
fi
exit "$FAILS"
