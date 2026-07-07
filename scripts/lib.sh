#!/usr/bin/env bash
# scripts/lib.sh — shared helpers for the campaign/driver scripts.
# Source it (do not execute):   . "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
#
# Provides:
#   STACK_ROOT             repo root (scripts/..), already cd'ed into
#   log_init FILE          start a fresh tee'd log; log MSG appends+echoes
#   ensure_data_rw DIR     remount the NTFS data volume rw,uid=1000 if needed
#   free_gpu / restore_gui best-effort compositor teardown/restore (ws helpers)
#   lock_gpu_clocks [GFX] [MEM]   persistence + locked clocks (default 1620/7001)
#   ape_of EVAL            RMSE-APE metres out of an eval.txt ('' if absent)
#   matched_of EVAL        matched-pose count out of an eval.txt
#
# Everything is idempotent and safe on hosts without sudo -n / nvidia-smi:
# helpers degrade to no-ops rather than aborting an otherwise-valid run.

STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STACK_ROOT"

_LIB_LOG=""
log_init()   { _LIB_LOG="$1"; : > "$_LIB_LOG"; }   # fresh log (tail -f friendly)
log_attach() { _LIB_LOG="$1"; }                     # append (resumable campaigns)
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${_LIB_LOG:-/dev/null}"; }

# Remount the dataset volume read-write with uid=1000 so writes into
# pre-existing (root-owned) output dirs succeed. No-op when already writable.
ensure_data_rw() {
    local dir="${1:-/mnt/data}" dev="${2:-/dev/sda2}"
    if ! touch "$dir/.w" 2>/dev/null; then
        sudo -n umount "$dir" 2>/dev/null
        sudo -n mount -t ntfs3 -o rw,force,uid=1000,gid=1000,umask=022 "$dev" "$dir" \
            || { log "FATAL: $dir not writable"; return 1; }
    fi
    rm -f "$dir/.w" 2>/dev/null
    mkdir -p "$dir"
}

free_gpu()    { [ -f ~/free_gpu.zsh ]    && zsh ~/free_gpu.zsh    >/dev/null 2>&1 || true; }
restore_gui() { [ -f ~/restore_gui.zsh ] && zsh ~/restore_gui.zsh >/dev/null 2>&1 || true; }

# Lock GPU clocks for reproducible captures (KDE compositor perturbs them, so
# call free_gpu first on the workstation). Retries until the lock sticks.
lock_gpu_clocks() {
    local gfx="${1:-1620}" mem="${2:-7001}" i
    command -v nvidia-smi >/dev/null 2>&1 || { log "lock_gpu_clocks: no nvidia-smi (skipped)"; return 0; }
    for i in 1 2 3; do
        sudo -n nvidia-smi -pm 1 >/dev/null 2>&1
        sudo -n nvidia-smi -lgc "$gfx,$gfx" >/dev/null 2>&1
        sudo -n nvidia-smi -lmc "$mem,$mem" >/dev/null 2>&1
        sleep 2
        [ "$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | head -1)" = "$gfx" ] && break
    done
    log "clocks: $(nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory --format=csv,noheader | head -1)"
}

ape_of()     { grep -oE "RMSE APE[^(]*\(([0-9.]+) m\)" "$1" 2>/dev/null | grep -oE "[0-9.]+ m" | head -1 | grep -oE "[0-9.]+"; }
matched_of() { grep -oE "matched poses[^0-9]*([0-9]+)" "$1" 2>/dev/null | grep -oE "[0-9]+" | head -1; }
