#!/usr/bin/env python3
"""gpu_warmup.py — bring the GPU to its sustained clock state before a capture.

On hosts that cannot lock clocks (laptops), capture-to-capture variance is
dominated by the clock state the run STARTS in: we measured back-to-back
identical cuVSLAM runs starting at 300/405 MHz (deep idle) vs 1035/3500 MHz
(active idle) differing 3.4× in GPU time, with memory-bound kernels shifting
even in relative share (memory and core clocks scale differently under DVFS).

Running a fixed light load until clocks stabilize — then starting the capture
immediately — puts every run at the same sustained operating point. This is
the unlocked-clock substitute for `nvidia-smi -lgc`; the real fix (locked
clocks) applies on the workstation per hw/*.toml [run].

Desktop-safe like measure_ceilings.py: queue depth 1 (sync per op), 32 MB
buffer, and a hard wall-clock cap.

Usage:  python3 profiling/env/gpu_warmup.py [--seconds 8]
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import time

D2D = 3


def clocks():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.memory",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception:
        return "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    rt = ctypes.CDLL("libcudart.so")
    rt.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    n = 32 * 1024 * 1024
    a, b = ctypes.c_void_p(), ctypes.c_void_p()
    if rt.cudaMalloc(ctypes.byref(a), n) or rt.cudaMalloc(ctypes.byref(b), n):
        raise SystemExit("cudaMalloc failed")
    print(f"[warmup] start clocks: {clocks()}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        rt.cudaMemcpy(b, a, n, D2D)
        rt.cudaDeviceSynchronize()          # queue depth 1 — desktop-safe
    rt.cudaFree(a)
    rt.cudaFree(b)
    print(f"[warmup] end clocks:   {clocks()}  ({args.seconds:.0f}s)")


if __name__ == "__main__":
    main()
