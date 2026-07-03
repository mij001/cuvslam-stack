#!/usr/bin/env python3
"""measure_ceilings.py — MEASURED roofline ceilings, no compiler required.

The roofline/classification ceilings must not be marketing numbers: reviewers
(rightly) reject rooflines drawn against unverified peaks. This tool measures
both ceilings with stdlib ctypes only — no nvcc, no pip — so it runs on any
host with a CUDA driver:

  * DRAM bandwidth  — device-to-device cudaMemcpy sweep (read+write counted).
    Standard achievable-BW methodology; expect 80–95% of the theoretical bus
    rate. This is the honest ceiling for a DRAM-bound roofline.
  * FP32 compute    — cublasSgemm sweep (2·M·N·K FLOP per call). Dense GEMM is
    the accepted FP32-peak proxy on non-tensor pipelines (≈90% of peak).

Reports the MEDIAN of the trials (plus min/max spread) and samples the
graphics/memory clocks during the run, because laptops DVFS: a ceiling without
its clock context is not reproducible.

DESKTOP-SAFETY DESIGN (learned the hard way — an earlier draft queued dozens of
unsynchronized 256 MB copies and GEMMs, starving the GPU scheduler long enough
to reset the driver and kill the desktop session):
  * queue depth 1 — every GPU op is followed by cudaDeviceSynchronize, so the
    GPU is never blocked for more than one ~2 ms operation;
  * VRAM budget from cudaMemGetInfo — buffers ≤ min(64 MB, 10% of free);
  * time-boxed phases (~2.5 s each) with early stop;
  * 50 ms host sleep between trials so compositors/other clients get slots;
  * refuses to run if the GPU drives an active display (override: --force).

Per-op sync costs ≲2% of measured throughput (ops are ≥ ~2 ms); the report is
labeled 'measured (sync-per-op; may underread ≤2%)' accordingly.

Usage:
  python3 profiling/env/measure_ceilings.py                     # print
  python3 profiling/env/measure_ceilings.py --update-hw profiling/hw/<gpu>.toml
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import re
import statistics
import subprocess
import time

CUDA_OK = 0
D2D = 3  # cudaMemcpyDeviceToDevice
PHASE_BUDGET_S = 2.5      # wall-clock cap per measurement phase
TRIAL_SLEEP_S = 0.05      # host yield between trials


def load(names):
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    found = ctypes.util.find_library(names[0].replace("lib", "").split(".")[0])
    if found:
        return ctypes.CDLL(found)
    raise SystemExit(f"cannot load any of {names} (set LD_LIBRARY_PATH)")


def check(rc, what):
    if rc != CUDA_OK:
        raise SystemExit(f"{what} failed (rc={rc})")


def smi(query):
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True).strip().splitlines()[0].strip()
    except Exception:
        return ""


def sample_clocks():
    out = smi("clocks.current.graphics,clocks.current.memory")
    return out or "unknown"


def vram_budget(rt):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    check(rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
    if free.value < 256 * 1024 * 1024:
        raise SystemExit(f"only {free.value//2**20} MiB VRAM free — refusing to "
                         "benchmark on a starved GPU")
    per_buffer = min(64 * 1024 * 1024, int(free.value * 0.10))
    return per_buffer, free.value


def measure_dram_bw(rt, buf_bytes, trials=7):
    """Median achievable D2D bandwidth. Sync after EVERY copy (queue depth 1)."""
    a, b = ctypes.c_void_p(), ctypes.c_void_p()
    check(rt.cudaMalloc(ctypes.byref(a), buf_bytes), "cudaMalloc")
    check(rt.cudaMalloc(ctypes.byref(b), buf_bytes), "cudaMalloc")
    try:
        check(rt.cudaMemcpy(b, a, buf_bytes, D2D), "warmup memcpy")
        check(rt.cudaDeviceSynchronize(), "sync")
        results, phase_start = [], time.monotonic()
        for _ in range(trials):
            if time.monotonic() - phase_start > PHASE_BUDGET_S:
                break
            t0, n = time.monotonic(), 0
            while time.monotonic() - t0 < 0.15:          # ~75 copies max/trial
                check(rt.cudaMemcpy(b, a, buf_bytes, D2D), "memcpy")
                check(rt.cudaDeviceSynchronize(), "sync")  # queue depth 1
                n += 1
            dt = time.monotonic() - t0
            results.append(2.0 * buf_bytes * n / dt / 1e9)  # read + write
            time.sleep(TRIAL_SLEEP_S)
        return results
    finally:
        rt.cudaFree(a)
        rt.cudaFree(b)


def measure_fp32(rt, bl, per_buffer_bytes, trials=7):
    """Best median cublasSgemm FP32 throughput over power-of-two sizes.

    Sizes are restricted to powers of two: odd tiles hit poor cuBLAS kernel
    selections (n=1536 measured 2.5× slower than n=2048 on an MX450) and would
    understate the ceiling. Sync after EVERY gemm (queue depth 1).
    """
    best, best_n = [], 0
    for n in (1024, 2048):
        if 3 * (n * n * 4) > 3 * per_buffer_bytes:
            continue
        res = _fp32_at(rt, bl, n, trials)
        if res and (not best or statistics.median(res) > statistics.median(best)):
            best, best_n = res, n
    return best, best_n


def _fp32_at(rt, bl, n, trials):
    nb = n * n * 4
    A, B, C = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    for buf in (A, B, C):
        check(rt.cudaMalloc(ctypes.byref(buf), nb), "cudaMalloc")
    handle = ctypes.c_void_p()
    check(bl.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate")
    try:
        one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
        N = ctypes.c_int(n)
        args = (handle, 0, 0, N, N, N, ctypes.byref(one), A, N, B, N,
                ctypes.byref(zero), C, N)
        check(bl.cublasSgemm_v2(*args), "warmup gemm")
        check(rt.cudaDeviceSynchronize(), "sync")
        flop = 2.0 * n * n * n
        results, phase_start = [], time.monotonic()
        for _ in range(trials):
            if time.monotonic() - phase_start > PHASE_BUDGET_S:
                break
            t0, k = time.monotonic(), 0
            while time.monotonic() - t0 < 0.15:
                check(bl.cublasSgemm_v2(*args), "gemm")
                check(rt.cudaDeviceSynchronize(), "sync")  # queue depth 1
                k += 1
            dt = time.monotonic() - t0
            results.append(flop * k / dt / 1e9)            # GFLOP/s
            time.sleep(TRIAL_SLEEP_S)
        return results
    finally:
        bl.cublasDestroy_v2(handle)
        for buf in (A, B, C):
            rt.cudaFree(buf)


def update_hw(path, bw_gbps, fp32_gflops, clocks):
    text = open(path).read()
    text = re.sub(r"(?m)^(dram_gbps_measured\s*=\s*).*$",
                  rf"\g<1>{bw_gbps:.1f}      # measured: D2D memcpy median, "
                  rf"sync-per-op (may underread ≤2%); clocks {clocks}", text)
    if "fp32_gflops_measured" not in text:
        text = re.sub(r"(?m)^(fp32_tflops_theoretical.*)$",
                      rf"\g<1>\nfp32_gflops_measured    = {fp32_gflops:.0f}      "
                      rf"# measured: cublasSgemm median; clocks {clocks}", text)
    else:
        text = re.sub(r"(?m)^(fp32_gflops_measured\s*=\s*).*$",
                      rf"\g<1>{fp32_gflops:.0f}      # measured: cublasSgemm median; "
                      rf"clocks {clocks}", text)
    open(path, "w").write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-hw", default=None, metavar="HW_TOML")
    ap.add_argument("--force", action="store_true",
                    help="run even if this GPU drives an active display")
    args = ap.parse_args()

    if smi("display_active").startswith("Enabled") and not args.force:
        raise SystemExit("this GPU drives an active display — a saturation "
                         "benchmark will contend with the compositor. "
                         "Re-run with --force if you accept that.")

    rt = load(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])
    rt.cudaMalloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    bl = load(["libcublas.so", "libcublas.so.13", "libcublas.so.12"])

    per_buffer, free = vram_budget(rt)
    print(f"[cfg] VRAM free {free//2**20} MiB → buffers {per_buffer//2**20} MiB, "
          f"queue depth 1, phase budget {PHASE_BUDGET_S}s/phase")
    clocks_before = sample_clocks()

    bw = measure_dram_bw(rt, per_buffer)
    clocks_mid = sample_clocks()
    fp, gemm_n = measure_fp32(rt, bl, per_buffer)
    clocks_after = sample_clocks()

    if not bw or not fp:
        raise SystemExit("phase produced no samples — aborting without results")
    bw_med, fp_med = statistics.median(bw), statistics.median(fp)
    print(f"[✓] DRAM D2D bandwidth: {bw_med:.1f} GB/s  "
          f"(min {min(bw):.1f} / max {max(bw):.1f}, {len(bw)} trials)")
    print(f"[✓] FP32 cublasSgemm (n={gemm_n}): {fp_med:.0f} GFLOP/s  "
          f"(min {min(fp):.0f} / max {max(fp):.0f}, {len(fp)} trials)")
    print(f"[i] clocks (graphics,memory) before/mid/after: "
          f"{clocks_before} | {clocks_mid} | {clocks_after}")

    if args.update_hw:
        update_hw(args.update_hw, bw_med, fp_med, clocks_mid)
        print(f"[✓] updated {args.update_hw}")


if __name__ == "__main__":
    main()
