#!/usr/bin/env python3
"""energy.py — whole-run GPU energy from NVML power sampling (stdlib only).

Answers the one measurement the publishability register flags as open
(PUB open-6 / THESIS G2: "PiM's headline win is joules; you never measure one").
Samples instantaneous board power via `nvidia-smi --query-gpu=power.draw` at a
fixed rate in a background thread for the lifetime of a run, then integrates
power over wall-time → whole-run joules plus mean/peak watts.

This is WHOLE-RUN energy (the cheap, real number available today); per-kernel
energy needs AccelWattch and stays Phase-4. Degrades to a null result where
power.draw is unavailable (some Jetson parts, no driver) — it NEVER raises, so
wrapping a run in it can't fail the run.

Use as a context manager around any subprocess:

    from profiling.env.energy import EnergySampler
    with EnergySampler() as e:
        subprocess.run(workload, ...)
    meta["energy"] = e.result()      # {joules, mean_w, peak_w, samples, seconds} | null-ish
"""
from __future__ import annotations

import subprocess
import threading
import time


class EnergySampler:
    def __init__(self, hz: float = 10.0, gpu_index: int = 0):
        self.dt = 1.0 / max(hz, 1.0)
        self.gpu = gpu_index
        self._samples: list[tuple[float, float]] = []   # (t, watts)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = self._probe()

    def _probe(self) -> bool:
        w = self._read()
        return w is not None

    def _read(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu}",
                 "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            v = out.stdout.strip().splitlines()[0].strip()
            return float(v)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _loop(self):
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self._samples.append((time.time(), w))
            self._stop.wait(self.dt)

    def __enter__(self):
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def result(self):
        """Integrate power over wall-time (trapezoid). None if no samples."""
        s = self._samples
        if len(s) < 2:
            return {"available": False, "reason": "power.draw unavailable on this GPU/driver"}
        joules = 0.0
        for (t0, w0), (t1, w1) in zip(s, s[1:]):
            joules += 0.5 * (w0 + w1) * (t1 - t0)      # W·s = J
        watts = [w for _t, w in s]
        return {
            "available": True,
            "joules": round(joules, 2),
            "mean_w": round(sum(watts) / len(watts), 2),
            "peak_w": round(max(watts), 2),
            "samples": len(s),
            "seconds": round(s[-1][0] - s[0][0], 2),
        }


if __name__ == "__main__":   # tiny self-test: sample this machine idle for ~1 s
    import json
    with EnergySampler() as e:
        time.sleep(1.0)
    print(json.dumps(e.result(), indent=1))
