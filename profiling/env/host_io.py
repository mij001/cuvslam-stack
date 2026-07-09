#!/usr/bin/env python3
"""host_io.py — host-side I/O + memory of a run, sampled from /proc (stdlib only).

Closes THESIS gap G3 ("we characterize GPU memory but never measured a byte of
host I/O"). Everything else in the stack is GPU-side; this adds the host
dimension that the near-sensor / ISP argument ultimately rests on:

  storage_read_bytes    /proc/<pid>/io read_bytes  — bytes actually fetched from
                        the block device (the dataset/sensor read that FEEDS the
                        H2D upload; misses page-cache hits, which is correct)
  storage_write_bytes   /proc/<pid>/io write_bytes — bytes sent to the block dev
                        (e.g. an LMDB map flush)
  syscall_read/write    rchar/wchar — bytes through read()/write() incl. cache
  major_faults          /proc/<pid>/stat majflt — MMAP page-ins from disk (the
                        signal /proc-io misses for an mmap-backed LMDB)
  peak_host_rss_mb      max VmHWM across the tree — peak resident host memory
                        (the session-scale keyframe DB lives here, not on the GPU)

Counters are cumulative per process and processes die, so it samples the whole
process TREE periodically and keeps each pid's last (max) reading, summing at
the end. Never raises — degrades to available:false where /proc is unreadable.

    from profiling.env.host_io import HostIOSampler
    with HostIOSampler(root_pid) as h:
        subprocess.run(workload, ...)
    meta["host_io"] = h.result()
"""
from __future__ import annotations

import os
import threading
import time

_PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _children(pid, out):
    """Recursively collect pid + descendants by scanning /proc/*/stat ppid."""
    out.add(pid)
    try:
        kids = []
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    fields = fh.read().rsplit(")", 1)[1].split()
                if int(fields[1]) == pid:      # ppid is field 4 (0-idx 1 after ')')
                    kids.append(int(d))
            except (OSError, ValueError, IndexError):
                continue
        for k in kids:
            if k not in out:
                _children(k, out)
    except OSError:
        pass


def _read_proc(pid):
    """(io dict, majflt, vmhwm_kb) for one pid, best-effort."""
    io = {}
    try:
        for line in open(f"/proc/{pid}/io"):
            k, _, v = line.partition(":")
            io[k.strip()] = int(v.strip())
    except (OSError, ValueError):
        pass
    majflt = 0
    try:
        fields = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        majflt = int(fields[9])                # majflt is stat field 12 (0-idx 9 here)
    except (OSError, ValueError, IndexError):
        pass
    vmhwm = 0
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                vmhwm = int(line.split()[1])   # kB
                break
    except (OSError, ValueError, IndexError):
        pass
    return io, majflt, vmhwm


class HostIOSampler:
    def __init__(self, root_pid=None, hz=4.0):
        self.root = root_pid
        self.dt = 1.0 / max(hz, 1.0)
        self._last: dict[int, dict] = {}     # pid -> {read_bytes, write_bytes, rchar, wchar, majflt, vmhwm}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def set_root(self, pid):
        self.root = pid

    def _sample(self):
        if not self.root:
            return
        pids = set()
        _children(self.root, pids)
        for pid in pids:
            io, majflt, vmhwm = _read_proc(pid)
            self._last[pid] = {
                "read_bytes": io.get("read_bytes", self._last.get(pid, {}).get("read_bytes", 0)),
                "write_bytes": io.get("write_bytes", self._last.get(pid, {}).get("write_bytes", 0)),
                "rchar": io.get("rchar", self._last.get(pid, {}).get("rchar", 0)),
                "wchar": io.get("wchar", self._last.get(pid, {}).get("wchar", 0)),
                "majflt": max(majflt, self._last.get(pid, {}).get("majflt", 0)),
                "vmhwm": max(vmhwm, self._last.get(pid, {}).get("vmhwm", 0)),
            }

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.dt)

    def __enter__(self):
        if os.path.isdir("/proc/self"):
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._sample()               # final reading before children exit
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return False

    def result(self):
        if not self._last:
            return {"available": False, "reason": "/proc not readable"}
        agg = {k: sum(p[k] for p in self._last.values())
               for k in ("read_bytes", "write_bytes", "rchar", "wchar", "majflt")}
        peak_rss = max((p["vmhwm"] for p in self._last.values()), default=0)
        return {
            "available": True,
            "storage_read_mb": round(agg["read_bytes"] / 1e6, 2),
            "storage_write_mb": round(agg["write_bytes"] / 1e6, 2),
            "syscall_read_mb": round(agg["rchar"] / 1e6, 2),
            "syscall_write_mb": round(agg["wchar"] / 1e6, 2),
            "mmap_pagein_mb": round(agg["majflt"] * _PAGE / 1e6, 2),
            "peak_host_rss_mb": round(peak_rss / 1024.0, 2),
            "procs": len(self._last),
        }


if __name__ == "__main__":   # self-test: read a chunk of a file, watch storage I/O
    import json
    import subprocess
    with HostIOSampler() as h:
        p = subprocess.Popen(["bash", "-c", "cat /proc/cpuinfo > /dev/null; sleep 0.5"])
        h.set_root(p.pid)
        p.wait()
    print(json.dumps(h.result(), indent=1))
