"""common.py — shared loaders for the profiling analysis layer.

Stdlib only, headless by construction. Everything reads the *derived* CSVs that
profiling/harness/profile.py writes into results/<run>/derived/, plus the
profiling/hw/*.toml descriptors. Analysis never re-opens the raw .ncu-rep /
.nsys-rep (those need the vendor tools); it consumes the exported tables, so it
runs on any machine — even one with no GPU.

Unit handling follows the Cao23 lesson: ncu's `--csv --page raw` emits a second
header row of units (us, Kbyte, Mbyte, ...); values are normalized here to SI
base units (seconds, bytes) once, so downstream modules never see a unit string.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field

# ── unit normalization ───────────────────────────────────────────────────────

_TIME = {"nsecond": 1e-9, "ns": 1e-9, "usecond": 1e-6, "us": 1e-6,
         "msecond": 1e-3, "ms": 1e-3, "second": 1.0, "s": 1.0}
_BYTE = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}
_RATE = {"byte/second": 1.0, "Kbyte/second": 1e3, "Mbyte/second": 1e6,
         "Gbyte/second": 1e9, "cycle/nsecond": 1e9, "cycle/usecond": 1e6,
         "cycle/second": 1.0, "inst/cycle": 1.0}


def to_si(value: str, unit: str) -> float | str:
    """Normalize an ncu CSV cell to SI base units. Non-numeric cells pass through."""
    v = value.replace(",", "").strip()
    if v in ("", "N/A", "n/a"):
        return float("nan")
    try:
        num = float(v)
    except ValueError:
        return value  # names, device strings, ...
    unit = (unit or "").strip()
    for table in (_TIME, _BYTE, _RATE):
        if unit in table:
            return num * table[unit]
    return num  # %, ratio, register counts, dimensionless


# ── kernel-name handling ─────────────────────────────────────────────────────

def base_kernel_name(demangled: str) -> str:
    """Collapse a demangled kernel signature to a short stable identifier.

    'cuvslam::cuda::sba::build_full_system_1_kernel(...)' -> 'sba::build_full_system_1_kernel'
    'void cub::...::DeviceMergeSortMergeKernel<...>(...)' -> 'cub::DeviceMergeSortMergeKernel'
    """
    name = demangled.strip().strip('"')
    name = re.sub(r"^void\s+", "", name)
    # cuSOLVER launches everything through 'void kernel<params_struct<...>>(...)';
    # the informative part is the params struct name, not the launcher.
    m = re.match(r"kernel<([A-Za-z0-9_]+)", name)
    if m:
        return m.group(1).rstrip("_")
    # cut the argument list of the OUTER function: find the '(' at template depth 0
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            name = name[:i]
            break
    name = re.sub(r"<.*", "", name)          # strip template args
    name = name.replace("cuvslam::cuda::", "")
    name = re.sub(r"cub::CUB_\w+::(detail::)?(merge_sort::)?", "cub::", name)
    name = name.replace("cub::detail::", "cub::").replace("cub::merge_sort::", "cub::")
    return name.strip()


# ── ncu derived CSV ──────────────────────────────────────────────────────────

@dataclass
class NcuLaunch:
    launch_id: int
    kernel: str            # base name
    kernel_full: str       # demangled signature
    grid: str
    block: str
    metrics: dict = field(default_factory=dict)   # metric -> SI float

    def m(self, name: str, default: float = float("nan")) -> float:
        v = self.metrics.get(name, default)
        return v if isinstance(v, float) else default


def load_ncu_csv(path: str) -> list[NcuLaunch]:
    """Parse profile.py's derived/ncu_metrics.csv (ncu --csv --page raw)."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 3:
        return []
    names, units = rows[0], rows[1]
    fixed = {n: i for i, n in enumerate(names)}
    launches = []
    for row in rows[2:]:
        if not row or len(row) != len(names):
            continue
        full = row[fixed["Kernel Name"]]
        lk = NcuLaunch(
            launch_id=int(row[fixed["ID"]]),
            kernel=base_kernel_name(full),
            kernel_full=full,
            grid=row[fixed.get("Grid Size", 0)],
            block=row[fixed.get("Block Size", 0)],
        )
        for i, (n, u) in enumerate(zip(names, units)):
            if "__" not in n:          # only hardware metrics, skip launch info cols
                continue
            v = to_si(row[i], u)
            if isinstance(v, float):
                lk.metrics[n] = v
        launches.append(lk)
    return launches


# ── nsys kernel-summary CSV ──────────────────────────────────────────────────

@dataclass
class KernelSummary:
    kernel: str
    kernel_full: str
    pct_time: float
    total_ns: float
    instances: int
    avg_ns: float


def load_kern_sum(path: str) -> list[KernelSummary]:
    """Parse derived/kern_sum_cuda_gpu_kern_sum.csv from `nsys stats`."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            full = r.get("Name", "")
            if not full:
                continue
            out.append(KernelSummary(
                kernel=base_kernel_name(full),
                kernel_full=full,
                pct_time=float(r["Time (%)"]),
                total_ns=float(r["Total Time (ns)"]),
                instances=int(r["Instances"]),
                avg_ns=float(r["Avg (ns)"]),
            ))
    return out


# ── hardware descriptor (light TOML: sections + scalar keys; skips ML strings) ─

def load_hw(path: str) -> dict:
    hw: dict = {}
    section = None
    in_multiline = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if in_multiline:
                if '"""' in s:
                    in_multiline = False
                continue
            if not s or s.startswith("#"):
                continue
            m = re.match(r"\[([A-Za-z0-9_.]+)\]$", s)
            if m:
                section = m.group(1)
                hw.setdefault(section, {})
                continue
            m = re.match(r"([A-Za-z0-9_]+)\s*=\s*(.+?)\s*(?:#.*)?$", s)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            if raw.startswith('"""'):
                if raw.count('"""') < 2:
                    in_multiline = True
                continue
            if raw.startswith('"') or raw.startswith("'"):
                val = raw.strip('"').strip("'")
            elif raw in ("true", "false"):
                val = raw == "true"
            else:
                try:
                    val = float(raw) if ("." in raw or "e" in raw.lower()) else int(raw)
                except ValueError:
                    val = raw
            (hw[section] if section else hw)[key] = val
    return hw


# ── results-dir helpers ──────────────────────────────────────────────────────

def load_metadata(run_dir: str) -> dict:
    p = os.path.join(run_dir, "metadata.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def find_derived(run_dir: str, suffix: str) -> str | None:
    d = os.path.join(run_dir, "derived")
    if not os.path.isdir(d):
        return None
    for f in sorted(os.listdir(d)):
        if f.endswith(suffix):
            return os.path.join(d, f)
    return None


def newest_run(results_root: str, profiler: str, contains: str = "") -> str | None:
    """Newest results dir for a profiler ('ncu'/'nsys'), optionally name-filtered."""
    if not os.path.isdir(results_root):
        return None
    cands = [d for d in sorted(os.listdir(results_root))
             if f"_{profiler}_" in d and contains in d
             and os.path.isdir(os.path.join(results_root, d))]
    return os.path.join(results_root, cands[-1]) if cands else None


# ── tiny markdown helpers ────────────────────────────────────────────────────

def md_table(headers: list[str], rows: list[list]) -> str:
    def fmt(x):
        if isinstance(x, float):
            if x != x:  # nan
                return "—"
            if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
                return f"{x:.3g}"
            return f"{x:.2f}".rstrip("0").rstrip(".")
        return str(x)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_csv(path: str, headers: list[str], rows: list[list]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)
