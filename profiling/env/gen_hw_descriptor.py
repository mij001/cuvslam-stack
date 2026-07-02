#!/usr/bin/env python3
"""gen_hw_descriptor.py — auto-generate a profiling/hw/*.toml for THIS machine.

Makes the harness self-tuning on any host: structural values (SM count, L2 size,
bus width, compute capability, VRAM, ECC) are read exactly from the CUDA driver
via ctypes on libcudart (cudaDeviceGetAttribute — stable C ABI, no pip deps);
name/driver/max clocks come from nvidia-smi. The two roofline CEILINGS
(DRAM GB/s, FP32 TFLOP/s) are derived estimates and are flagged `verify`: run the
Empirical Roofline Toolkit before using them in published numbers.

Usage:
  python3 profiling/env/gen_hw_descriptor.py            # writes profiling/hw/<host>_sm<cc>.toml
  python3 profiling/env/gen_hw_descriptor.py --stdout   # print instead of write
  python3 profiling/env/gen_hw_descriptor.py --device 1 # non-default GPU
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HW_DIR = os.path.abspath(os.path.join(HERE, "..", "hw"))

# cudaDeviceGetAttribute enum values (stable across CUDA versions)
ATTR = {
    "clock_khz": 13,
    "sms": 16,
    "ecc": 32,
    "mem_clock_khz": 36,
    "bus_bits": 37,
    "l2_bytes": 38,
    "cc_major": 75,
    "cc_minor": 76,
    "smem_per_sm": 81,
}

# FP32 CUDA cores per SM by compute capability (NVIDIA architecture whitepapers)
CORES_PER_SM = {(6, 0): 64, (6, 1): 128, (7, 0): 64, (7, 5): 64,
                (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
                (9, 0): 128, (10, 0): 128, (12, 0): 128}

ARCH = {6: "Pascal", 7: "Volta/Turing", 8: "Ampere/Ada", 9: "Hopper",
        10: "Blackwell", 12: "Blackwell"}

# Accel-Sim stock configs by CC (Slice-3 hint)
ACCELSIM = {(7, 0): "SM7_QV100", (7, 5): "SM75_RTX2060",
            (8, 6): "SM86_RTX3070", (8, 9): "SM75_RTX2060  # adapt: no stock Ada cfg",
            (8, 7): "SM86_RTX3070  # adapt for Orin"}


def load_cudart():
    names = ["libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"]
    paths = [p for base in ("/usr/local/cuda/lib64", "/opt/cuda/lib64", "/usr/lib") for p in
             [os.path.join(base, "libcudart.so")]]
    found = ctypes.util.find_library("cudart")
    for cand in ([found] if found else []) + names + paths:
        try:
            return ctypes.CDLL(cand)
        except (OSError, TypeError):
            continue
    raise SystemExit("libcudart not found — install the CUDA toolkit or set LD_LIBRARY_PATH")


def cuda_attrs(device: int) -> dict:
    rt = load_cudart()
    out = {}
    v = ctypes.c_int()
    for key, attr in ATTR.items():
        rc = rt.cudaDeviceGetAttribute(ctypes.byref(v), attr, device)
        out[key] = v.value if rc == 0 else None
        if rc != 0:
            print(f"[!] cudaDeviceGetAttribute({attr}) rc={rc}", file=sys.stderr)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    rt.cudaSetDevice(device)
    if rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0:
        out["vram_bytes"] = total.value
    return out


def smi(query: str, device: int) -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader",
             "-i", str(device)], text=True).strip().splitlines()[0].strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--role", default="auto-generated",
                    help="prototype | production | auto-generated")
    args = ap.parse_args()

    a = cuda_attrs(args.device)
    name = smi("name", args.device) or "unknown GPU"
    driver = smi("driver_version", args.device) or "?"
    max_gclk = smi("clocks.max.graphics", args.device).replace(" MHz", "") or "0"
    max_mclk = smi("clocks.max.memory", args.device).replace(" MHz", "") or "0"

    cc = (a["cc_major"], a["cc_minor"])
    cores_sm = CORES_PER_SM.get(cc, 128)
    cores = (a["sms"] or 0) * cores_sm
    boost_ghz = int(max_gclk or 0) / 1000.0 or (a["clock_khz"] or 0) / 1e6
    fp32_tflops = round(cores * 2 * boost_ghz / 1000.0, 2)
    # DDR factor 2 on the runtime-reported memory clock; GDDR6X/HBM differ — hence `verify`
    dram_gbps = round((a["mem_clock_khz"] or 0) / 1e6 * 2 * (a["bus_bits"] or 0) / 8, 1)

    host = os.uname().nodename.split(".")[0]
    slug = re.sub(r"[^a-z0-9]+", "", host.lower()) or "host"
    fname = f"{slug}_sm{cc[0]}{cc[1]}.toml"

    # laptop GPUs generally cannot lock clocks; heuristic by name
    is_laptop = any(t in name for t in ("MX", "Laptop", "Max-Q"))

    toml = f"""\
# Hardware descriptor — {name}  (auto-generated on host `{host}`)
#
# Structural values are exact (cudaDeviceGetAttribute). The two CEILINGS
# (dram_gbps_theoretical, fp32_tflops_theoretical) are derived estimates:
# VERIFY with the Empirical Roofline Toolkit before publishing numbers.
# Regenerate any time with: python3 profiling/env/gen_hw_descriptor.py

[device]
name         = "{name}"
arch         = "{ARCH.get(cc[0], '?')}"
compute_cap  = "{cc[0]}.{cc[1]}"          # sm_{cc[0]}{cc[1]}
sms          = {a['sms']}
cuda_cores   = {cores}            # {a['sms']} SM x {cores_sm} FP32 lanes (sm_{cc[0]}{cc[1]})
role         = "{args.role}"

[memory]
l2_bytes              = {a['l2_bytes']}
l1_shared_per_sm_bytes = {a['smem_per_sm']}
vram_bytes            = {a.get('vram_bytes', 0)}
dram_bus_bits         = {a['bus_bits']}
dram_gbps_theoretical = {dram_gbps}       # verify: derived from runtime mem clock x2 DDR
dram_gbps_measured    = 0.0        # TODO: fill from ERT
ecc                   = {'true' if a['ecc'] else 'false'}

[compute]
fp32_tflops_theoretical = {fp32_tflops}      # verify: {cores} lanes x 2 (FMA) x {boost_ghz:.2f} GHz
clock_max_graphics_mhz  = {max_gclk or 0}
clock_max_memory_mhz    = {max_mclk or 0}

[roofline]
l1_fits_below_bytes  = {a['smem_per_sm']}
l2_fits_below_bytes  = {a['l2_bytes']}
dram_above_bytes     = {a['l2_bytes']}

[run]
enforce_clock_locks = {'false' if is_laptop else 'true'}{'' if is_laptop else chr(10) + 'graphics_clock_mhz  = ' + str(max_gclk or 0) + chr(10) + 'memory_clock_mhz    = ' + str(max_mclk or 0)}
persistence_mode    = true
cpu_governor        = "performance"

[accelsim]
base_config = "{ACCELSIM.get(cc, 'NONE  # no stock config for sm_' + str(cc[0]) + str(cc[1]))}"

[notes]
text = \"\"\"
Auto-generated (driver {driver}). NVBit requires CUDA driver <= 575.xx —
check `nvidia-smi` before assuming the Slice-3 data-movement track runs here.
\"\"\"
"""
    if args.stdout:
        print(toml)
    else:
        os.makedirs(HW_DIR, exist_ok=True)
        path = os.path.join(HW_DIR, fname)
        open(path, "w").write(toml)
        print(f"[✓] {path}")
        print("    ceilings are estimates — verify with ERT before publishing")


if __name__ == "__main__":
    main()
