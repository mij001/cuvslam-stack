#!/usr/bin/env python3
"""gen_ndp_config.py — groundwork for Phase 3 (Accel-Sim NDP evaluation).

The characterization says WHICH kernels want near-data placement and WHY
(substrate.py). Phase 3 has to show the speedup/energy DELTA on a simulated NDP
GPU. The part of that which needs NO Accel-Sim checkout — and which can be built
and tested now — is turning the verdicts + the device descriptor + the placement
model's parameters into concrete NDP config OVERLAYS. This does exactly that.

For each scenario (conservative / moderate, same k=internal-BW-multiple and
c=PiM-compute-ratio as pim_placement_model.py), it emits:

  <out>/<scenario>.ndp.config     an Accel-Sim gpgpusim-config OVERLAY: the
                                  knobs that change for an NDP-GPU vs the
                                  (gated, calibrated) sm_89 BASE config — a
                                  smaller/bypassed L2 (near-bank data isn't
                                  cached) and DRAM internal bandwidth ×k.
  <out>/<scenario>.manifest.csv   which kernels the NDP models and why: the
                                  substrate verdict, the baseline GPU time, and
                                  whether this scenario offloads it (affinity in
                                  the scenario's allowed set).

These overlay the base config named in the hw descriptor's [accelsim] section;
run_accelsim.sh (Phase 3, gated on the Accel-Sim checkout) applies base+overlay,
runs the NVBit SASS traces, and reports base-vs-NDP deltas. Simulated numbers
are DELTAS vs the measured baseline, never absolutes.

  python3 profiling/sim/gen_ndp_config.py \
      --hw profiling/hw/dellworkstation_sm89.toml \
      --verdicts reports/2026-07-07_substrate/substrate_verdicts.csv \
      --out profiling/sim/configs
"""
from __future__ import annotations

import argparse
import csv
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# same knobs as pim_placement_model.py (kept here so the sim module is
# self-contained — the model file is a local scratch, not committed).
SCENARIOS = {
    "conservative": dict(k=4.0, c=0.50, affinities={"strong"}),
    "moderate":     dict(k=8.0, c=0.75, affinities={"strong", "conditional"}),
}

# substrate verdict -> the PiM affinity it implies (mirrors classify.pim_affinity)
AFFINITY_OF = {
    "PiM-near-bank": "strong", "ISP/near-storage": "strong",
    "near-sensor SRAM (consume before DRAM)": "strong",
    "near-memory compute (latency, uncacheable set)": "strong",
    "PiM-scatter": "conditional", "scatter-capable PiM — or a data-layout fix first": "conditional",
    "GPU+layout-fix": "conditional",
}


def hw_val(text, key, default=0.0):
    m = re.search(rf"(?m)^{key}\s*=\s*([0-9.]+)", text)
    return float(m.group(1)) if m else default


def affinity(substrate):
    for pref, aff in AFFINITY_OF.items():
        if substrate.startswith(pref) or substrate == pref:
            return aff
    return "none"


def emit_accelsim_overlays(base_cfg_path, outdir):
    """Concretize the NDP knobs against a REAL Accel-Sim base gpgpusim.config:
    parse its -gpgpu_clock_domains and -gpgpu_cache:dl2 lines and emit, per
    scenario, an overlay config (later -config wins in accel-sim.out) with

      DRAM clock x k      near-bank internal bandwidth (and ~1/k access time —
                          both are what 'compute at the bank' buys)
      core clock x c_r    the PiM compute ratio
      L2 sets / 8         near-bank data is not L2-cached; shrinking the L2 to
                          a token size models bypass without touching the
                          tag/MSHR machinery

    The baseline for DELTAS is the unmodified base config (standing rule 5:
    simulated numbers are deltas, never absolutes)."""
    text = open(base_cfg_path).read()
    m_clk = re.search(r"(?m)^-gpgpu_clock_domains\s+([\d.:]+)", text)
    m_dl2 = re.search(r"(?m)^-gpgpu_cache:dl2\s+(\S+)", text)
    if not (m_clk and m_dl2):
        raise SystemExit(f"{base_cfg_path}: missing clock_domains or cache:dl2")
    core, icnt, l2c, dram = (float(x) for x in m_clk.group(1).split(":"))
    dl2 = m_dl2.group(1)

    written = []
    for scen, p in SCENARIOS.items():
        # dl2 like S:64:128:16,L:B:m:L:P,... -> divide the sets field by 8
        # (floor 4) AND switch the set-index fn from IPOLY ('P') to linear
        # ('L'): IPOLY asserts on non-{16,32,64} set counts (hashing.cc:88)
        groups = dl2.split(",")
        f0 = groups[0].split(":")
        f0[1] = str(max(int(f0[1]) // 8, 4))
        groups[0] = ":".join(f0)
        if len(groups) > 1 and groups[1].endswith(":P"):
            groups[1] = groups[1][:-2] + ":L"
        dl2_ndp = ",".join(groups)
        parts = f0
        cfg = os.path.join(outdir, f"{scen}.accelsim.config")
        with open(cfg, "w") as fh:
            fh.write(
                f"# Accel-Sim NDP overlay ({scen}) — appended after the base config\n"
                f"# derived from {os.path.basename(base_cfg_path)}: DRAM clk x{p['k']:g}, "
                f"core clk x{p['c']:g}, L2 sets/8.\n"
                f"# ICNT + L2 clocks scale WITH the DRAM (near-bank compute sits on the\n"
                f"# bank fabric — v1 kept them core-side and the first sims showed the\n"
                f"# request path throttling away the entire bandwidth gain: the fabric,\n"
                f"# not the DRAM, was the wall. That v1 result is kept on record.)\n"
                f"-gpgpu_clock_domains {core * p['c']:.1f}:{icnt * p['k']:.1f}:"
                f"{l2c * p['k']:.1f}:{dram * p['k']:.1f}\n"
                f"-gpgpu_cache:dl2 {dl2_ndp}\n")
        written.append(cfg)
        print(f"[{scen}] accel-sim overlay: core {core:g}->{core * p['c']:g} MHz, "
              f"dram {dram:g}->{dram * p['k']:g} MHz, dl2 sets {dl2.split(':')[1]}->{parts[1]}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hw", default="profiling/hw/dellworkstation_sm89.toml")
    ap.add_argument("--verdicts", default="reports/2026-07-07_substrate/substrate_verdicts.csv")
    ap.add_argument("--out", default="profiling/sim/configs")
    ap.add_argument("--accelsim-base", default=None,
                    help="path to a real gpgpusim.config (e.g. tested-cfgs/SM86_RTX3070/"
                         "gpgpusim.config) — also emit concrete Accel-Sim overlays")
    args = ap.parse_args(argv)

    hw = open(os.path.join(REPO, args.hw)).read()
    l2_bytes = int(hw_val(hw, "l2_bytes"))
    dram_gbps = hw_val(hw, "dram_gbps_measured") or hw_val(hw, "dram_gbps_theoretical")
    base_cfg = (re.search(r'base_config\s*=\s*"([^"#]+)', hw) or [None, "SM75_RTX2060"])[1].strip()

    vpath = os.path.join(REPO, args.verdicts)
    verdicts = list(csv.DictReader(open(vpath))) if os.path.isfile(vpath) else []
    # one verdict per kernel (modal across workloads)
    by_kernel = {}
    for r in verdicts:
        by_kernel.setdefault(r["kernel"], []).append(r)

    outdir = os.path.join(REPO, args.out)
    os.makedirs(outdir, exist_ok=True)
    written = []
    for scen, p in SCENARIOS.items():
        # NDP overlay: near-bank data is not cached (shrink L2 to a token size to
        # model bypass without changing the tag/MSHR machinery) and DRAM internal
        # bandwidth is ×k. These are the knobs Phase 3 sweeps; the base config
        # supplies everything else.
        ndp_l2 = max(int(l2_bytes / p["k"]), 32 * 1024)      # /k, floor 32 KiB
        cfg = os.path.join(outdir, f"{scen}.ndp.config")
        with open(cfg, "w") as fh:
            fh.write(
                f"# NDP overlay ({scen}) — OVERLAYS the calibrated base config "
                f"'{base_cfg}'\n"
                f"# derived by gen_ndp_config.py from {os.path.basename(args.hw)} + the "
                f"placement model (k={p['k']:g} internal-BW multiple, c={p['c']:g} PiM "
                f"compute ratio).\n"
                f"# Phase 3 (run_accelsim.sh) applies base+overlay and reports DELTAS.\n\n"
                f"# near-bank data is not L2-cached -> shrink L2 from {l2_bytes} to {ndp_l2} B\n"
                f"-gpgpu_cache:dl2_ndp_bytes {ndp_l2}\n"
                f"# DRAM effective internal bandwidth x{p['k']:g} of {dram_gbps:g} GB/s baseline\n"
                f"-gpgpu_dram_ndp_bw_gbps {dram_gbps * p['k']:g}\n"
                f"# PiM compute throughput = {p['c']:g}x the host-SM lane rate\n"
                f"-gpgpu_ndp_compute_ratio {p['c']:g}\n")
        written.append(cfg)

        # manifest: which kernels this scenario offloads
        man = os.path.join(outdir, f"{scen}.manifest.csv")
        with open(man, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["kernel", "modal_substrate", "affinity", "offloaded_this_scenario",
                        "baseline_time_ms_max"])
            n_off = 0
            for k, rs in sorted(by_kernel.items()):
                subs = [r["substrate"] for r in rs]
                modal = max(set(subs), key=subs.count)
                aff = affinity(modal)
                off = aff in p["affinities"]
                n_off += off
                tmax = max((float(r.get("time_ms") or 0) for r in rs), default=0.0)
                w.writerow([k, modal, aff, "yes" if off else "no", round(tmax, 3)])
        written.append(man)
        print(f"[{scen}] L2 {l2_bytes}->{ndp_l2} B, DRAM x{p['k']:g}, offloads "
              f"{n_off}/{len(by_kernel)} kernels -> {os.path.relpath(cfg, REPO)}")

    if args.accelsim_base:
        written += emit_accelsim_overlays(
            args.accelsim_base if os.path.isabs(args.accelsim_base)
            else os.path.join(REPO, args.accelsim_base), outdir)

    print(f"\n[✓] {len(written)} NDP config/manifest files -> {os.path.relpath(outdir, REPO)}")
    print("    next (Phase 3, gated): run_accelsim.sh applies base+overlay over the "
          "NVBit traces and reports base-vs-NDP deltas.")


if __name__ == "__main__":
    main()
