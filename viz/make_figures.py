#!/usr/bin/env python3
"""make_figures.py — visual counterparts for every machine-readable output in
this repo. Each figure function reads a committed CSV/TSV artifact and writes a
PNG next to it (in a figs/ subdir); build_site.py then assembles them into the
browsable results site.

Figures produced (data source in parens):
  accuracy      APE by dataset x mode, run scatter, matched-pose coverage
                (reports/2026-07-07_accuracy_full/accuracy_matrix.csv)
  coverage      plain-vs-nsys neutrality scatter + delta CDF + OK/CHECK by mode
                (reports/2026-07-07_profiling_coverage/coverage_results.tsv)
  toggles       feature-toggle accuracy effects per representative sequence
                (same tsv, __<toggle> variants)
  neutrality    plain/nsys/ncu/nvbit APE bars per sequence
                (reports/2026-07-07_profiler_neutrality/neutrality.tsv)
  attribution   per-kernel memory-space composition + top-tag agreement
                (profiling/reports/2026-07-05_attribution_campaign/*.csv)
  taxonomy      kernel class distribution + cross-sequence agreement
                (profiling/reports/2026-07-04_campaign/class_agreement.csv)
  roofline      DRAM roofline scatter per device report
                (profiling/reports/*/data/roofline.csv)
  pim           PiM affinity time share + placement-model speedups
                (profiling/reports/*/data/classification.csv,
                 results/pim_model_per_sequence.csv)
  trajectories  est-vs-GT small multiples per dataset family — needs run dirs,
                so this one is generated on the workstation (--traj-root)

Usage:
  python3 viz/make_figures.py                 # all data figures (laptop-safe)
  python3 viz/make_figures.py --only coverage,toggles
  python3 viz/make_figures.py --only trajectories \
      --traj-root /mnt/data/accuracy_out      # on the workstation
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# one consistent look
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "figure.facecolor": "white",
})
OK_C, CHECK_C, ACC = "#2e7d32", "#c62828", "#1565c0"
PROFILER_C = {"plain": "#546e7a", "nsys": "#1565c0", "ncu": "#6a1b9a", "nvbit": "#ef6c00"}


def read_rows(path, delim=None):
    if not os.path.isfile(path):
        return []
    delim = delim or ("\t" if path.endswith(".tsv") else ",")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def fnum(v, default=math.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def save(fig, relpath):
    out = os.path.join(ROOT, relpath)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {relpath}")


# ────────────────────────── accuracy matrix ──────────────────────────
def fig_accuracy():
    rows = read_rows(os.path.join(ROOT, "reports/2026-07-07_accuracy_full/accuracy_matrix.csv"))
    if not rows:
        return
    conv = [r for r in rows if fnum(r["ape_m"]) == fnum(r["ape_m"]) and fnum(r["ape_m"]) < 50]

    # (1) APE by dataset x variant/mode — grouped box-ish strip
    fig, ax = plt.subplots(figsize=(11, 4.6))
    groups, labels = [], []
    for ds in sorted({r["dataset"] for r in conv}):
        for var in sorted({r["variant"] for r in conv if r["dataset"] == ds}):
            for mode in ("odom", "slam"):
                vals = [fnum(r["ape_m"]) for r in conv
                        if r["dataset"] == ds and r["variant"] == var and r["mode"] == mode]
                if vals:
                    groups.append(vals)
                    labels.append(f"{ds}\n{var}/{mode}")
    pos = np.arange(len(groups))
    for i, vals in enumerate(groups):
        ax.scatter([i] * len(vals), vals, s=14, alpha=0.65,
                   c=[ACC if v < 1 else CHECK_C for v in vals], zorder=3)
        ax.plot([i - 0.28, i + 0.28], [np.median(vals)] * 2, c="k", lw=1.6, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(pos, labels, rotation=60, ha="right", fontsize=6.5)
    ax.set_ylabel("ATE / RMSE APE (m, log)")
    ax.set_title("Accuracy matrix — 141 runs: APE by dataset × sensor variant × pipeline mode "
                 "(bar = median; blue <1 m)")
    save(fig, "reports/2026-07-07_accuracy_full/figs/ape_by_config.png")

    # (2) SLAM-vs-odometry improvement scatter
    by_key = {}
    for r in conv:
        by_key[(r["dataset"], r["sequence"], r["variant"], r["mode"])] = fnum(r["ape_m"])
    pts = []
    for (ds, seq, var, mode), v in by_key.items():
        if mode == "odom" and (ds, seq, var, "slam") in by_key:
            pts.append((v, by_key[(ds, seq, var, "slam")], ds))
    if pts:
        fig, ax = plt.subplots(figsize=(5.4, 5))
        dss = sorted({p[2] for p in pts})
        cmap = plt.cm.tab10(np.linspace(0, 1, len(dss)))
        for ds, c in zip(dss, cmap):
            sel = [(x, y) for x, y, d in pts if d == ds]
            ax.scatter(*zip(*sel), s=26, alpha=0.8, color=c, label=ds)
        lim = [min(min(p[0], p[1]) for p in pts) * 0.7, max(max(p[0], p[1]) for p in pts) * 1.4]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("odometry-only APE (m)"); ax.set_ylabel("SLAM APE (m)")
        ax.set_title("Loop closure helps: SLAM vs odometry APE\n(below diagonal = SLAM better)")
        ax.legend(fontsize=7)
        save(fig, "reports/2026-07-07_accuracy_full/figs/slam_vs_odom.png")


# ────────────────────────── coverage campaign ──────────────────────────
def _coverage_rows():
    return read_rows(os.path.join(
        ROOT, "reports/2026-07-07_profiling_coverage/coverage_results.tsv"))


def fig_coverage():
    rows = _coverage_rows()
    if not rows:
        return
    ok = [r for r in rows if r["status"] == "OK"]
    chk = [r for r in rows if r["status"] == "CHECK"]

    # (1) the headline: plain vs nsys APE, log-log
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for sel, c, lab, m in ((ok, OK_C, f"OK ({len(ok)})", "o"),
                           (chk, CHECK_C, f"CHECK ({len(chk)})", "^")):
        x = [max(fnum(r["plain_APE_m"]), 1e-4) for r in sel]
        y = [max(fnum(r["nsys_APE_m"]), 1e-4) for r in sel]
        ax.scatter(x, y, s=22, c=c, alpha=0.75, marker=m, label=lab, zorder=3)
    lim = [8e-4, 2e5]
    ax.plot(lim, lim, "k-", lw=1, alpha=0.7, label="y = x (bit-identical)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("plain (un-profiled) APE (m)")
    ax.set_ylabel("under nsys APE (m)")
    ax.set_title("Profiling neutrality, 192 variants: APE with vs without Nsight Systems\n"
                 "points on the diagonal = profiling changed nothing")
    ax.legend(fontsize=8)
    save(fig, "reports/2026-07-07_profiling_coverage/figs/neutrality_scatter.png")

    # (2) |delta| CDF for finite pairs
    deltas = sorted(fnum(r["delta_m"]) for r in rows
                    if fnum(r["delta_m"]) == fnum(r["delta_m"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xs = [max(d, 1e-5) for d in deltas]
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=ACC, lw=1.8)
    ax.axvline(0.05, color=CHECK_C, ls="--", lw=1, label="5 cm tolerance")
    n0 = sum(1 for d in deltas if d == 0)
    ax.set_xscale("log")
    ax.set_xlabel("|APE(nsys) − APE(plain)| (m, log)")
    ax.set_ylabel("fraction of variants")
    ax.set_title(f"Neutrality delta CDF — {n0}/{len(deltas)} variants bit-identical (Δ=0)")
    ax.legend()
    save(fig, "reports/2026-07-07_profiling_coverage/figs/delta_cdf.png")

    # (3) OK / CHECK by pipeline mode
    modes = sorted({r["mode"] for r in rows})
    okc = [sum(1 for r in ok if r["mode"] == m) for m in modes]
    ckc = [sum(1 for r in chk if r["mode"] == m) for m in modes]
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    p = np.arange(len(modes))
    ax.bar(p, okc, color=OK_C, label="OK")
    ax.bar(p, ckc, bottom=okc, color=CHECK_C, label="CHECK (all classified benign)")
    for i, (a, b) in enumerate(zip(okc, ckc)):
        ax.text(i, a + b + 0.6, str(a + b), ha="center", fontsize=8)
    ax.set_xticks(p, modes, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("variants")
    ax.set_title("Coverage campaign: neutrality verdict by sensor × pipeline mode")
    ax.legend()
    save(fig, "reports/2026-07-07_profiling_coverage/figs/verdict_by_mode.png")


def fig_toggles():
    rows = _coverage_rows()
    if not rows:
        return
    # depth variants: <base>__<toggle>, toggle != base
    per_base = {}
    for r in rows:
        m = re.match(r"(.+)__([a-z_]+)$", r["variant"])
        if not m or m.group(2) == "base":
            continue
        per_base.setdefault(m.group(1), {})[m.group(2)] = fnum(r["plain_APE_m"])
    base_ape = {re.sub(r"__base$", "", r["variant"]): fnum(r["plain_APE_m"])
                for r in rows if r["variant"].endswith("__base")}

    reps = [b for b in per_base if len(per_base[b]) >= 8]
    if not reps:
        return
    fig, axes = plt.subplots(1, len(reps), figsize=(3.1 * len(reps), 4.4), sharey=False)
    if len(reps) == 1:
        axes = [axes]
    for ax, b in zip(axes, sorted(reps)):
        togs = sorted(per_base[b])
        vals = [per_base[b][t] for t in togs]
        ref = base_ape.get(b)
        colors = ["#78909c" if ref is None or not (v == v) else
                  (OK_C if v <= ref * 1.05 else CHECK_C) for v in vals]
        ax.barh(np.arange(len(togs)), vals, color=colors)
        if ref is not None and ref == ref:
            ax.axvline(ref, color="k", ls="--", lw=1.2, label=f"base = {ref:g} m")
            ax.legend(fontsize=6.5, loc="lower right")
        ax.set_yticks(np.arange(len(togs)), togs, fontsize=7)
        ax.set_xscale("log")
        from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        sf = ScalarFormatter(); sf.set_scientific(False)
        ax.xaxis.set_major_formatter(sf)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("APE (m, log)", fontsize=7.5)
        ax.set_title(b.replace("_", " ")[:34], fontsize=8)
    fig.suptitle("One dataset → many cuVSLAM behaviors: feature-toggle effect on accuracy "
                 "(green ≤ base, red > base)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "reports/2026-07-07_profiling_coverage/figs/toggle_effects.png")


# ────────────────────────── three-profiler neutrality ──────────────────────────
def fig_neutrality():
    rows = read_rows(os.path.join(
        ROOT, "reports/2026-07-07_profiler_neutrality/neutrality.tsv"))
    if not rows:
        return
    seqs = [r["sequence"] for r in rows]
    profs = ["plain", "nsys", "ncu", "nvbit"]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    w = 0.2
    p = np.arange(len(seqs))
    for i, pr in enumerate(profs):
        vals = [fnum(r[f"{pr}_APE"]) for r in rows]
        ax.bar(p + (i - 1.5) * w, vals, w, color=PROFILER_C[pr], label=pr)
    ax.set_yscale("log")
    ax.set_xticks(p, [s.replace("_", "\n", 2) for s in seqs], fontsize=7)
    ax.set_ylabel("APE (m, log)")
    ax.set_title("Three-profiler neutrality: identical accuracy under nsys / ncu / NVBit\n"
                 "(nsys+ncu bit-identical to plain on deterministic sequences; NVBit ≤2 mm)")
    ax.legend(ncols=4, fontsize=8)
    save(fig, "reports/2026-07-07_profiler_neutrality/figs/profiler_neutrality.png")


# ────────────────────────── attribution ──────────────────────────
def fig_attribution():
    path = os.path.join(
        ROOT, "profiling/reports/2026-07-05_attribution_campaign/attribution_consistency.csv")
    rows = read_rows(path)
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -fnum(r["med_global_pct"], 0))
    kerns = [r["kernel"][:34] for r in rows]
    sh = [fnum(r["med_shared_pct"], 0) for r in rows]
    sp = [fnum(r["med_local_spill_pct"], 0) for r in rows]
    gl = [fnum(r["med_global_pct"], 0) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(kerns) + 1.6))
    p = np.arange(len(kerns))
    ax.barh(p, sh, color="#4db6ac", label="shared (on-chip)")
    ax.barh(p, sp, left=sh, color="#ffb74d", label="local / register spill")
    ax.barh(p, gl, left=[a + b for a, b in zip(sh, sp)], color="#7986cb", label="global (DRAM-visible)")
    ax.set_yticks(p, kerns, fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("median % of memory accesses")
    ax.set_xlim(0, 100)
    ax.set_title("Attribution campaign (27 seq): per-kernel memory-space composition\n"
                 "48/49 kernels have a unanimous top data-structure tag")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "profiling/reports/2026-07-05_attribution_campaign/figs/memory_space_composition.png")

    # agreement histogram
    agree = [fnum(r["agreement_pct"], 0) for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.hist(agree, bins=np.arange(0, 105, 5), color=ACC, alpha=0.85)
    ax.set_xlabel("cross-sequence top-tag agreement (%)")
    ax.set_ylabel("kernels")
    ax.set_title(f"Top data-structure tag agreement across sequences "
                 f"({sum(1 for a in agree if a >= 99.9)}/{len(agree)} unanimous)")
    save(fig, "profiling/reports/2026-07-05_attribution_campaign/figs/tag_agreement.png")


# ────────────────────────── taxonomy / campaign ──────────────────────────
def fig_taxonomy():
    rows = read_rows(os.path.join(
        ROOT, "profiling/reports/2026-07-04_campaign/class_agreement.csv"))
    if not rows:
        return
    classes = {}
    for r in rows:
        classes.setdefault(r["modal_class"], []).append(fnum(r["modal_frac"], 0) * 100)
    labels = sorted(classes)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    a1.bar(np.arange(len(labels)), [len(classes[c]) for c in labels], color=ACC, alpha=0.85)
    a1.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    a1.set_ylabel("kernels")
    a1.set_title("Kernel taxonomy: modal class distribution")
    fracs = [fnum(r["modal_frac"], 0) * 100 for r in rows]
    a2.hist(fracs, bins=np.arange(0, 105, 5), color="#6a1b9a", alpha=0.8)
    a2.axvline(np.mean(fracs), color="k", ls="--", lw=1.2,
               label=f"mean {np.mean(fracs):.0f}%")
    a2.set_xlabel("cross-sequence modal agreement (%)")
    a2.set_ylabel("kernels")
    a2.set_title("Class stability across 27 sequences (91% time-weighted)")
    a2.legend(fontsize=8)
    save(fig, "profiling/reports/2026-07-04_campaign/figs/taxonomy.png")


# ────────────────────────── roofline + classification ──────────────────────────
def _device_reports():
    base = os.path.join(ROOT, "profiling/reports")
    for d in sorted(os.listdir(base)):
        data = os.path.join(base, d, "data")
        if os.path.isdir(data):
            yield d, data


def fig_roofline():
    for rep, data in _device_reports():
        rows = read_rows(os.path.join(data, "roofline.csv"))
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        stages = sorted({r["stage"] for r in rows})
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(stages), 3)))
        for st, c in zip(stages, cmap):
            sel = [r for r in rows if r["stage"] == st]
            x = [max(fnum(r["ai_dram"], 1e-3), 1e-3) for r in sel]
            y = [max(fnum(r["gflops"], 1e-3), 1e-3) for r in sel]
            s = [8 + 3 * math.sqrt(fnum(r["time_ms"], 1)) for r in sel]
            ax.scatter(x, y, s=s, color=c, alpha=0.75, label=st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
        ax.set_ylabel("achieved GFLOP/s")
        ax.set_title(f"DRAM roofline — {rep}\n(marker size ∝ kernel time)")
        ax.legend(fontsize=7)
        save(fig, f"profiling/reports/{rep}/figs/roofline.png")


def fig_pim():
    # classification: PiM-affinity share of GPU time per device report
    for rep, data in _device_reports():
        rows = read_rows(os.path.join(data, "classification.csv"))
        if not rows:
            continue
        buckets = {}
        for r in rows:
            buckets.setdefault(r.get("pim_affinity", "?") or "?", 0.0)
            buckets[r.get("pim_affinity", "?") or "?"] += fnum(r["time_ms"], 0)
        total = sum(buckets.values()) or 1
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        order = sorted(buckets, key=buckets.get, reverse=True)
        colors = {"strong": "#2e7d32", "conditional": "#f9a825", "none": "#90a4ae",
                  "weak": "#bdbdbd"}
        ax.pie([buckets[k] for k in order],
               labels=[f"{k}\n{buckets[k]/total*100:.0f}%" for k in order],
               colors=[colors.get(k, "#ce93d8") for k in order],
               startangle=90, wedgeprops={"edgecolor": "w"})
        ax.set_title(f"PiM affinity — share of GPU time\n{rep}")
        save(fig, f"profiling/reports/{rep}/figs/pim_affinity.png")

    # placement model speedups
    rows = read_rows(os.path.join(ROOT, "results/pim_model_per_sequence.csv"))
    if rows:
        scens = sorted({r["scenario"] for r in rows})
        fig, ax = plt.subplots(figsize=(8.6, 3.8))
        seqs = sorted({r["sequence"] for r in rows})
        p = np.arange(len(seqs))
        w = 0.8 / len(scens)
        cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(scens)))
        for i, sc in enumerate(scens):
            v = {r["sequence"]: fnum(r["speedup_selective"]) for r in rows
                 if r["scenario"] == sc}
            ax.bar(p + i * w, [v.get(s, math.nan) for s in seqs], w,
                   color=cmap[i], label=sc)
        ax.axhline(1.0, color="k", lw=1, ls="--")
        ax.set_xticks(p + 0.4 - w / 2, [s.replace("_", "\n", 1) for s in seqs], fontsize=6)
        ax.set_ylabel("selective-offload speedup ×")
        ax.set_title("PiM placement model: selective-offload speedup per sequence")
        ax.legend(fontsize=8)
        save(fig, "results/figs/pim_model_speedups.png")


# ────────────────────────── trajectories (workstation) ──────────────────────────
def _load_tum(path):
    pts = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.split()
            if len(f) >= 4:
                pts.append((float(f[1]), float(f[2]), float(f[3])))
    return np.array(pts)


def _load_gt_any(path, fmt):
    if fmt == "kitti":
        M = np.loadtxt(path)
        return M[:, [3, 7, 11]]
    if fmt == "euroc":
        pts = []
        with open(path) as fh:
            for ln in fh:
                if ln.startswith("#") or "," not in ln:
                    continue
                f = ln.split(",")
                pts.append((float(f[1]), float(f[2]), float(f[3])))
        return np.array(pts)
    return _load_tum(path)  # tum


def _umeyama_align(est, gt_i):
    """SE3 (rot+trans, no scale) alignment of est onto gt for plotting."""
    mu_e, mu_g = est.mean(0), gt_i.mean(0)
    E, G = est - mu_e, gt_i - mu_g
    U, _, Vt = np.linalg.svd(E.T @ G)
    S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ S @ Vt).T
    return est @ R.T + (mu_g - R @ mu_e)


def fig_trajectories(traj_root, config_dir="configs/accuracy_matrix"):
    """est-vs-GT top-down small multiples grouped by dataset family. Runs where
    the run dirs + datasets live (the workstation)."""
    cdir = os.path.join(ROOT, config_dir)
    runs = {}
    for cfg in sorted(os.listdir(cdir)):
        if not cfg.endswith("_slam.toml"):
            continue  # SLAM runs only: one per sequence family
        tag = cfg[:-5]
        text = open(os.path.join(cdir, cfg)).read()
        gt = re.search(r'ground_truth\s*=\s*"([^"]+)"', text)
        fmt = re.search(r'gt_format\s*=\s*"([^"]+)"', text)
        traj = os.path.join(traj_root, tag, "traj_tum.txt")
        if gt and fmt and os.path.isfile(traj) and os.path.isfile(gt.group(1)):
            fam = re.match(r"[a-z]+", tag).group(0)  # kitti06 -> kitti
            runs.setdefault(fam, []).append((tag, traj, gt.group(1), fmt.group(1)))

    for fam, items in runs.items():
        items = items[:12]
        n = len(items)
        cols = min(4, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(3.1 * cols, 3.0 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis("off")
        for ax, (tag, traj, gtp, fmt) in zip(axes, items):
            try:
                est, gt_arr = _load_tum(traj), _load_gt_any(gtp, fmt)
                if len(est) < 10 or len(gt_arr) < 10:
                    raise ValueError("too short")
                idx = np.linspace(0, len(gt_arr) - 1, min(len(est), len(gt_arr))).astype(int)
                est_i = est[np.linspace(0, len(est) - 1, len(idx)).astype(int)]
                est_a = _umeyama_align(est_i, gt_arr[idx])
                # top-down: pick the two highest-variance axes of GT
                var_order = np.argsort(gt_arr[idx].var(0))[::-1][:2]
                a, b = sorted(var_order)
                ax.plot(gt_arr[idx][:, a], gt_arr[idx][:, b], c="#9e9e9e", lw=1.4,
                        label="ground truth")
                ax.plot(est_a[:, a], est_a[:, b], c=ACC, lw=1.0, label="cuVSLAM")
                ax.set_title(tag.replace(f"{fam}_", "").replace("_slam", ""), fontsize=7)
                ax.set_aspect("equal", adjustable="datalim")
                ax.tick_params(labelsize=6)
            except Exception as e:  # noqa: BLE001 — skip unplottable, keep grid
                ax.text(0.5, 0.5, f"{tag}\n({e})", ha="center", va="center",
                        fontsize=6, transform=ax.transAxes)
                ax.axis("off")
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{fam} — estimated vs ground-truth trajectories (SLAM, SE3-aligned, top-down)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"reports/2026-07-07_accuracy_full/figs/traj_{fam}.png")


ALL = {
    "accuracy": fig_accuracy,
    "coverage": fig_coverage,
    "toggles": fig_toggles,
    "neutrality": fig_neutrality,
    "attribution": fig_attribution,
    "taxonomy": fig_taxonomy,
    "roofline": fig_roofline,
    "pim": fig_pim,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma list of figure sets (default: all data figures)")
    ap.add_argument("--traj-root", help="run-output root (e.g. /mnt/data/accuracy_out) "
                                        "to also draw trajectory figures")
    args = ap.parse_args()
    wanted = args.only.split(",") if args.only else list(ALL)
    for name in wanted:
        if name == "trajectories":
            continue
        if name not in ALL:
            sys.exit(f"unknown figure set: {name} (have {', '.join(ALL)})")
        ALL[name]()
    if args.traj_root or (args.only and "trajectories" in args.only):
        if not args.traj_root:
            sys.exit("trajectories needs --traj-root")
        fig_trajectories(args.traj_root)


if __name__ == "__main__":
    main()
