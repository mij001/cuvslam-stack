#!/usr/bin/env python3
"""dashboard/serve.py — the profiler app (this machine = the CONTROLLER).

One simple loop, stdlib only:

  1 TARGET   pick a machine (local / ssh: workstation, gaming laptop, Jetson
             Orin/Nano/AGX ...) and run the DOCTOR — every failed environment
             check (driver/CUDA/ncu/NVBit/kernel) prints its one-line fix.
  2 CONFIG   pick an existing validated config, or create one: a dataset
             config (cuVSLAM adapter) or ANY GPU workload (command adapter);
             edit files directly when needed.
  3 PROFILE  run on the target (full pipeline by default: nsys → ncu → nvbit
             → analyses; accuracy/QoR evaluated on EVERY run); logs stream
             back live over ssh.
  FINDINGS   the studies' results, cohesively: PiM/ISP substrate candidacy
             per kernel + cross-workload dynamics, memory behaviour
             (attribution, taxonomy, rooflines), recent pipeline runs;
             accuracy/coverage validity is kept but secondary.

Usage:
  python3 dashboard/serve.py [--port 8642] [--bind 127.0.0.1]
Then open http://127.0.0.1:8642/
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mutate_configs import remove_slam, set_key  # noqa: E402  (reuse, one source of truth)

VENV_PY = os.path.join(ROOT, "cuvslam_venv", "bin", "python")
if not os.path.isfile(VENV_PY):
    VENV_PY = sys.executable
CUSTOM_DIR = os.path.join(ROOT, "configs", "custom")
LOG_DIR = os.path.join(ROOT, "out", "dashboard", "logs")
OUT_DIR = os.path.join(ROOT, "out", "dashboard")

# dataset presets → a validated accuracy-matrix config used as the template
PRESETS = {
    "kitti_stereo":   ("configs/base/kitti06_stereo_slam.toml",
                       "KITTI-style stereo (image_2/3 + times.txt, KITTI-pose GT)"),
    "euroc_stereo":   ("configs/base/euroc_MH_01_easy_stereo_slam.toml",
                       "EuRoC-style stereo (mav0/cam0+cam1, EuRoC-csv GT)"),
    "euroc_inertial": ("configs/base/euroc_MH_01_easy_inertial_slam.toml",
                       "EuRoC-style stereo + IMU"),
    "euroc_mono":     ("configs/base/euroc_MH_01_easy_mono_odom.toml",
                       "EuRoC-style monocular (odometry only)"),
    "tum_rgbd":       ("configs/base/tum_fr3_long_office_household_rgbd_slam.toml",
                       "TUM-RGBD-style (rgb.txt/depth.txt associations, TUM GT)"),
    "icl_rgbd":       ("configs/base/icl_living_room_traj1_rgbd_slam.toml",
                       "ICL-NUIM-style RGB-D (TUM layout, synthetic)"),
}

# feature variants → transform applied to the preset template (reused machinery)
VARIANTS = {
    "slam":            ("SLAM (as template)", lambda t: t),
    "odom_only":       ("odometry only", remove_slam),
    "slam_async":      ("SLAM, async loop closure", lambda t: set_key(t, "slam", "sync_mode", "false")),
    "slam_cpu":        ("SLAM on CPU", lambda t: set_key(t, "slam", "use_gpu", "false")),
    "sba_async":       ("async sparse bundle adjustment", lambda t: set_key(t, "odometry", "async_sba", "true")),
    "no_motion_model": ("motion model off", lambda t: set_key(t, "odometry", "use_motion_model", "false")),
    "denoising":       ("input denoising on", lambda t: set_key(t, "odometry", "use_denoising", "true")),
}

JOBS: dict[str, dict] = {}          # jobid -> {proc, cmd, log, cfg, t0}
STATIC_OK = ("site/", "reports/", "profiling/reports/", "results/", "configs/", "out/dashboard/")


# ─────────────────────────── config generation ───────────────────────────
def make_configs(form):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", form.get("name", "")).strip("_")
    preset = form.get("preset", "")
    if not name:
        raise ValueError("dataset name is required")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    tpl_path = os.path.join(ROOT, PRESETS[preset][0])
    text0 = open(tpl_path).read()

    def sub_str(text, key, value, section=None):
        if value:
            return set_key(text, section, key, value) if section else re.sub(
                rf'(?m)^(\s*){key}\s*=\s*".*"$', lambda m: f'{m.group(1)}{key} = "{value}"',
                text, count=1)
        return text

    text0 = sub_str(text0, "path", form.get("input_path", ""))
    text0 = sub_str(text0, "ground_truth", form.get("gt_path", ""))
    if form.get("gt_format"):
        text0 = re.sub(r'(?m)^(\s*)gt_format\s*=\s*".*"$',
                       rf'\g<1>gt_format = "{form["gt_format"]}"', text0, count=1)
    # optional camera overrides (applied to every [[rig.cameras]] block)
    for key in ("size", "focal", "principal"):
        v = form.get(key, "").strip()
        if v:
            nums = re.findall(r"[-\d.]+", v)
            if len(nums) == 2:
                text0 = re.sub(rf"(?m)^(\s*){key}\s*=\s*\[[^\]]*\]",
                               rf"\g<1>{key} = [{nums[0]}, {nums[1]}]", text0)

    chosen = [v for v in VARIANTS if form.get(f"var_{v}") in ("on", "true", "1")]
    if not chosen:
        chosen = ["slam"]
    return _write_variants(name, preset, text0, chosen)


def _write_variants(name, preset, text0, chosen):
    written = []
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    for var in chosen:
        tag = f"{name}__{var}"
        text = VARIANTS[var][1](text0)
        # point outputs at out/dashboard/<tag>/ regardless of template paths
        for key, fname in (("trajectory", "traj_tum.txt"), ("report", "eval.txt")):
            text = re.sub(rf'(?m)^(\s*){key}\s*=\s*".*"$',
                          rf'\g<1>{key} = "out/dashboard/{tag}/{fname}"', text, count=1)
        text = re.sub(r"(?m)^# .*$", f"# dashboard config: {tag} (preset {preset})",
                      text, count=1)
        out = os.path.join(CUSTOM_DIR, tag + ".toml")
        with open(out, "w") as fh:
            fh.write(text)
        os.makedirs(os.path.join(OUT_DIR, tag), exist_ok=True)
        written.append(os.path.relpath(out, ROOT))
    return written


def make_workload_config(form):
    """ANY GPU codebase -> a [workload] config the whole harness understands."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", form.get("name", "")).strip("_")
    cmd = form.get("cmd", "").strip()
    if not name or not cmd:
        raise ValueError("workload name and command are required")
    lines = [f"# GPU workload: {name} (registered via dashboard)",
             "", "[workload]", f'cmd = "{cmd}"']
    if form.get("cwd", "").strip():
        lines.append(f'cwd = "{form["cwd"].strip()}"')
    env_lines = [l for l in form.get("env", "").splitlines() if "=" in l]
    if env_lines:
        lines.append("[workload.env]")
        for l in env_lines:
            k, _, v = l.partition("=")
            lines.append(f'{k.strip()} = "{v.strip()}"')
    if form.get("qor_regex", "").strip():
        lines += ["", "[qor]", f'stdout_regex = "{form["qor_regex"].strip()}"']
    if form.get("nvbit") in ("on", "true", "1"):
        lines += ["", "[profiling]", "nvbit = true"]
    os.makedirs(CUSTOM_DIR, exist_ok=True)
    out = os.path.join(CUSTOM_DIR, f"{name}_workload.toml")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return [os.path.relpath(out, ROOT)]


# ─────────────────────────── profiling targets (controller -> ssh) ─────────
TARGETS_FILE = os.path.join(ROOT, "configs", "targets.toml")


def load_targets():
    """targets.toml -> {name: {ssh, repo, hw, data}}. 'local' always exists."""
    targets = {"local": {"ssh": "", "repo": ".", "hw": "", "data": ""}}
    if os.path.isfile(TARGETS_FILE):
        text = open(TARGETS_FILE).read()
        for m in re.finditer(r"(?ms)^\[targets\.([A-Za-z0-9_-]+)\]\s*$(.*?)(?=^\[|\Z)", text):
            t = {"ssh": "", "repo": "~/Projects/cuvslam-stack", "hw": "", "data": ""}
            for kv in re.finditer(r'(?m)^(ssh|repo|hw|data)\s*=\s*"([^"]*)"', m.group(2)):
                t[kv.group(1)] = kv.group(2)
            targets[m.group(1)] = t
    return targets


def add_target(form):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", form.get("name", "")).strip("_")
    ssh_host = form.get("ssh", "").strip()
    if not name or not ssh_host:
        raise ValueError("target name and ssh host (user@host) are required")
    block = (f'\n[targets.{name}]\nssh = "{ssh_host}"\n'
             f'repo = "{form.get("repo", "~/Projects/cuvslam-stack").strip()}"\n'
             f'hw = "{form.get("hw", "").strip()}"\n'
             f'data = "{form.get("data", "").strip()}"\n')
    with open(TARGETS_FILE, "a") as fh:
        fh.write(block)
    return name


def target_exec(name, remote_cmd, timeout=90):
    """Run a shell command on a target (locally or over ssh); return text."""
    t = load_targets().get(name)
    if t is None:
        raise ValueError(f"unknown target {name!r}")
    if t["ssh"]:
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", t["ssh"],
               f"cd {t['repo']} && {remote_cmd}"]
    else:
        cmd = ["bash", "-c", remote_cmd]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 and "Permission denied" in out or "Could not resolve" in out:
        out += ("\n[controller] ssh to this target failed — check the host is up, "
                "keys are installed (ssh-copy-id), and any VPN (e.g. Tailscale) "
                "is authenticated.")
    return out, r.returncode


# ─────────────────────────── run management ───────────────────────────
def list_configs():
    pats = ["configs/custom/*.toml", "configs/base/*.toml",
            "configs/generated/*.toml", "configs/*.toml",
            "configs/profiling/*.toml"]
    out = []
    for p in pats:
        out += sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, p)))
    return out


def hw_files():
    return sorted(os.path.relpath(f, ROOT)
                  for f in glob.glob(os.path.join(ROOT, "profiling/hw/*.toml")))


def _job_argv(cfg, profiler, hw, remote=False):
    """The command for one job. remote=True -> repo-relative paths + venv python
    (executed on the target via `cd <repo> && ...`)."""
    py = "./cuvslam_venv/bin/python" if remote else VENV_PY
    j = (lambda p: p) if remote else (lambda p: os.path.join(ROOT, p))
    if profiler in ("nsys", "ncu", "nvbit"):
        hw = hw or (hw_files()[0] if hw_files() else "")
        cmd = [py, j("profiling/harness/profile.py"),
               "--config", j(cfg), "--profiler", profiler, "--hw", j(hw)]
        if profiler == "ncu":
            cmd += ["--metrics", "quick", "--launch-skip", "3000", "--launch-count", "20"]
        if profiler == "nvbit":
            cmd += ["--launch-skip", "1000", "--launch-count", "200"]
    elif profiler == "regime":
        # the whole cohesive pipeline: nsys -> window -> ncu -> nvbit -> analyses
        hw = hw or (hw_files()[0] if hw_files() else "")
        cmd = [py, j("profiling/regime.py"), "--config", j(cfg), "--hw", j(hw)]
    else:
        cmd = [py, j("run.py"), j(cfg)]
    return cmd


def start_job(cfg, profiler, hw, target="local"):
    cfg_abs = os.path.join(ROOT, cfg)
    if not os.path.isfile(cfg_abs) or not cfg.startswith("configs") and not cfg.startswith("profiling"):
        raise ValueError("config must be an existing .toml under configs/ or profiling/")
    t = load_targets().get(target)
    if t is None:
        raise ValueError(f"unknown target {target!r}")
    jobid = time.strftime("%H%M%S") + "_" + re.sub(r"\W+", "_", os.path.basename(cfg))[:36] \
        + ("" if target == "local" else f"@{target}")
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, jobid + ".log")
    if t["ssh"]:
        # controller -> target: the config file is shipped first (configs are
        # tiny; targets share the repo via git but custom/generated ones may
        # only exist here), then the job runs in the target repo over ssh —
        # its stdout streams back through the ssh pipe into the local log.
        remote = " ".join(_job_argv(cfg, profiler, hw, remote=True))
        ship = subprocess.run(["scp", "-q", cfg_abs, f"{t['ssh']}:{t['repo']}/{cfg}"],
                              capture_output=True, text=True)
        if ship.returncode != 0:
            raise ValueError(f"cannot ship config to {target}: {ship.stderr.strip()[:200]} "
                             "(is ssh set up? run the doctor on this target)")
        cmd = ["ssh", "-o", "ConnectTimeout=10", t["ssh"],
               f"cd {t['repo']} && exec {remote}"]
    else:
        cmd = _job_argv(cfg, profiler, hw, remote=False)
    env = dict(os.environ)
    env.setdefault("PATH", "")
    # laptop CUDA-install repair shim (harmless elsewhere): prepend if present
    repair = os.path.expanduser("~/.local/cuda-repair/lib")
    if os.path.isdir(repair):
        env["LD_LIBRARY_PATH"] = repair + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    lf = open(log, "w")
    lf.write("$ " + " ".join(cmd) + "\n\n")
    lf.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    JOBS[jobid] = {"proc": proc, "cmd": cmd, "log": log, "cfg": cfg, "t0": time.time(),
                   "target": target}
    return jobid


# ─────────────────────────── file editor (configs) ─────────────────────────
def _editable(relpath):
    """Only TOML under configs/ or profiling/hw/ — and inside the repo."""
    full = os.path.realpath(os.path.join(ROOT, relpath))
    ok_root = full.startswith(os.path.realpath(os.path.join(ROOT, "configs"))) or \
        full.startswith(os.path.realpath(os.path.join(ROOT, "profiling", "hw")))
    if not (ok_root and full.endswith(".toml")):
        raise ValueError("editable files: configs/**.toml and profiling/hw/*.toml")
    return full


def read_file(relpath):
    return open(_editable(relpath)).read()


def save_file(relpath, content):
    full = _editable(relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fh:
        fh.write(content)
    return os.path.relpath(full, ROOT)


def job_state(jobid):
    j = JOBS.get(jobid)
    if not j:
        return None
    rc = j["proc"].poll()
    tail = ""
    try:
        with open(j["log"], "rb") as fh:
            fh.seek(max(0, os.path.getsize(j["log"]) - 8000))
            tail = fh.read().decode(errors="replace")
    except OSError:
        pass
    return {"job": jobid, "cfg": j["cfg"], "running": rc is None, "returncode": rc,
            "elapsed_s": round(time.time() - j["t0"], 1), "log_tail": tail}


def rebuild_site():
    log = os.path.join(LOG_DIR, "rebuild_site.log")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log, "w") as lf:
        for script in ("viz/gen_findings.py", "viz/gen_methodology.py",
                       "viz/make_figures.py", "viz/build_site.py"):
            subprocess.run([VENV_PY, os.path.join(ROOT, script)], cwd=ROOT,
                           stdout=lf, stderr=subprocess.STDOUT, check=False)
    return open(log).read()[-4000:]


# ─────────────────────────── the page ───────────────────────────
def list_summaries():
    """Every summary.json (the standard evidence schema) -> selector list.

    Studies (deep evidence: full metric sets, stages, roofline) come first,
    then the campaign cells (quick ncu windows, one per config)."""
    out = []
    pats = [("study", "profiling/reports/*/summary.json"),
            ("study", "profiling/results/*/summary.json"),
            ("study", "reports/*/summary.json"),
            ("campaign", "reports/*/*.summary.json")]
    for group, pat in pats:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            try:
                s = json.load(open(p))
            except json.JSONDecodeError:
                continue
            out.append({"source": os.path.relpath(p, ROOT),
                        "workload": s.get("workload", "?"),
                        "device": s.get("device", "?"),
                        "adapter": s.get("adapter", "?"),
                        "group": group})
    return out


def read_summary(src):
    full = os.path.realpath(os.path.join(ROOT, src))
    if not full.startswith(os.path.realpath(ROOT)) or not full.endswith(("summary.json",)):
        raise ValueError("src must be a summary.json inside the repo")
    return json.load(open(full))


def regime_runs():
    """Recent cohesive-pipeline runs (manifest.json) -> list for the UI."""
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "profiling/results/*_regime_*")),
                    reverse=True)[:12]:
        mf = os.path.join(d, "manifest.json")
        if not os.path.isfile(mf):
            continue
        try:
            m = json.load(open(mf))
        except json.JSONDecodeError:
            continue
        out.append({"dir": os.path.relpath(d, ROOT), "tag": m.get("tag", "?"),
                    "created": m.get("created", "?"),
                    "captures": list(m.get("captures", {})),
                    "analyses": list(m.get("analyses", {}))})
    return out


def substrate_rows(limit=12):
    """Headline substrate verdicts (kernel -> best substrate), if computed."""
    path = os.path.join(ROOT, "reports/2026-07-07_substrate/substrate_verdicts.csv")
    if not os.path.isfile(path):
        return [], []
    import csv as _csv
    with open(path, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    cols = [c for c in (rows[0] if rows else {})][:6]
    return cols, [[r.get(c, "") for c in cols] for r in rows[:limit]]


def _fig(path, cap):
    if not os.path.isfile(os.path.join(ROOT, path)):
        return ""
    return (f'<a href="/{path}" target="_blank"><img loading="lazy" src="/{path}">'
            f'<div class="figcap">{html.escape(cap)}</div></a>')


def page():
    targets = load_targets()
    tgt_opts = "".join(f"<option>{html.escape(t)}</option>" for t in targets)
    prim = []
    for pat in ("configs/custom/*.toml", "configs/base/*.toml",
                "configs/profiling/*.toml", "configs/*.toml"):
        prim += sorted(os.path.relpath(f, ROOT)
                       for f in glob.glob(os.path.join(ROOT, pat)))
    cfg_opts = "".join(f"<option>{html.escape(c)}</option>" for c in prim)
    hw_opts = "".join(f"<option>{html.escape(h)}</option>" for h in hw_files())
    presets = "".join(f'<option value="{k}">{k} — {html.escape(d)}</option>'
                      for k, (_t, d) in PRESETS.items())
    runs = regime_runs()
    runs_html = "".join(
        f'<tr><td>{html.escape(r["tag"])}</td><td>{html.escape(r["created"])}</td>'
        f'<td>{", ".join(r["captures"])}</td><td>{", ".join(r["analyses"])}</td>'
        f'<td><a href="/{r["dir"]}/manifest.json" target="_blank">manifest</a></td></tr>'
        for r in runs) or '<tr><td colspan="5">none yet</td></tr>'

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU workload characterizer — PiM/ISP placement evidence</title>
<style>
:root {{ --acc:#1565c0; --acc2:#0d47a1; --line:#e3e7ee; --ok:#2e7d32; --err:#c62828;
        --ink:#1c2733; --sub:#5a6b7c; --bg:#f4f6fa; --card:#fff; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; }}
body {{ margin:0; font:14.5px/1.55 "Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
       background:var(--bg); color:var(--ink); }}
header {{ background:linear-gradient(115deg,#0b3d91 0%,#1565c0 70%,#1e88e5 100%);
         color:#fff; padding:20px 34px 0; }}
header h1 {{ margin:0; font-size:21px; letter-spacing:.2px; }}
header p {{ margin:4px 0 14px; opacity:.85; font-size:13px; }}
nav.tabs {{ display:flex; gap:4px; }}
nav.tabs a {{ color:#dce8f7; text-decoration:none; padding:9px 18px; font-size:13.5px;
  border-radius:9px 9px 0 0; background:rgba(255,255,255,.08); }}
nav.tabs a.on {{ background:var(--bg); color:var(--acc2); font-weight:600; }}
main {{ max-width:1180px; margin:22px auto 60px; padding:0 18px; }}
.tab {{ display:none; }} .tab.on {{ display:block; animation:fade .18s ease; }}
@keyframes fade {{ from {{ opacity:.4; transform:translateY(3px); }} to {{ opacity:1; }} }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:18px 22px; margin-bottom:18px; box-shadow:0 1px 3px rgba(15,40,80,.06); }}
.card h2 {{ margin:0 0 8px; font-size:16px; }}
h3 {{ font-size:13.5px; margin:20px 0 8px; text-transform:uppercase; letter-spacing:.6px;
     color:var(--sub); }}
label {{ display:block; margin:8px 0 2px; font-size:12.5px; color:var(--sub); }}
input[type=text],select,textarea {{ width:100%; padding:7px 10px; border:1px solid #c3ccd8;
  border-radius:7px; font:inherit; background:#fff; }}
.row {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 14px; }}
.row2 {{ display:grid; grid-template-columns:2fr 1fr; gap:0 14px; }}
button {{ background:var(--acc); color:#fff; border:0; border-radius:7px;
  padding:9px 18px; font:inherit; cursor:pointer; margin-top:10px; }}
button:hover {{ filter:brightness(1.08); }}
button.sec {{ background:#5a6b7c; }}
pre.log {{ background:#101418; color:#8fd18f; padding:12px; border-radius:9px;
  max-height:300px; overflow:auto; font-size:12px; white-space:pre-wrap; }}
.status {{ font-size:13px; margin:6px 0; }}
.ok {{ color:var(--ok); }} .err {{ color:var(--err); }}
.note {{ font-size:12px; color:var(--sub); }}
details {{ margin:8px 0; }} summary {{ cursor:pointer; color:var(--acc); font-size:13px; }}
table {{ border-collapse:collapse; font-size:12px; width:100%; }}
th,td {{ border:1px solid var(--line); padding:4px 8px; text-align:left; }}
th {{ background:#f1f4f9; }}
.tablewrap {{ max-height:320px; overflow:auto; border:1px solid var(--line); border-radius:9px; }}
/* findings */
.fgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }}
.fcard {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; box-shadow:0 1px 3px rgba(15,40,80,.06); display:flex;
  flex-direction:column; }}
.fstep {{ align-self:flex-start; font-size:10.5px; text-transform:uppercase;
  letter-spacing:.8px; color:#fff; border-radius:10px; padding:2px 10px; }}
.fnum {{ font-size:30px; font-weight:700; margin:8px 0 0; color:var(--acc2);
  font-variant-numeric:tabular-nums; }}
.funit {{ font-size:12px; color:var(--sub); margin-bottom:6px; }}
.ftitle {{ font-size:14px; font-weight:600; margin:4px 0; }}
.fstmt {{ font-size:12.5px; color:#33424f; flex:1; }}
.fev {{ margin-top:10px; }}
.fev a {{ display:inline-block; font-size:12px; color:var(--acc); text-decoration:none;
  border:1px solid var(--acc); border-radius:14px; padding:3px 12px; margin:2px 4px 2px 0; }}
.fev a:hover {{ background:var(--acc); color:#fff; }}
#chartpanel {{ display:none; }}
#chartpanel.on {{ display:block; }}
.charthost {{ min-height:220px; }}
/* methodology stepper */
.mflow {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0 18px; }}
.mnode {{ background:var(--card); border:2px solid var(--acc); color:var(--acc2);
  border-radius:10px; padding:8px 14px; font-weight:600; font-size:13px; }}
.marrow {{ color:var(--sub); font-size:18px; }}
.mnode {{ transition:transform .1s; }} .mnode:hover {{ transform:translateY(-1px); }}
details.mstep {{ border-left:4px solid var(--acc); padding:0; }}
details.mstep > summary {{ list-style:none; padding:16px 20px; }}
details.mstep > summary::-webkit-details-marker {{ display:none; }}
details.mstep > summary::before {{ content:"▸"; color:var(--sub); margin-right:8px; font-size:12px; }}
details.mstep[open] > summary::before {{ content:"▾"; }}
.msectitle {{ font-size:15px; font-weight:600; color:var(--acc2); }}
.mbody {{ padding:0 20px 16px; }}
.mprose {{ font-size:13px; color:#33424f; margin:8px 0; }}
.formula {{ background:#f7f9fc; border-left:3px solid var(--acc); border-radius:0 8px 8px 0;
  padding:10px 14px; margin:10px 0; }}
.formula .fname {{ font-size:12.5px; font-weight:600; }}
.formula .fexpr {{ font-family:ui-monospace,monospace; font-size:13px; color:#0d47a1;
  background:#eef3fb; padding:6px 10px; border-radius:6px; margin:6px 0; overflow-x:auto; }}
.formula .fcounters {{ margin:4px 0; }}
.ccode {{ font-family:ui-monospace,monospace; font-size:11px; background:#e8eef7;
  border-radius:5px; padding:1px 6px; margin:2px 4px 2px 0; display:inline-block; }}
.formula .fnote {{ font-size:11.5px; color:var(--sub); margin-top:4px; white-space:pre-line; }}
.mtabtitle {{ font-size:12px; font-weight:600; color:var(--sub); margin:10px 0 4px;
  text-transform:uppercase; letter-spacing:.4px; }}
.mlink {{ display:inline-block; font-size:12.5px; color:var(--acc); text-decoration:none;
  border:1px solid var(--acc); border-radius:14px; padding:3px 12px; margin:6px 4px 0 0; }}
.mlink:hover {{ background:var(--acc); color:#fff; }}
.dt-template {{ background:#f6f8fb; }}
.mrel a {{ font-size:12px; color:var(--acc); margin-right:10px; }}
/* explorer bits (unchanged classes) */
.step {{ display:inline-block; background:var(--acc); color:#fff; border-radius:50%;
  width:24px; height:24px; text-align:center; line-height:24px; font-size:13px;
  margin-right:8px; }}
.xp-cols {{ display:grid; grid-template-columns:1fr 1fr; gap:0 18px; }}
@media (max-width:900px) {{ .xp-cols {{ grid-template-columns:1fr; }} .row,.row2 {{ grid-template-columns:1fr; }} }}
.xp-row {{ display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:7px;
  cursor:pointer; font-size:12.5px; }}
.xp-row:hover {{ background:#eaf1fa; }}
.xp-sel {{ background:#e2edf9; outline:1px solid var(--acc); }}
.xp-tail {{ color:#8896a5; cursor:default; }}
.xp-pct {{ width:46px; text-align:right; font-variant-numeric:tabular-nums; }}
.xp-bar {{ width:90px; height:9px; background:#e8ecf2; border-radius:5px; overflow:hidden; flex:none; }}
.xp-fill {{ height:100%; background:var(--acc); }}
.xp-name {{ flex:1; font-family:ui-monospace,monospace; font-size:12px; overflow:hidden; white-space:nowrap; }}
.xp-chip {{ color:#fff; border-radius:10px; padding:1px 9px; font-size:10.5px; flex:none; }}
.xp-sub {{ background:#37474f; }}
.xp-evbox {{ border:1px solid var(--line); border-radius:10px; padding:12px 16px; background:#fbfdff; }}
.xp-evtitle {{ font-family:ui-monospace,monospace; font-size:13px; font-weight:600; word-break:break-all; }}
.xp-evsub {{ font-size:12px; color:var(--sub); margin:2px 0 10px; }}
.xp-metric {{ margin:7px 0; }}
.xp-mlabel {{ font-size:12px; }}
.xp-mbar {{ position:relative; height:10px; background:#e8ecf2; border-radius:5px; }}
.xp-mfill {{ height:100%; background:var(--acc); border-radius:5px; }}
.xp-mcut {{ position:absolute; top:-3px; width:2px; height:16px; background:var(--err); }}
.xp-mnote {{ font-size:10.5px; color:#8896a5; }}
.xp-rat {{ font-size:11.5px; color:var(--sub); margin-top:8px; font-style:italic; }}
.xp-verdict {{ margin-top:10px; padding:9px 13px; border-left:4px solid var(--ok);
  background:#f0f7f1; font-size:13px; border-radius:0 8px 8px 0; }}
.xp-stack {{ display:flex; height:20px; border-radius:6px; overflow:hidden; }}
.xp-seg {{ height:100%; color:#fff; font-size:10px; line-height:20px; padding-left:4px;
  white-space:nowrap; overflow:hidden; }}
.xp-legend {{ margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }}
.xp-note {{ font-size:12px; color:var(--sub); }}
.xp-qor {{ font-size:11.5px; color:var(--ok); }}
/* decision trace */
.dt-rule {{ display:flex; gap:8px; align-items:baseline; font-size:12px; padding:5px 8px;
  border-radius:6px; margin:2px 0; font-family:ui-monospace,monospace; }}
.dt-fired {{ background:#e8f5e9; outline:1px solid #a5d6a7; }}
.dt-failed {{ background:#fff; color:#8a97a5; }}
.dt-skipped {{ background:#fff; color:#c2cbd4; }}
.dt-badge {{ font-size:10px; border-radius:8px; padding:1px 8px; color:#fff; flex:none;
  font-family:system-ui; }}
.dt-vals {{ color:#33424f; }}
/* shared tooltip */
#tip {{ position:fixed; display:none; background:#101418; color:#eef3f8; font-size:11.5px;
  padding:7px 10px; border-radius:7px; pointer-events:none; z-index:99; max-width:340px;
  white-space:pre-line; box-shadow:0 3px 12px rgba(0,0,0,.3); }}
footer {{ text-align:center; color:var(--sub); font-size:12px; padding:20px; }}
</style></head><body>
<header>
<h1>GPU workload characterizer</h1>
<p>the full methodology, every number, and the decision behind every PiM / ISP / GPU placement verdict — for cuVSLAM and any adapted codebase</p>
<nav class="tabs">
<a href="#findings" data-tab="findings">Findings</a>
<a href="#explore" data-tab="explore">Explore a run</a>
<a href="#method" data-tab="method">Methodology</a>
<a href="#profile" data-tab="profile">Profile</a>
<a href="#setup" data-tab="setup">Setup</a>
</nav>
</header>
<main>

<!-- ═══════════ FINDINGS ═══════════ -->
<section class="tab" id="tab-findings">
<div class="card" id="chartpanel"><h2 id="chartttl"></h2>
<div class="charthost" id="charthost"></div>
<button class="sec" id="chartclose">close</button></div>
<p class="note" style="margin:4px 2px 12px">every number below is computed from the committed
measurement artifacts (never hand-typed); each card links to its interactive evidence.
Grouped by methodology step — see the <a href="#method">Methodology</a> tab for the pipeline itself.</p>
<div class="fgrid" id="findings"></div>
</section>

<!-- ═══════════ EXPLORE ═══════════ -->
<section class="tab" id="tab-explore">
<div class="card"><h2>Evidence explorer — one profiled run, the whole reasoning chain</h2>
<p class="note"><b>where</b> time goes → <b>who</b> dominates → <b>why</b> (every metric vs its
decision threshold, the attribution join, taxonomy stability) → <b>the decision process</b>
(the exact classifier rules with this kernel's numbers) → <b>verdict</b>.</p>
<div class="row">
  <div><label>filter runs (★ = deep study; rest = campaign cells)</label>
    <input type="text" id="xp-filter" placeholder="kitti, tum, __denoising, _cpu …"></div>
  <div><label>profiled run <span id="xp-count" class="note"></span></label>
    <select id="xp-run"></select></div>
  <div><label>&nbsp;</label><span id="xp-meta" class="note"></span></div>
</div>
<h3>1 · where the GPU time goes</h3>
<div id="xp-stages"></div>
<div class="xp-cols">
<div><h3>2 · who dominates (click a kernel)</h3><div id="xp-kernels"></div></div>
<div><h3>3 · why — evidence vs thresholds</h3><div id="xp-evidence" class="xp-evbox"></div></div>
</div>
<h3>4 · the decision process — classifier rules with this kernel's numbers</h3>
<div id="xp-decision" class="xp-evbox"></div>
<h3>5 · roofline position (hover; size = time share)</h3>
<div id="xp-roofline"></div>
<details><summary>recent full-pipeline runs</summary>
<div class="tablewrap"><table>
<tr><th>workload</th><th>when</th><th>captures</th><th>analyses</th><th></th></tr>
{runs_html}</table></div></details>
</div>
</section>

<!-- ═══════════ METHODOLOGY ═══════════ -->
<section class="tab" id="tab-method">
<div class="card"><h2>The pipeline — how a workload becomes placement decisions</h2>
<div class="mflow" id="mflow"></div>
<div id="msteps"></div>
</div>
</section>

<!-- ═══════════ PROFILE ═══════════ -->
<section class="tab" id="tab-profile">
<div class="card"><h2><span class="step">▶</span>Run / profile on a target</h2>
<div class="row">
  <div><label>config</label><select id="runcfg">{cfg_opts}</select></div>
  <div><label>what to run</label><select id="runprof">
    <option value="regime">full profile — nsys → ncu → nvbit → analyses (+ accuracy)</option>
    <option value="nsys">nsys only (timeline + stages)</option>
    <option value="ncu">ncu only (kernel metrics, windowed)</option>
    <option value="nvbit">nvbit only (memory trace, windowed)</option>
    <option value="none">plain run (accuracy / QoR only)</option></select></div>
  <div><label>hardware descriptor</label><select id="runhw">{hw_opts}</select></div>
</div>
<div class="row">
  <div><label>target</label><select id="runtarget">{tgt_opts}</select></div>
  <div><label>&nbsp;</label><button id="runbtn">start on target</button></div>
  <div><label>&nbsp;</label><span id="runstat" class="status"></span></div>
</div>
<p class="note">every run evaluates accuracy/QoR; ncu+nvbit use bounded launch windows (the app
still runs the full sequence); logs stream back live over ssh.</p>
<pre class="log" id="runlog" style="display:none"></pre></div>
</section>

<!-- ═══════════ SETUP ═══════════ -->
<section class="tab" id="tab-setup">
<div class="card"><h2>Targets — this machine is the controller</h2>
<div class="row">
  <div><label>doctor a target</label><select id="doctortg">{tgt_opts}</select></div>
  <div><label>&nbsp;</label><button type="button" class="sec" id="doctorbtn">doctor — is it ready?</button></div>
  <div><label>&nbsp;</label><span id="tgout" class="status"></span></div>
</div>
<pre class="log" id="doctorout" style="display:none"></pre>
<details><summary>add a target (workstation, gaming laptop, Jetson Orin/Nano/AGX …)</summary>
<form id="tgform"><div class="row">
  <div><label>name</label><input type="text" name="name" placeholder="jetson_orin"></div>
  <div><label>ssh (user@host)</label><input type="text" name="ssh" placeholder="nvidia@jetson"></div>
  <div><label>repo path on target</label><input type="text" name="repo" placeholder="~/cuvslam-stack"></div>
</div><div class="row">
  <div><label>hw descriptor</label><input type="text" name="hw" placeholder="profiling/hw/jetson_orin_sm87.toml"></div>
  <div><label>data root</label><input type="text" name="data" placeholder="/data"></div>
  <div><label>&nbsp;</label><button type="submit">add</button></div>
</div></form>
<p class="note">keys first: <code>ssh-copy-id user@host</code>. The doctor prints a one-line fix for
every failed layer (driver/CUDA/ncu/NVBit/kernel traps).</p></details></div>

<div class="card"><h2>Configs & workloads</h2>
<details><summary>create: dataset config (cuVSLAM adapter)</summary>
<form id="mkform"><div class="row">
  <div><label>name</label><input type="text" name="name" placeholder="my_lab_corridor" required></div>
  <div><label>template preset</label><select name="preset">{presets}</select></div>
  <div><label>dataset path (on the target)</label><input type="text" name="input_path" placeholder="/mnt/data/MySet/seq01"></div>
</div><div class="row">
  <div><label>ground truth (optional)</label><input type="text" name="gt_path"></div>
  <div><label>GT format</label><select name="gt_format"><option value="">(template)</option>
    <option>tum</option><option>euroc</option><option>kitti</option></select></div>
  <div><label>&nbsp;</label><button type="submit">create config</button></div>
</div><input type="hidden" name="var_slam" value="on"><div id="mkout" class="status"></div></form></details>
<details><summary>create: ANY GPU workload (command adapter — torch, CUDA, image processing …)</summary>
<form id="wlform"><div class="row">
  <div><label>name</label><input type="text" name="name" placeholder="my_dnn" required></div>
  <div><label>command</label><input type="text" name="cmd" placeholder="python3 train.py --iters 100" required></div>
  <div><label>working dir (your codebase)</label><input type="text" name="cwd"></div>
</div><div class="row">
  <div><label>QoR regex (proves neutrality)</label><input type="text" name="qor_regex" placeholder="loss=([0-9.eE+-]+)"></div>
  <div><label style="margin-top:24px"><input type="checkbox" name="nvbit"> deep NVBit trace</label></div>
  <div><label>&nbsp;</label><button type="submit">register</button></div>
</div><div id="wlout" class="status"></div></form></details>
<details><summary>edit config files directly</summary>
<div class="row2">
  <div><label>file</label><select id="edsel">{cfg_opts}</select></div>
  <div><label>&nbsp;</label><button type="button" id="edload">load</button>
    <button type="button" id="edsave" class="sec">save</button>
    <span id="edstat" class="status" style="display:inline"></span></div>
</div>
<textarea id="edtext" style="height:220px;font:12px ui-monospace,monospace;display:none"></textarea></details>
<p class="note"><a href="/site/index.html" target="_blank">full static results site</a> ·
<button class="sec" id="rebuild" style="margin:0;padding:4px 10px">rebuild figures + site</button>
<span id="rebuildout"></span></p></div>
</section>

</main>
<div id="tip"></div>
<footer>every chart is drawn from the same CSV/JSON artifacts the reports cite — hover for numbers, click to drill</footer>
<script src="/explorer.js"></script>
<script src="/app.js"></script>
<script>
const $ = s => document.querySelector(s);
$("#tgform") && $("#tgform").addEventListener("submit", async ev => {{
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  const j = await (await fetch("/api/add_target", {{method:"POST", body:JSON.stringify(data)}})).json();
  $("#tgout").innerHTML = j.error ? `<span class="err">✗ ${{j.error}}</span>`
                                  : `<span class="ok">✓ added ${{j.added}} — reload to select</span>`;
}});
$("#doctorbtn").addEventListener("click", async () => {{
  const t = $("#doctortg").value;
  $("#doctorout").style.display = "block";
  $("#doctorout").textContent = `running doctor on ${{t}} …`;
  const j = await (await fetch(`/api/doctor?target=${{encodeURIComponent(t)}}`)).json();
  $("#doctorout").textContent = j.output;
  $("#tgout").innerHTML = j.ready
    ? `<span class="ok">✓ ${{t}} is READY</span>`
    : `<span class="err">✗ ${{t}} not ready — each finding shows its fix</span>`;
}});
$("#mkform").addEventListener("submit", async ev => {{
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  const j = await (await fetch("/api/create", {{method:"POST", body:JSON.stringify(data)}})).json();
  $("#mkout").innerHTML = j.error ? `<span class="err">✗ ${{j.error}}</span>`
    : `<span class="ok">✓ ${{j.written.join(", ")}}</span> — reload to select`;
}});
$("#wlform").addEventListener("submit", async ev => {{
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  const j = await (await fetch("/api/create_workload", {{method:"POST", body:JSON.stringify(data)}})).json();
  $("#wlout").innerHTML = j.error ? `<span class="err">✗ ${{j.error}}</span>`
    : `<span class="ok">✓ ${{j.written[0]}}</span> — reload to select`;
}});
$("#edload").addEventListener("click", async () => {{
  const j = await (await fetch(`/api/file?path=${{encodeURIComponent($("#edsel").value)}}`)).json();
  if (j.error) {{ $("#edstat").innerHTML = `<span class="err">✗ ${{j.error}}</span>`; return; }}
  $("#edtext").style.display = "block"; $("#edtext").value = j.content;
  $("#edstat").innerHTML = `<span class="ok">loaded</span>`;
}});
$("#edsave").addEventListener("click", async () => {{
  const j = await (await fetch("/api/savefile", {{method:"POST", body:JSON.stringify(
    {{path: $("#edsel").value, content: $("#edtext").value}})}})).json();
  $("#edstat").innerHTML = j.error ? `<span class="err">✗ ${{j.error}}</span>`
                                   : `<span class="ok">✓ saved</span>`;
}});
let poll = null;
$("#runbtn").addEventListener("click", async () => {{
  const body = JSON.stringify({{config: $("#runcfg").value, profiler: $("#runprof").value,
    hw: $("#runhw").value, target: $("#runtarget").value}});
  const r = await fetch("/api/run", {{method:"POST", body}});
  const j = await r.json();
  if (j.error) {{ $("#runstat").innerHTML = `<span class="err">✗ ${{j.error}}</span>`; return; }}
  $("#runstat").innerHTML = `<span class="ok">job ${{j.job}}</span>`;
  $("#runlog").style.display = "block";
  if (poll) clearInterval(poll);
  poll = setInterval(async () => {{
    const s = await (await fetch(`/api/job?id=${{encodeURIComponent(j.job)}}`)).json();
    if (!s) return;
    $("#runlog").textContent = s.log_tail || "(no output yet)";
    $("#runlog").scrollTop = $("#runlog").scrollHeight;
    $("#runstat").innerHTML = s.running
      ? `<span>⏳ ${{s.elapsed_s}}s</span>`
      : `<span class="${{s.returncode===0?"ok":"err"}}">${{s.returncode===0?"✓ done":"✗ exit "+s.returncode}} (${{s.elapsed_s}}s)</span>`;
    if (!s.running) clearInterval(poll);
  }}, 2000);
}});
$("#rebuild").addEventListener("click", async () => {{
  $("#rebuildout").textContent = " rebuilding…";
  await (await fetch("/api/rebuild", {{method:"POST"}})).json();
  $("#rebuildout").innerHTML = ` <span class="ok">✓</span>`;
}});
</script></body></html>"""


# ─────────────────────────── http plumbing ───────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _static(self, relpath):
        full = os.path.realpath(os.path.join(ROOT, relpath))
        if not full.startswith(os.path.realpath(ROOT)) or not os.path.isfile(full):
            return self._send(404, "not found", "text/plain")
        ext = os.path.splitext(full)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".png": "image/png",
                 ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml",
                 ".csv": "text/plain; charset=utf-8", ".tsv": "text/plain; charset=utf-8",
                 ".md": "text/plain; charset=utf-8", ".toml": "text/plain; charset=utf-8",
                 ".txt": "text/plain; charset=utf-8", ".json": "application/json",
                 ".pdf": "application/pdf"}.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path).lstrip("/")
        if path in ("", "index.html"):
            return self._send(200, page())
        if path == "api/job":
            q = urllib.parse.parse_qs(url.query)
            return self._json(job_state(q.get("id", [""])[0]))
        if path == "api/configs":
            return self._json(list_configs())
        if path == "api/targets":
            return self._json(load_targets())
        if path == "api/summaries":
            return self._json(list_summaries())
        if path == "api/summary":
            q = urllib.parse.parse_qs(url.query)
            try:
                return self._json(read_summary(q.get("src", [""])[0]))
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
        if path in ("explorer.js", "app.js"):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   path), "rb") as fh:
                return self._send(200, fh.read(), "text/javascript")
        if path == "api/findings":
            fp = os.path.join(ROOT, "reports", "findings.json")
            if not os.path.isfile(fp):
                return self._json({"findings": [], "methodology": []})
            return self._json(json.load(open(fp)))
        if path == "api/methodology":
            fp = os.path.join(ROOT, "reports", "methodology.json")
            if not os.path.isfile(fp):
                return self._json({"sections": [], "roadmap": []})
            return self._json(json.load(open(fp)))
        if path == "api/csv":
            q = urllib.parse.parse_qs(url.query)
            src = q.get("src", [""])[0]
            full = os.path.realpath(os.path.join(ROOT, src))
            ok_roots = (os.path.realpath(os.path.join(ROOT, "reports")),
                        os.path.realpath(os.path.join(ROOT, "profiling", "reports")))
            if not (full.startswith(ok_roots) and full.endswith((".csv", ".tsv"))
                    and os.path.isfile(full)):
                return self._json({"error": "src must be a csv/tsv under reports/"}, 400)
            import csv as _csv
            delim = "\t" if full.endswith(".tsv") else ","
            with open(full, newline="") as fh:
                rows = list(_csv.DictReader(fh, delimiter=delim))
            return self._json({"src": src, "rows": rows})
        if path == "api/doctor":
            q = urllib.parse.parse_qs(url.query)
            name = q.get("target", ["local"])[0]
            try:
                out, rc = target_exec(name, "bash scripts/doctor.sh", timeout=120)
            except Exception as e:  # noqa: BLE001
                out, rc = f"doctor failed to reach target: {e}", 1
            return self._json({"target": name, "output": out, "ready": rc == 0})
        if path == "api/file":
            q = urllib.parse.parse_qs(url.query)
            try:
                return self._json({"path": q.get("path", [""])[0],
                                   "content": read_file(q.get("path", [""])[0])})
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
        if any(path.startswith(p) for p in STATIC_OK):
            return self._static(path)
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            form = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "bad JSON"}, 400)
        try:
            if self.path == "/api/create":
                return self._json({"written": make_configs(form)})
            if self.path == "/api/create_workload":
                return self._json({"written": make_workload_config(form)})
            if self.path == "/api/add_target":
                return self._json({"added": add_target(form)})
            if self.path == "/api/savefile":
                return self._json({"saved": save_file(form.get("path", ""),
                                                      form.get("content", ""))})
            if self.path == "/api/run":
                job = start_job(form.get("config", ""), form.get("profiler", "none"),
                                form.get("hw", ""), form.get("target", "local"))
                return self._json({"job": job})
            if self.path == "/api/rebuild":
                return self._json({"log": rebuild_site()})
        except Exception as e:  # noqa: BLE001 — surface to the UI
            return self._json({"error": str(e)}, 400)
        return self._json({"error": "unknown endpoint"}, 404)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8642)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[dashboard] http://{args.bind}:{args.port}/  (root: {ROOT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
