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
        for script in ("viz/make_figures.py", "viz/build_site.py"):
            subprocess.run([VENV_PY, os.path.join(ROOT, script)], cwd=ROOT,
                           stdout=lf, stderr=subprocess.STDOUT, check=False)
    return open(log).read()[-4000:]


# ─────────────────────────── the page ───────────────────────────
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
    # existing, already-validated configs first (the original studies' set),
    # customs next; the generated mutation matrix stays out of the primary list
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
        for r in runs) or '<tr><td colspan="5">none yet — run one above</td></tr>'
    scols, srows = substrate_rows()
    subst_html = ""
    if srows:
        head = "".join(f"<th>{html.escape(c)}</th>" for c in scols)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in row) + "</tr>"
                       for row in srows)
        subst_html = (f'<div class="tablewrap"><table><tr>{head}</tr>{body}</table></div>'
                      f'<p class="note">first {len(srows)} kernels — full table: '
                      f'<a href="/reports/2026-07-07_substrate/substrate_verdicts.csv">CSV</a></p>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cuvslam-stack profiler</title>
<style>
:root {{ --acc:#1565c0; --line:#e0e0e0; --ok:#2e7d32; --err:#c62828; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14.5px/1.5 system-ui,sans-serif; background:#f5f7fa; color:#222; }}
header {{ background:linear-gradient(120deg,#0d47a1,#1976d2); color:#fff; padding:16px 30px; }}
header h1 {{ margin:0; font-size:20px; }} header p {{ margin:2px 0 0; opacity:.85; font-size:13px; }}
main {{ max-width:1100px; margin:20px auto; padding:0 16px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:10px;
        padding:16px 20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.card h2 {{ margin:0 0 8px; font-size:16px; }}
.step {{ display:inline-block; background:var(--acc); color:#fff; border-radius:50%;
         width:24px; height:24px; text-align:center; line-height:24px; font-size:13px;
         margin-right:8px; }}
label {{ display:block; margin:8px 0 2px; font-size:12.5px; color:#555; }}
input[type=text],select,textarea {{ width:100%; padding:7px 9px; border:1px solid #bbb;
  border-radius:6px; font:inherit; }}
.row {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 14px; }}
.row2 {{ display:grid; grid-template-columns:2fr 1fr; gap:0 14px; }}
button {{ background:var(--acc); color:#fff; border:0; border-radius:6px;
  padding:9px 18px; font:inherit; cursor:pointer; margin-top:10px; }}
button.sec {{ background:#607d8b; }}
pre.log {{ background:#111; color:#8fd18f; padding:12px; border-radius:8px;
  max-height:300px; overflow:auto; font-size:12px; white-space:pre-wrap; }}
.status {{ font-size:13px; margin:6px 0; }}
.ok {{ color:var(--ok); }} .err {{ color:var(--err); }}
.note {{ font-size:12px; color:#777; }}
details {{ margin:8px 0; }} summary {{ cursor:pointer; color:var(--acc); font-size:13px; }}
.figgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
.figgrid a {{ display:block; border:1px solid var(--line); border-radius:8px;
  overflow:hidden; background:#fff; }}
.figgrid img {{ width:100%; display:block; }}
.figcap {{ font-size:11.5px; color:#666; padding:5px 9px; border-top:1px solid var(--line); }}
table {{ border-collapse:collapse; font-size:12px; width:100%; }}
th,td {{ border:1px solid var(--line); padding:4px 7px; text-align:left; }}
th {{ background:#f5f5f5; }}
.tablewrap {{ max-height:300px; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
h3 {{ font-size:14px; margin:18px 0 6px; }}
</style></head><body>
<header><h1>cuvslam-stack profiler</h1>
<p>controller: pick a target → pick a config → profile → findings (PiM/ISP candidacy, memory behaviour)</p></header>
<main>

<div class="card"><h2><span class="step">1</span>Target</h2>
<div class="row">
  <div><label>profile on</label><select id="runtarget">{tgt_opts}</select></div>
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
<p class="note">set up keys first: <code>ssh-copy-id user@host</code>; then run the doctor — every failed
check prints its one-line fix (driver/CUDA/ncu/NVBit/kernel incompatibilities).</p></details></div>

<div class="card"><h2><span class="step">2</span>Config</h2>
<div class="row2">
  <div><label>existing (validated studies + your own)</label><select id="runcfg">{cfg_opts}</select></div>
  <div><label>hw descriptor (auto per target ok)</label><select id="runhw">{hw_opts}</select></div>
</div>
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
<textarea id="edtext" style="height:220px;font:12px monospace;display:none"></textarea></details></div>

<div class="card"><h2><span class="step">3</span>Profile</h2>
<div class="row">
  <div><label>what to run</label><select id="runprof">
    <option value="regime">full profile — nsys → ncu → nvbit → analyses (+ accuracy)</option>
    <option value="nsys">nsys only (timeline + stages)</option>
    <option value="ncu">ncu only (kernel metrics, windowed)</option>
    <option value="nvbit">nvbit only (memory trace, windowed)</option>
    <option value="none">plain run (accuracy / QoR only)</option></select></div>
  <div><label>&nbsp;</label><button id="runbtn">start on target</button></div>
  <div><label>&nbsp;</label><span id="runstat" class="status"></span></div>
</div>
<p class="note">every run evaluates accuracy/QoR; ncu+nvbit use bounded launch windows (the app
still runs the full sequence). Logs stream back live from the target.</p>
<pre class="log" id="runlog" style="display:none"></pre></div>

<div class="card"><h2>Findings</h2>
<p class="note">what the studies found — one place. Click any figure for full size.</p>

<h3>PiM / ISP substrate candidacy (per kernel, with cross-workload dynamics)</h3>
<div class="figgrid">
{_fig("reports/2026-07-07_substrate/figs/verdict_heatmap.png", "substrate verdict per kernel x workload (GPU / CPU / PiM-bank / PiM-scatter / ISP)")}
{_fig("reports/2026-07-07_substrate/figs/substrate_mix.png", "GPU-time share by best substrate — the offload opportunity")}
{_fig("reports/2026-07-07_substrate/figs/flip_drivers.png", "kernels whose verdict FLIPS across workloads + the metric that drives it")}
</div>
{subst_html}

<h3>Memory behaviour (why: attribution + roofline)</h3>
<div class="figgrid">
{_fig("profiling/reports/2026-07-05_attribution_campaign/figs/memory_space_composition.png", "per-kernel memory-space composition (shared / spill / DRAM) — 27 sequences")}
{_fig("profiling/reports/2026-07-04_campaign/figs/taxonomy.png", "kernel taxonomy classes + cross-sequence stability (91%)")}
{_fig("profiling/reports/2026-07-03_tum_office_rtx2000ada/figs/roofline.png", "DRAM roofline, TUM office (RTX 2000 Ada)")}
{_fig("profiling/reports/2026-07-03_kitti06_rtx2000ada/figs/roofline.png", "DRAM roofline, KITTI 06 (RTX 2000 Ada)")}
{_fig("profiling/reports/2026-07-03_tum_office_rtx2000ada/figs/pim_affinity.png", "PiM-affinity share of GPU time (61%)")}
{_fig("profiling/reports/2026-07-05_attribution_campaign/figs/tag_agreement.png", "data-structure attribution: 48/49 kernels unanimous")}
</div>

<h3>Recent profiled runs (cohesive pipeline)</h3>
<div class="tablewrap"><table>
<tr><th>workload</th><th>when</th><th>captures</th><th>analyses</th><th></th></tr>
{runs_html}</table></div>

<details><summary>validity: accuracy vs the paper + profiler neutrality (checked on every run — not the headline)</summary>
<div class="figgrid">
{_fig("reports/2026-07-07_accuracy_full/figs/traj_kitti.png", "estimated vs ground-truth trajectories (KITTI)")}
{_fig("reports/2026-07-07_accuracy_full/figs/ape_by_config.png", "APE across the 141-run accuracy matrix")}
{_fig("reports/2026-07-07_profiling_coverage/figs/neutrality_scatter.png", "profiling neutrality: APE with vs without profilers")}
{_fig("reports/2026-07-07_profiler_neutrality/figs/profiler_neutrality.png", "nsys / ncu / nvbit all accuracy-neutral")}
</div>
<p class="note"><a href="/site/index.html" target="_blank">full results site (all reports, tables, figures)</a>
&nbsp;·&nbsp; <button class="sec" id="rebuild" style="margin:0;padding:4px 10px">rebuild figures + site</button>
<span id="rebuildout"></span></p></details>
</div>

</main>
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
  const t = $("#runtarget").value;
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
