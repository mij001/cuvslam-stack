#!/usr/bin/env python3
"""dashboard/serve.py — one-page web dashboard for the cuvslam-stack.

Three things, no dependencies beyond the stdlib:
  1. REGISTER a new dataset → generate TOML config variants for it
     (preset templates are the validated accuracy-matrix configs; feature
     variants reuse gen_profiling_coverage's transforms).
  2. RUN any config — plain (run.py) or under a profiler
     (profiling/harness/profile.py --profiler nsys|ncu) — with a live log tail.
  3. VIEW all results: embeds the static results site (viz/build_site.py),
     with a rebuild button that re-runs make_figures + build_site.

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
sys.path.insert(0, ROOT)
from gen_profiling_coverage import remove_slam, set_key  # noqa: E402  (reuse, one source of truth)

VENV_PY = os.path.join(ROOT, "cuvslam_venv", "bin", "python")
if not os.path.isfile(VENV_PY):
    VENV_PY = sys.executable
CUSTOM_DIR = os.path.join(ROOT, "configs", "custom")
LOG_DIR = os.path.join(ROOT, "out", "dashboard", "logs")
OUT_DIR = os.path.join(ROOT, "out", "dashboard")

# dataset presets → a validated accuracy-matrix config used as the template
PRESETS = {
    "kitti_stereo":   ("configs/accuracy_matrix/kitti06_stereo_slam.toml",
                       "KITTI-style stereo (image_2/3 + times.txt, KITTI-pose GT)"),
    "euroc_stereo":   ("configs/accuracy_matrix/euroc_MH_01_easy_stereo_slam.toml",
                       "EuRoC-style stereo (mav0/cam0+cam1, EuRoC-csv GT)"),
    "euroc_inertial": ("configs/accuracy_matrix/euroc_MH_01_easy_inertial_slam.toml",
                       "EuRoC-style stereo + IMU"),
    "euroc_mono":     ("configs/accuracy_matrix/euroc_MH_01_easy_mono_odom.toml",
                       "EuRoC-style monocular (odometry only)"),
    "tum_rgbd":       ("configs/accuracy_matrix/tum_fr3_long_office_household_rgbd_slam.toml",
                       "TUM-RGBD-style (rgb.txt/depth.txt associations, TUM GT)"),
    "icl_rgbd":       ("configs/accuracy_matrix/icl_living_room_traj1_rgbd_slam.toml",
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


# ─────────────────────────── run management ───────────────────────────
def list_configs():
    pats = ["configs/custom/*.toml", "configs/accuracy_matrix/*.toml",
            "configs/profiling_coverage/*.toml", "configs/*.toml",
            "profiling/configs/*.toml"]
    out = []
    for p in pats:
        out += sorted(os.path.relpath(f, ROOT) for f in glob.glob(os.path.join(ROOT, p)))
    return out


def hw_files():
    return sorted(os.path.relpath(f, ROOT)
                  for f in glob.glob(os.path.join(ROOT, "profiling/hw/*.toml")))


def start_job(cfg, profiler, hw):
    cfg_abs = os.path.join(ROOT, cfg)
    if not os.path.isfile(cfg_abs) or not cfg.startswith("configs") and not cfg.startswith("profiling"):
        raise ValueError("config must be an existing .toml under configs/ or profiling/")
    jobid = time.strftime("%H%M%S") + "_" + re.sub(r"\W+", "_", os.path.basename(cfg))[:40]
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, jobid + ".log")
    if profiler in ("nsys", "ncu"):
        hw = hw or (hw_files()[0] if hw_files() else "")
        cmd = [VENV_PY, os.path.join(ROOT, "profiling/harness/profile.py"),
               "--config", cfg_abs, "--profiler", profiler, "--hw", os.path.join(ROOT, hw)]
        if profiler == "ncu":
            cmd += ["--metrics", "quick", "--launch-skip", "3000", "--launch-count", "20"]
    else:
        cmd = [VENV_PY, os.path.join(ROOT, "run.py"), cfg_abs]
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
    JOBS[jobid] = {"proc": proc, "cmd": cmd, "log": log, "cfg": cfg, "t0": time.time()}
    return jobid


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
def page():
    presets = "".join(
        f'<option value="{k}">{k} — {html.escape(d)}</option>'
        for k, (_t, d) in PRESETS.items())
    variants = "".join(
        f'<label class="chk"><input type="checkbox" name="var_{k}" '
        f'{"checked" if k == "slam" else ""}> {k} <span>({html.escape(d)})</span></label>'
        for k, (d, _f) in VARIANTS.items())
    cfg_opts = "".join(f"<option>{html.escape(c)}</option>" for c in list_configs())
    hw_opts = "".join(f"<option>{html.escape(h)}</option>" for h in hw_files())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cuvslam-stack dashboard</title>
<style>
:root {{ --acc:#1565c0; --line:#e0e0e0; --ok:#2e7d32; --err:#c62828; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:14.5px/1.5 system-ui,sans-serif; background:#f5f7fa; color:#222; }}
header {{ background:linear-gradient(120deg,#0d47a1,#1976d2); color:#fff; padding:18px 30px; }}
header h1 {{ margin:0; font-size:20px; }} header p {{ margin:2px 0 0; opacity:.85; font-size:13px; }}
main {{ max-width:1100px; margin:22px auto; padding:0 16px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:10px;
        padding:18px 22px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.card h2 {{ margin:0 0 10px; font-size:16.5px; }}
label {{ display:block; margin:8px 0 2px; font-size:12.5px; color:#555; }}
input[type=text],select {{ width:100%; padding:7px 9px; border:1px solid #bbb;
  border-radius:6px; font:inherit; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:0 18px; }}
.grid3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:0 18px; }}
.chk {{ display:inline-block; margin:4px 14px 4px 0; font-size:13px; color:#333; }}
.chk span {{ color:#888; font-size:11.5px; }}
button {{ background:var(--acc); color:#fff; border:0; border-radius:6px;
  padding:9px 18px; font:inherit; cursor:pointer; margin-top:12px; }}
button:hover {{ filter:brightness(1.1); }}
button.sec {{ background:#607d8b; }}
pre.log {{ background:#111; color:#8fd18f; padding:12px; border-radius:8px;
  max-height:340px; overflow:auto; font-size:12px; white-space:pre-wrap; }}
.status {{ font-size:13px; margin:8px 0; }}
.ok {{ color:var(--ok); }} .err {{ color:var(--err); }}
iframe {{ width:100%; height:75vh; border:1px solid var(--line); border-radius:8px; background:#fff; }}
.note {{ font-size:12px; color:#777; }}
</style></head><body>
<header><h1>cuvslam-stack dashboard</h1>
<p>register a dataset → generate TOML variants → run / profile → view results</p></header>
<main>

<div class="card"><h2>1 · New dataset → TOML configs</h2>
<form id="mkform">
<div class="grid2">
  <div><label>dataset name (tag)</label>
    <input type="text" name="name" placeholder="my_lab_corridor" required></div>
  <div><label>preset (template = validated accuracy-matrix config)</label>
    <select name="preset">{presets}</select></div>
</div>
<div class="grid2">
  <div><label>dataset path (on the machine that runs it)</label>
    <input type="text" name="input_path" placeholder="/mnt/data/MyDataset/seq01"></div>
  <div><label>ground-truth file (optional, for [eval])</label>
    <input type="text" name="gt_path" placeholder="/mnt/data/MyDataset/seq01/groundtruth.txt"></div>
</div>
<div class="grid3">
  <div><label>GT format</label>
    <select name="gt_format"><option value="">(keep template)</option>
    <option>tum</option><option>euroc</option><option>kitti</option></select></div>
  <div><label>image size override, e.g. 640, 480 (optional)</label>
    <input type="text" name="size" placeholder=""></div>
  <div><label>focal fx, fy override (optional)</label>
    <input type="text" name="focal" placeholder=""></div>
</div>
<label>principal cx, cy override (optional)</label>
<input type="text" name="principal" placeholder="">
<label style="margin-top:10px">cuVSLAM feature variants to emit</label>
{variants}
<br><button type="submit">generate configs</button>
<div id="mkout" class="status"></div>
</form></div>

<div class="card"><h2>2 · Run / profile a config</h2>
<div class="grid3">
  <div><label>config</label><select id="runcfg">{cfg_opts}</select></div>
  <div><label>profiler</label><select id="runprof">
    <option value="none">none — plain run.py (accuracy)</option>
    <option value="nsys">nsys — Nsight Systems timeline</option>
    <option value="ncu">ncu — Nsight Compute (windowed)</option></select></div>
  <div><label>hardware descriptor (profiled runs)</label>
    <select id="runhw">{hw_opts}</select></div>
</div>
<button id="runbtn">start run</button>
<span class="note">&nbsp; jobs run detached (start_new_session) with logs under out/dashboard/logs/</span>
<div id="runstat" class="status"></div>
<pre class="log" id="runlog" style="display:none"></pre></div>

<div class="card"><h2>3 · Results</h2>
<button class="sec" id="rebuild">rebuild figures + site</button>
<span class="note">&nbsp; runs viz/make_figures.py + viz/build_site.py</span>
<div id="rebuildout" class="status"></div>
<iframe src="/site/index.html" title="results site"></iframe></div>

</main>
<script>
const $ = s => document.querySelector(s);
$("#mkform").addEventListener("submit", async ev => {{
  ev.preventDefault();
  const data = Object.fromEntries(new FormData(ev.target).entries());
  const r = await fetch("/api/create", {{method:"POST", body:JSON.stringify(data)}});
  const j = await r.json();
  $("#mkout").innerHTML = j.error
    ? `<span class="err">✗ ${{j.error}}</span>`
    : `<span class="ok">✓ wrote ${{j.written.length}} config(s):</span> ` +
      j.written.map(w=>`<code>${{w}}</code>`).join(", ") +
      ` — <a href="#" onclick="location.reload();return false">refresh run list</a>`;
}});
let poll = null;
$("#runbtn").addEventListener("click", async () => {{
  const body = JSON.stringify({{config: $("#runcfg").value,
    profiler: $("#runprof").value, hw: $("#runhw").value}});
  const r = await fetch("/api/run", {{method:"POST", body}});
  const j = await r.json();
  if (j.error) {{ $("#runstat").innerHTML = `<span class="err">✗ ${{j.error}}</span>`; return; }}
  $("#runstat").innerHTML = `<span class="ok">job ${{j.job}} started</span>`;
  $("#runlog").style.display = "block";
  if (poll) clearInterval(poll);
  poll = setInterval(async () => {{
    const s = await (await fetch(`/api/job?id=${{encodeURIComponent(j.job)}}`)).json();
    if (!s) return;
    $("#runlog").textContent = s.log_tail || "(no output yet)";
    $("#runlog").scrollTop = $("#runlog").scrollHeight;
    $("#runstat").innerHTML = s.running
      ? `<span>⏳ running — ${{s.elapsed_s}}s</span>`
      : `<span class="${{s.returncode===0?"ok":"err"}}">
         ${{s.returncode===0?"✓ finished":"✗ exit "+s.returncode}} in ${{s.elapsed_s}}s</span>`;
    if (!s.running) clearInterval(poll);
  }}, 2000);
}});
$("#rebuild").addEventListener("click", async () => {{
  $("#rebuildout").textContent = "rebuilding…";
  const j = await (await fetch("/api/rebuild", {{method:"POST"}})).json();
  $("#rebuildout").innerHTML = `<span class="ok">✓ rebuilt</span>`;
  document.querySelector("iframe").src = "/site/index.html?" + Date.now();
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
            if self.path == "/api/run":
                job = start_job(form.get("config", ""), form.get("profiler", "none"),
                                form.get("hw", ""))
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
