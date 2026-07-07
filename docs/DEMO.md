# Demo runbook — controller + target profiling app

**Setup**: dev box (laptop) = CONTROLLER running the dashboard; `ws`
(dell-workstation, RTX 2000 Ada) = profiling TARGET over ssh; adapter =
cuvslam. A full-scale validation campaign (~673 cells: 192 configs ×
{plain,nsys,ncu} + 97 nvbit-marked) is running on the target — the demo shows
its live progress plus the committed results from the completed campaigns.

## Start

```bash
cd ~/Projects/cuvslam-stack
./cuvslam_venv/bin/python dashboard/serve.py        # http://127.0.0.1:8642/
```

## Flow (top to bottom of the page)

1. **Card 0 — targets.** Point at the registered `ws` target → *run doctor*.
   The doctor sshes in and prints PASS/WARN/FAIL per layer with a concrete fix
   per finding — it live-catches the real trap on this machine: bare-shell
   `ncu 2025.3 rejects driver 575.64.05 → fix: export NCU_BIN=…cuda-12.9 ncu`
   (which the harness scripts set automatically). Add a fictional Jetson
   target on stage to show the fleet story (ws, gaming laptop, orin, nano, agx).
   Optionally run the doctor on `local` too — it flags the laptop's corrupt
   `libnvJitLink.so.13` with the exact one-line unblock.

2. **Card 1 — new dataset → configs.** Pick a preset (validated base config as
   template), name it, tick variants → TOML files appear in `configs/custom/`.

3. **Card 1b — ANY GPU workload.** Register a command (e.g.
   `python3 examples/gpu_workloads/torch_matmul.py`), a QoR regex
   (`checksum=([0-9.eE+-]+)`), tick NVBit → the same harness profiles a
   non-cuVSLAM codebase (adapter = command; NVTX ranges become stages).

4. **Card 2 — run on a target.** Config + profiler (none/nsys/ncu/nvbit/
   regime) + target `ws`. The config is shipped over scp, the job runs in the
   target repo, stdout streams back live into the job log below the button.
   *During the demo, prefer `--check`-style fast actions or the `local`
   target — the ws GPU is running the campaign. If a live profiled run on ws
   is wanted: pause the campaign first (`pkill -f validation_regime.sh` on
   ws), demo, then relaunch — it resumes from its ledger.*

5. **Card 2b — edit files directly.** Load any `configs/**.toml`, edit (e.g.
   flip a base's `[profiling] nvbit = true` — the deep-trace knob), save.

6. **Card 3 — results.** The embedded site: accuracy matrix vs the paper
   (141 runs, trajectory grids), profiler-neutrality (nsys/ncu bit-identical,
   NVBit ≤2 mm), 192-variant coverage campaign, substrate candidacy
   (GPU/CPU/PiM/ISP per kernel + the 25 verdict flips with driving metrics).

## Live campaign status (talking point)

```bash
ssh ndpvslam@dell-workstation tail -5 ~/validation_regime.log
ssh ndpvslam@dell-workstation 'wc -l /mnt/data/validation_regime_out/REGIME.tsv'
```

Every cell so far: Δ=0 vs plain (profiling is accuracy-neutral, cell by cell).

## If something breaks

- Tailscale ssh may demand re-auth (browser link) — do it before the demo.
- Dashboard port busy: `--port 8643`.
- The doctor IS the fallback content: run it on any machine and walk through
  the findings — that's the product story (an evening of driver/CUDA/kernel
  debugging turned into one actionable line each).
