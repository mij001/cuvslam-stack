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

## Flow — three steps + findings (top to bottom)

1. **Step 1 — Target.** Select `ws` → *doctor — is it ready?* The doctor
   sshes in and prints PASS/WARN/FAIL per layer with a concrete fix per
   finding — it live-catches the real trap on this machine: bare-shell
   `ncu 2025.3 rejects driver 575.64.05 → fix: export NCU_BIN=…cuda-12.9 ncu`
   (which the harness scripts set automatically). Optionally run it on
   `local` too — it flags the laptop's corrupt `libnvJitLink.so.13` with the
   exact one-line unblock. The collapsed "add a target" form shows the fleet
   story (gaming laptop, Jetson Orin/Nano/AGX).

2. **Step 2 — Config.** The dropdown holds the validated study set (bases +
   profiling workloads) + your own. Creation is two small collapsed forms:
   *dataset config* (name + preset + path) and *ANY GPU workload* (name +
   command + QoR regex, e.g. `python3 examples/gpu_workloads/torch_matmul.py`
   with `checksum=([0-9.eE+-]+)`) — adapter model: cuVSLAM or command.
   "Edit config files directly" opens the in-UI editor.

3. **Step 3 — Profile.** One button: *start on target* — default mode is the
   full pipeline (nsys → ncu → nvbit → analyses), accuracy/QoR evaluated on
   every run. Logs stream back live over ssh.
   *During the demo prefer the `local` target or fast-fail actions — the ws
   GPU is running the campaign. For a live ws run: pause the campaign
   (`pkill -f validation_regime.sh`), demo, relaunch — it resumes from its
   ledger.*

4. **Findings (the point of the app).** In one place: PiM/ISP substrate
   candidacy per kernel + the cross-workload verdict flips with their driving
   metrics, memory-space attribution, kernel taxonomy, rooflines,
   PiM-affinity time share, recent pipeline runs. Accuracy + neutrality
   validity is there but collapsed — checked on every run, not the headline.

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
