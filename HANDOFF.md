# Project Handoff — cuVSLAM Memory Characterization for PiM/ISP

**You are the incoming Claude agent picking this project up (this is a
hand-*back* to the account that originally handed it off, 2026-07-04 → work
done on another account → returning 2026-07-06).** The git repo is the source
of truth and is fully pushed; the *operational knowledge* below lived in a
private per-project file-memory that does **not** transfer between accounts, so
it is reproduced here and mirrored in `docs/agent-memory/`. After reading, seed
those notes into your memory (§7).

- **Date of this handoff:** 2026-07-06
- **Repo:** `github.com/mij001/cuvslam-stack`, branch `main`, tree clean
  (only untracked: `Qs.txt`, `main.pdf` — user's own scratch/paper, see §8),
  all commits pushed. HEAD = `031f95d` ("docs: fold accuracy validation …").
- **Live state:** nothing running. Laptop + workstation idle. All campaigns
  complete. A safety branch `backup/pre-accuracy-audit-20260706` exists (local
  only) marking the pre-accuracy-work point.
- **First move:** `git pull`, read this file, seed `docs/agent-memory/` (§7).
  Then read `docs/THESIS_FINDINGS.md` (the master findings + roadmap) and
  `docs/DEFENSE_PRIMER.md` (zero-background explainer of the whole project).

---

## 1. What this project is (30 seconds)

Goal (`goal.md`): motivate memory-centric hardware — Processing-in-Memory (PiM)
and In-Storage-Processing (ISP) — for Physical AI, via a rigorous **per-kernel,
per-data-structure memory characterization of cuVSLAM** (NVIDIA's production
visual-SLAM stack). Methodology: DAMOV [Oliveira21] adapted to GPUs + Cao23 NCU
roofline. Venue path: ISPASS/IISWC characterization → MICRO/ASPLOS/ISCA/HPCA
architecture paper.

**Status: the characterization is COMPLETE, at data-structure granularity,
and accuracy-validated against NVIDIA's paper.** The two master documents are
new since the original handoff and are the best entry points:
- `docs/THESIS_FINDINGS.md` — findings **F1–F13**, what they license, the
  ranked reviewer-gap register (G1–G10), and three step-by-step paths forward.
- `docs/DEFENSE_PRIMER.md` — everything from zero (SLAM, GPU memory spaces,
  the statistics, every finding retold slowly, an 18+1-question hostile-panel
  drill, glossary). Written for a 2nd-year undergrad to defend the thesis.

Living status/registers: `profiling/PROJECT_STATUS.md`,
`profiling/PUBLISHABILITY.md`, `profiling/WALKTHROUGH.md`,
`profiling/METHODOLOGY.md`. Reports in `profiling/reports/`.

## 2. What's done since the original handoff (all committed & pushed)

The original handoff left off at "characterization essentially complete, next =
TaggedAllocator attribution." That, and more, is now done:

- **TaggedAllocator + NVTX data-structure attribution (M7)** —
  `reports/2026-07-04_attribution/`. Instrumented cu12 wheel (RelWithDebInfo,
  `patches/0002-tagged-allocator-nvtx.patch`, USE_NVTX=ON), NVBit alloc-event
  sidecar (`profiling/blocked/mem_trace_alloc_events.patch`), and
  `profiling/analysis/attribution.py` (resolve+join). **Result:** GPU memory
  budget is *static* (240 allocs, 108.65 MB; keyframe state a fixed 6.7 MB →
  the session-scale DB grows host-side in LMDB). Measured NVTX kernel→stage map.
- **Full-matrix attribution campaign** — `reports/2026-07-05_attribution_campaign/`.
  All 27 SLAM sequences, coverage-audited and gap-filled to **0 missing
  kernels**; **48/49 kernels have a unanimous top data-structure tag** across
  every sequence → composition is a kernel property, not a workload property.
- **Slice-3 CORRECTION (important — a headline claim reversed)** —
  `reports/2026-07-04_slice3_locality/FINDINGS.md §5` + `data_v2/`. The
  original handoff said "the trace overturned the ncu counter proxy; st_track
  is coalesced streaming." **That was wrong and is withdrawn.** The attribution
  join revealed 94% of st_track's accesses are register spill (coalesced by
  construction); the space-filtered re-derivation shows its *global* accesses
  are a **scattered gather (23–30 sectors/warp) — matching the ncu counters.**
  Two methods now agree; the G2-scatter label stands. `analysis/locality.py`
  gained `--spaces {global,shared,local,all}` (default global). Front-end
  streaming claim *strengthened* (conv scatter was shared tiles).
- **Accuracy validation vs the cuVSLAM paper** — `reports/2026-07-06_accuracy/`.
  Full **104-run config matrix** (all datasets × stereo/inertial/mono/RGBD ×
  odom/slam/sync/async/GPU/CPU) scored vs ground truth, compared to
  arXiv:2506.04359 Tables 2/6. **The profiled stereo/RGB-D SLAM modes reproduce
  the paper** (EuRoC stereo APE 0.114/0.051 m vs 0.13/0.054; TUM long_office
  beats it; KITTI 500 m-drift 0.82% vs leaderboard 0.85%). **QoR: the
  instrumented wheel is trajectory-identical to baseline** → the memory
  characterization is accuracy-neutral = valid. This is finding **F13**.
- PUBLISHABILITY issues **1, 2, 3, 4, 7, 8 closed**; the two self-correction
  rows (NEW, NEW2) closed with independent methods in agreement.

## 3. The tooling (stdlib-only, headless, GPU-optional)

Everything reruns from committed CSVs with no GPU/dataset. New since the
original handoff, on top of `build_dag · screen · roofline · bandwidth ·
transfers · variance · classify · compare · cluster · locality · campaign ·
make_report`:
- `analysis/attribution.py` — resolve (symbolize alloc backtraces → tags) +
  join (stream trace, memory-space bucket, attribute global accesses;
  `--max-accesses-per-kernel` early-stop). `analysis/attribution_campaign.py`
  — 27-seq cross-sequence synthesis. `analysis/locality.py --spaces`.
- `campaign/ws_attribution_all.sh` + `ws_attribution_gapfill.sh` +
  `plan_gapfill.py` — resumable full-matrix attribution capture with a
  coverage-audit gap-filler (anchors windows on dense launch clusters to beat
  run-to-run launch-id drift).
- Repo root: `gen_accuracy_configs.py` (104 configs), `ws_accuracy_matrix.sh`
  (matrix + QoR), `accuracy_report.py` (paper comparison, convergence-gated).
- `tests/test_analysis.py` — now **18 GPU-free tests** (all pass).

## 4. NEXT STEPS (pick from `docs/THESIS_FINDINGS.md §5`)

The characterization is done; the remaining work is either paper-writing or
substrate-side. Ranked:
1. **Path A — write the ISPASS/IISWC characterization paper NOW.** Evidence is
   complete; §5 lists the figures (all from committed CSVs) and two 1–2-day
   quick adds (NVML whole-run energy; host LMDB I/O characterization).
2. **Path B — Phase 4 architecture paper:** Accel-Sim NDP config (replay the
   committed window traces) + AccelWattch energy → PiM/ISP substrate deltas.
3. **Path C strengtheners:** host-side cold-persistent (LMDB) characterization
   (arms the ISP leg, ~days); Jetson Orin re-run (edge + re-validate the
   codegen-dependent spill split); Layer-3 kernel-arg correlation to name the
   static-memory residuals.
4. **Accuracy follow-ups (low priority, do NOT affect the characterization):**
   G9 — tune EuRoC IMU from each `sensor.yaml`, re-run the 22 inertial configs
   (inertial mode is under-tuned, only 3/11 converge); G10 — undistort TUM-VI
   to pinhole + add a Sim3 alignment path for mono.

**Open decision for the user:** repo LICENSE (blocks artifact evaluation).

## 5. Environment & machine access (THE crown jewels — not in git elsewhere)

### Laptop (this host, `iNOMAL`, CachyOS, GeForce MX450)
Dev + CPU analysis. **CUDA install is corrupt** (zero-filled files from a
2026-07-01 disk-full pacman upgrade). `import cuvslam` fails without the
workaround: prefix workload commands with
`LD_LIBRARY_PATH=$HOME/.local/cuda-repair/lib`. Permanent fix (user sudo):
`sudo pacman -S cuda`, then delete `~/.local/cuda-repair`.

### Workstation (`ssh ndpvslam@dell-workstation`, RTX 2000 Ada, the results box)
- **Access:** key auth; passwordless sudo. SSH is **Tailscale SSH** and
  periodically demands re-auth (`login.tailscale.com/…` URL) — only the user
  can clear it.
- **Unified 575 stack:** driver **575.64.05 + CUDA 12.9.1**, linux-lts 6.12.39.
  NVBit works here; the CUDA-13 Nsight tools reject 575, so ncu is the CUDA-12.9
  **ncu 2025.2.1.3** at `~/ncu2025/` (`profile.py` honors `NCU_BIN`) and nsys
  2025.3.2. NVBit + ncu 2025.2 + nsys coexist on one driver.
- **Two venvs:** `cuvslam_venv` = baseline cu12 wheel; **`cuvslam_venv_tagged`**
  = instrumented wheel (TaggedAllocator+NVTX, `patches/0002`, RelWithDebInfo).
  `cuvslam_src_cu12` has patch 0002 applied (`.orig` backups); baseline wheel
  at `~/Projects/cuvslam_src_cu12/dist/baseline/`. Rebuild in podman image
  `cuvslam-wheel-builder` (host gcc16 breaks nvcc 12.9); NVBit/mem_trace also
  build there; NVBit needs `PATH=/opt/cuda/bin` at runtime.
- **Clocks:** `sudo nvidia-smi -pm 1; -lgc 1620,1620; -lmc 7001,7001` (ceilings
  at lock: 205.0 GB/s, 5445 GFLOP/s). **Reset on reboot — re-apply.**
- **GPU cleanup:** `~/free_gpu.zsh` kills the KDE compositor before captures
  (BIOS boots on AC + KDE autologin); `~/restore_gui.zsh` restores. free_gpu
  resets clocks — re-lock after.
- **sda2 write access (learned 2026-07-06):** `/mnt/data` is fstab-**ro** and
  the NTFS volume has a dirty bit (`ntfsfix` not installed). To write:
  `sudo -n umount /mnt/data; sudo -n mount -t ntfs3 -o rw,force /dev/sda2 /mnt/data`
  (all campaign scripts re-assert this; re-apply after reboots).
- **Datasets:** `/mnt/data` (KITTI, EuRoC, TUM_RGBD/extracted, tumvi) + also
  `~/Projects/cuvslam_datasets/`.
- **RULE: long remote jobs MUST run under `setsid nohup … &`** — power cuts and
  SSH drops have killed several.

## 6. Data locations

- **Committed** (reproduce findings, no GPU): `profiling/reports/**` (incl. the
  three new report dirs), `profiling/results_ws/**`, `accuracy_report.py`
  results.
- **Gitignored / on-disk only:** `profiling/results_ws_campaign/` (35 MB, on
  the laptop; needed to re-run `analysis.campaign`); on the workstation sda:
  `/mnt/data/attribution_out/` (~85 GB, 27-seq attribution raw+joins),
  `/mnt/data/accuracy_out/` (104-run matrix + QoR + results), `~/slice3/*.zst`
  (GB-scale NVBit traces). The committed reports reproduce every finding.

## 7. Recreate the memory (do this after reading)

`docs/agent-memory/` holds the portable copy of the private file-memory (kept
current as of 2026-07-06). Read and seed all three notes:
`workstation-access` (with the 2026-07-06 sda/venv addendum),
`cuda-corruption-ld-library-path`, `cuvslam-stack-profiling-state` (now says:
characterization + attribution + accuracy all done, next = Phase 4). Keep them
updated after any reboot (clocks), power cut, or Tailscale re-auth.

## 8. Loose ends / untracked files

- `Qs.txt` (repo root, untracked) — the user's questions about the
  instrumentation; all three are answered inline in the file itself (NVTX
  double-gating, TUM canonical-vs-campaign loop coverage, why patch NVBit).
  Not committed (user scratch).
- `main.pdf` (repo root, untracked, 312 KB) — the project's paper/proposal PDF.
  Not committed. If it should be version-controlled, ask the user first.
- `backup/pre-accuracy-audit-20260706` — a local safety branch at the point
  before the accuracy work; delete once you're confident `main` is good.
