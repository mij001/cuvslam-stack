# Project Handoff — cuVSLAM Memory Characterization for PiM/ISP

**You are a fresh Claude agent (new account) picking up this project. Read this
first.** The git repo is the source of truth and is fully pushed; but the
*operational knowledge* below lived in the previous agent's private file-memory
and does **not** transfer with the repo — it is reproduced here. After reading,
recreate it as your own memory (see §7).

- **Date of handoff:** 2026-07-04
- **Repo:** `github.com/mij001/cuvslam-stack`, branch `main`; this handoff + `docs/agent-memory/` is the final commit (tip of `main`), tree clean, all pushed.
- **Live state:** nothing running. Laptop + workstation idle. Campaign complete.
- **First move on the new account:** `git clone`, read this file top-to-bottom, then seed `docs/agent-memory/` into your memory (§7). Nothing else is needed to be fully current.

---

## 1. What this project is (30 seconds)

Goal (`goal.md`): motivate memory-centric hardware — Processing-in-Memory (PiM)
and In-Storage-Processing (ISP) — for Physical AI, via a rigorous **per-kernel
memory characterization of cuVSLAM** (NVIDIA's production visual-SLAM stack).
Methodology: DAMOV [Oliveira21] adapted to GPUs + Cao23 NCU roofline. Venue
path: ISPASS/IISWC characterization → MICRO/ASPLOS/ISCA/HPCA architecture paper.

**Status: the ISPASS/IISWC-grade characterization is essentially complete.**
Full read-in: `profiling/PROJECT_STATUS.md` (living status), then
`profiling/PUBLISHABILITY.md` (reviewer-issue register — what's closed, what's
open), `profiling/WALKTHROUGH.md` (guided tour), `profiling/METHODOLOGY.md` (how
every number is made). Results in `profiling/reports/`.

## 2. What's done (headline results, all committed)

- **Full-scale campaign** (`reports/2026-07-04_campaign/`): 27 sequences × 4
  datasets (KITTI 00-10, EuRoC MH/V1/V2 ×11, TUM fr3 ×4, TUM-VI), odom+SLAM,
  locked-clock RTX 2000 Ada, 0 failures. Modal class consistency **91%**;
  pooled k-means prefers **k=7–8** (validates the taxonomy); **61% of GPU time
  carries PiM affinity**.
- **Slice-3 locality** (`reports/2026-07-04_slice3_locality/`): first
  architecture-independent locality from real NVBit traces. Front-end reuse CDF
  **flat across 64 KiB→48 MiB** (cache-immune streaming). **KEY result:** the
  address trace **overturned the ncu counter proxy** on the loop-closure scan
  (`st_track` is coalesced converged streaming, not the scattered gather the
  counters implied) → the ISP case re-grounds to session-scale database growth.
- **Rigor**: measured ceilings, ±25% classification sensitivity, ×5/×3 variance
  (CoV 0.14% locked), cold/warm cache bracket, host↔device transfers.
- PUBLISHABILITY issues **1, 2, 4, 7 closed**.

## 3. The tooling (stdlib-only, headless, GPU-optional)

`profiling/` — everything reruns from committed CSVs with no GPU/dataset:
- `harness/profile.py` — nsys/ncu capture wrapper (honors `NCU_BIN`/`NSYS_BIN`).
- `analysis/` — `build_dag · screen · roofline · bandwidth · transfers ·
  variance · classify` (GPU-DAMOV G1–G7 → PiM/ISP) `· compare · cluster ·
  locality` (NVBit reuse distance) `· campaign` (27-seq synthesis) `· make_report`.
- `campaign/` — `gen_configs.py`, `run_campaign.sh`, `ws_slice3_campaign.sh`.
- `env/` — `measure_ceilings.py`, `gen_hw_descriptor.py`, `check_env.sh`, etc.
- `blocked/` — NVBit/Accel-Sim runners + `mem_trace_windowing.patch`.
- `tests/test_analysis.py` — 12 GPU-free tests.

Reproduce the campaign synthesis with no GPU:
```bash
cd profiling && python3 -m analysis.campaign results_ws_campaign \
    --hw hw/dellworkstation_sm89.toml --out /tmp/campaign   # needs the raw derived data (§5)
```

## 4. NEXT STEP (the plan)

**TaggedAllocator + NVTX data-structure attribution** — the next big scientific
upgrade. Turns kernel-level claims ("st_track is memory-bound") into
data-structure-level ones ("the keyframe database belongs in ISP", by observing
*which allocation* each kernel reads). Feasible now: the cu12 cuVSLAM **source
tree is on the workstation** at `~/Projects/cuvslam_src_cu12` (the wheel was
built from it), and NVBit + a `cuMemAlloc` hook work on the 575 stack. After
that: Accel-Sim NDP config + AccelWattch energy → PiM/ISP substrate design
(Phase 4 = the architecture paper). Open decision for the user: repo LICENSE.

## 5. Environment & machine access (THE crown jewels — not in git elsewhere)

### Laptop (this host, `iNOMAL`, CachyOS, GeForce MX450)
Development + CPU analysis. **CUDA install is corrupt** (118 zero-filled files
from a 2026-07-01 disk-full pacman upgrade, incl. libnvJitLink.so.13,
libnvrtc.so.13). `import cuvslam` fails ("invalid ELF header") without the
workaround: prefix workload commands with
`LD_LIBRARY_PATH=$HOME/.local/cuda-repair/lib` (clean .so copies extracted from
the pacman cache, with soname symlinks). Permanent fix (needs the user's sudo):
`sudo pacman -S cuda`, then delete `~/.local/cuda-repair`. Verify:
`LC_ALL=C pacman -Qkk cuda | grep -c "SHA256 checksum mismatch"` → 0.

### Workstation (`ssh ndpvslam@dell-workstation`, RTX 2000 Ada, the results box)
- **Access:** key auth from the laptop; passwordless sudo (`/etc/sudoers.d/ndpvslam`).
  SSH is **Tailscale SSH** and periodically demands re-auth (prints a
  `login.tailscale.com/...` URL on *every* connection until the user
  re-authorizes) — a hard blocker only the user can clear.
- **Unified 575 stack (learned the hard way):** driver **575.64.05 + CUDA 12.9.1**,
  booting **linux-lts 6.12.39** (GRUB default). This unblocks NVBit (needs
  driver ≤575). BUT the CUDA-13 Nsight tools reject a 575 driver — so ncu must
  be the CUDA-12.9 **ncu 2025.2.1.3** (from NVIDIA's public CUDA redist,
  extracted to `~/ncu2025/`; `profile.py` uses `NCU_BIN`) and nsys 2025.3.2.
  Now NVBit + ncu 2025.2 + nsys 2025.3.2 all coexist on one driver. Revert to
  610 via `~/driver-rollback/revert.sh` (packages in pacman cache) only if
  needed (would re-break NVBit).
- **cuVSLAM venv:** `~/Projects/cuvslam-stack/cuvslam_venv` holds a **cu12 wheel
  rebuilt from source** (cu13 original at `cuvslam_venv_cu13`). Rebuild recipe:
  podman image `cuvslam-wheel-builder` (CUDA 12.6 base — host gcc16 breaks
  nvcc 12.9). mem_trace must also build in that container; NVBit needs
  `PATH=/opt/cuda/bin` at runtime (nvdisasm).
- **Clocks:** lock with `sudo nvidia-smi -pm 1; -lgc 1620,1620; -lmc 7001,7001`
  (measured ceilings at lock: 205.0 GB/s DRAM, 5445 GFLOP/s FP32). **Locks
  reset on reboot — re-apply after every power cut.**
- **GPU cleanup:** the user's `~/free_gpu.zsh` kills the KDE compositor (BIOS
  boots on AC + KDE autologin, so a power cut leaves KDE on the GPU — it perturbs
  ncu/nsys captures). `~/restore_gui.zsh` brings it back. Run free_gpu before
  captures; note it resets clocks, so re-lock after.
- **Datasets:** sda2 mounted ro at `/mnt/data` (fstab) — KITTI color 00-21,
  EuRoC ×11 (`/mnt/data/EuRoC/{MH,VR1,VR2}`), TUM fr3 ×4, TUM-VI tars. Also
  `~/tumvi_extracted/`. Campaign configs use `${CUVSLAM_DATA2}`→/mnt/data.
- **RULE: long remote jobs MUST run under `setsid nohup … &`** — power cuts and
  SSH drops have killed several. Survive by design.

## 6. Data locations

- **Committed** (reproduce findings, no GPU): `profiling/reports/**`,
  `profiling/results_ws/**` (first matrix), per-sequence campaign classifications.
- **Gitignored, on-disk only:** `profiling/results/` (local raw),
  `profiling/results_ws_campaign/` (35 MB campaign derived — on the laptop;
  needed to re-run `analysis.campaign` from scratch). Raw `.ncu-rep`/`.nsys-rep`
  and the GB-scale NVBit traces (`~/slice3/*.zst`) stay on the workstation.

## 7. For you, the incoming agent — recreate the memory

The previous agent's file-memory is copied verbatim into `docs/agent-memory/`
(minus account-specific frontmatter). Read those, then write them into your own
memory system so the cross-session thread survives. The three notes:
`workstation-access`, `cuda-corruption-ld-library-path`,
`cuvslam-stack-profiling-state`. Keep them updated as you work — especially
after any reboot (clocks), power cut, or Tailscale re-auth.
