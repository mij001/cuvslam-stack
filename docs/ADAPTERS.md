# Adapters — profiling YOUR codebase with this harness

The harness automates the bottleneck-and-improvement analysis this project did
for cuVSLAM: **where** GPU time goes → **who** dominates → **why** (metrics vs
decision thresholds) → **verdict** (best substrate: GPU / CPU / PiM / ISP).
To run it on your own workload you write an **adapter** — one small Python
class — or, for simple cases, no code at all.

## The split: what the adapter owns vs what the profiler owns

| | ADAPTER (your workload knowledge) | PROFILER (harness — hardware & analysis knowledge) |
|---|---|---|
| launch | `argv()`, `env()`, `cwd()` — a deterministic command | runs it under nsys / ncu / NVBit with bounded launch windows |
| meaning of stages | NVTX ranges in your code (preferred), or nothing (kernels group as one stage) | turns NVTX ranges into the stage DAG + time shares |
| quality of result | `qor()` — extract the number(s) that prove the run computed the right thing | re-runs plain vs profiled and checks your QoR is unchanged (neutrality) |
| metrics | — | curated ncu metric sets (roofline / characterize), memory traces, per-kernel evidence |
| conclusions | — | limiter classification (taxonomy), substrate verdict + rationale |
| output | — | **`summary.json` — the standard schema** every consumer reads (evidence explorer, cross-run dynamics) |

The contract is intentionally thin: the adapter never touches profilers or
analysis; the profiler never needs to understand your codebase.

## Integration — three levels

**Level 0 — no code** (most workloads). Describe the launch in the config:

```toml
# configs/custom/my_dnn.toml
[workload]
cmd = "python3 train.py --data /mnt/data/imagenet --iters 100"
cwd = "/home/me/my_dnn_repo"
[workload.env]
CUDA_VISIBLE_DEVICES = "0"
[qor]
stdout_regex = "final_loss=([0-9.eE+-]+)"   # proves profiling neutrality
[profiling]
nvbit = true                                 # opt into the deep memory-trace leg
```

This uses the built-in `command` adapter. Add NVTX ranges to your code
(`torch.cuda.nvtx.range_push/pop`, or `nvtx3` in C++) and the stage analyses
light up automatically.

**Level 1 — a drop-in adapter file** when extraction logic needs code
(QoR from a file, computed arguments, multi-step launch). Create
`profiling/adapters.d/myproject.py`:

```python
from adapters import Adapter        # resolved by the harness at load time

class MyProjectAdapter(Adapter):
    name = "myproject"              # unique; select with --adapter myproject
                                    # or [workload] adapter = "myproject"
    def argv(self):
        return ["./build/bench", "--config", self.config_path]

    def qor(self):                  # anything comparable run-to-run
        import json
        return json.load(open("results/metrics.json"))
```

It is auto-discovered at import (files starting with `_` are skipped — see
`_example_myproject.py`). A broken drop-in prints a warning and is skipped;
it can never take the harness down.

**Level 2 — upstream adapter.** If the workload family is general (like
`cuvslam`), promote the class into `profiling/adapters.py` via PR.

## What you get back — the standard evidence schema

Every profiled run reduces to `summary.json`
(schema: `profiling/analysis/summarize_run.py`):

```
stages:  [{name, time_ms, share_pct, n_kernels}]
kernels: [{name, stage, time_ms, share_pct,
           limiter,                # taxonomy class (what bounds it)
           substrate,              # GPU / CPU / PiM-near-bank / PiM-scatter / ISP
           rationale, pim_affinity,
           evidence: {dram_sol_pct, sectors_per_req, lfmr, mpki,
                      occupancy_pct, ai, dominant_stall},
           roofline: {ai, gflops}}]
```

The dashboard's **evidence explorer** renders this interactively — stage bar →
dominant-kernel list → per-kernel metric bars against the decision thresholds
(DRAM SoL ≥ 60 %, sectors/request ≤ 8 coalesced / ≥ 16 scattered, occupancy
< 25 %, LFMR ≥ 0.5, MPKI ≥ 30) → the substrate verdict those numbers imply.
Because the schema is adapter-independent, your DNN or image pipeline gets the
identical inspectable reasoning chain the cuVSLAM study used.

## Checklist for a new codebase

1. `scripts/doctor.sh` on the target — fix anything it flags (it knows the
   driver/CUDA/ncu/NVBit traps and prints the fix).
2. Write the Level-0 config (or a drop-in adapter) + a QoR extractor.
3. Add NVTX ranges around your pipeline stages (strongly recommended).
4. Dashboard → step 2 → register; step 3 → *full profile* on your target.
5. Findings → evidence explorer: your run appears in the selector.
