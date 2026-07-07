# configs/ — the single config tree: human bases + scripted mutations

The **human owns `base/`** — one full-featured SLAM config per
dataset-sequence × sensor modality (odometry-only where SLAM does not apply,
e.g. mono), each with an `[eval]` block and a `[profiling]` marker. **Every
other config is a mutation of a base**, produced by
`scripts/mutate_configs.py` — the mutation matrix is defined in exactly one
place, and the campaign (`scripts/validation_regime.sh`) runs accuracy AND
profiling over all of it.

| directory | what | committed? | produced by |
|---|---|---|---|
| `*.toml` (this level) | hand-written example configs, one per input source / mode — the runner's documented API (incl. `workload_torch_matmul.toml`, the profile-any-GPU-codebase example) | yes | by hand |
| `base/` | 65 canonical bases (11 KITTI + 33 EuRoC stereo/inertial/mono + 11 TUM fr3 + 8 ICL + 2 TUM-VI) | yes | human-owned (`scripts/gen_base_configs.py --root /mnt/data` bootstraps them from a dataset volume) |
| `generated/` | every derived variant, FLAT: pipeline kinds (`_odom` everywhere, `_async` KITTI, `_cpu` TUM RGB-D) + feature toggles (`__<toggle>`) on one representative per modality; `--window START:COUNT` optionally adds bounded-frame `__win` variants | no (regenerate) | `scripts/mutate_configs.py` |
| `profiling/` | hand-tuned steady-state profiling workloads (pre-date the base/mutate regime; kept for the historical device reports) | yes | by hand |
| `custom/` | configs created through the dashboard (new datasets / new GPU workloads) | no (local) | `dashboard/serve.py` |

**Deep-profiling selection lives in the configs themselves**: a
`[profiling] nvbit = true` block marks the runs that also get the expensive
NVBit memory-trace leg. Every base carries the marker (edit it per config —
it is the human's knob); the mutator keeps it on mutations of proven-important
toggles and drops it elsewhere (~bases + important toggles ≈ 30% of the matrix).

Reproducibility chain (each step verified by byte-identical regeneration):

```
dataset volume ──gen_base_configs.py──▶ configs/base   (65, committed, human-owned)
configs/base  ──mutate_configs.py────▶ configs/generated  (127 mutations, flat)
matrix (192) ──validation_regime.sh──▶ accuracy + nsys + ncu (+ nvbit where marked)
```

The mutator is also what the dashboard imports for its custom-config variants,
so a config mutated anywhere in the stack goes through the same transforms.
A `[workload]` table in any config switches the harness to the command adapter
(`profiling/adapters.py`) — that is how non-cuVSLAM GPU codebases run through
the identical pipeline.
