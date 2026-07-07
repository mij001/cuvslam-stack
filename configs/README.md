# configs/ — the single config tree: bases + mutations

One canonical **base** config per dataset-sequence × sensor modality lives in
`base/` (a full-featured SLAM config with `[eval]`, or odometry-only where SLAM
does not apply, e.g. mono). **Every other config is a mutation of a base**,
produced by `scripts/mutate_configs.py` — the mutation matrix is defined in
exactly one place.

| directory | what | committed? | produced by |
|---|---|---|---|
| `*.toml` (this level) | hand-written example configs, one per input source / mode — the runner's documented API | yes | by hand |
| `base/` | 65 canonical bases (11 KITTI + 33 EuRoC stereo/inertial/mono + 11 TUM fr3 + 8 ICL + 2 TUM-VI) | yes | `scripts/gen_base_configs.py --root /mnt/data` |
| `generated/accuracy/` | the 141-run accuracy matrix: base + `_odom` + `_async` (KITTI) + `_cpu` (TUM) pipeline mutations | no (regenerate) | `scripts/mutate_configs.py --select accuracy` |
| `generated/coverage/` | 192 feature-toggle variants: every accuracy config as `__base` + toggle mutations on one representative per modality | no (regenerate) | `scripts/mutate_configs.py --select coverage` |
| `generated/window/` | bounded-frame captures for kernel-level profiling (`__win<START>x<COUNT>`) | no (regenerate) | `scripts/mutate_configs.py --select window --window 200:260` |
| `profiling/` | hand-tuned steady-state profiling workloads (pre-date the base/mutate regime; kept for the historical device reports) | yes | by hand |
| `custom/` | configs created through the dashboard (new datasets) | no (local) | `dashboard/serve.py` |

Reproducibility chain (each step verified by byte-identical regeneration):

```
dataset volume ──gen_base_configs.py──▶ configs/base (65, committed)
configs/base  ──mutate_configs.py───▶ configs/generated/{accuracy,coverage,window}
```

The mutator is also what the dashboard imports for its custom-config variants,
so a config mutated anywhere in the stack goes through the same transforms.
