# configs/ — the single config tree

Every TOML the stack consumes lives here. One file = one fully-described run
(see the TOML reference in the top-level README).

| directory | what | produced by |
|---|---|---|
| `*.toml` (this level) | hand-written example configs, one per input source / mode | maintained by hand |
| `accuracy_matrix/` | the 141-run accuracy matrix vs the cuVSLAM paper (dataset × sensor variant × pipeline mode) | `scripts/gen_accuracy_configs.py` |
| `profiling_coverage/` | 192 feature-toggle variants derived from the accuracy matrix (accuracy-under-profiling campaign) | `scripts/gen_profiling_coverage.py` |
| `profiling/` | hand-tuned profiling workloads (steady-state windows) for the nsys/ncu harness | maintained by hand |
| `campaign/` | characterization-campaign configs (odom+slam per sequence, `${CUVSLAM_DATA2}` paths) — generated, not committed | `profiling/campaign/gen_configs.py` |
| `custom/` | configs created through the dashboard (new datasets) — local, not committed | `dashboard/serve.py` |

The generated trees (`accuracy_matrix/`, `profiling_coverage/`) are committed so
the exact campaigns are reproducible as-run; the generators remain the source of
truth — regenerate against your own data root with the commands in the top-level
README (§ Full workflow).
