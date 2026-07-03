# results_ws/ — workstation capture data (derived tables only, committed on purpose)

Mirrors the `derived/` CSVs + `metadata.json` of every locked-clock capture on
`dell-workstation` (RTX 2000 Ada). Unlike `results/` (local, gitignored, holds
multi-hundred-MB raw `.nsys-rep`/`.ncu-rep`), these are the small portable
tables the analysis layer consumes — committing them makes every report,
comparison, and clustering result in `reports/` reproducible **from the repo
alone** (no GPU, no datasets, no workstation):

```bash
cd profiling
python3 -m analysis.classify results_ws/<ncu_run> --hw hw/dellworkstation_sm89.toml --out /tmp/x
python3 -m analysis.compare  tum=reports/2026-07-03_tum_office_rtx2000ada/data/classification.csv ...
```

Raw profiler reports stay on the workstation at
`~/Projects/cuvslam-stack/profiling/results/` (same dir names).
