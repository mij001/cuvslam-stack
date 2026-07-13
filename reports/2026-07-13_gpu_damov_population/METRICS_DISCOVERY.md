# Metric discovery — does anything separate the classes better?

307 population kernels. Per-feature one-way F-statistic (higher = separates the classes more):

| feature | F |
|---|---|
| dram_sol | 169.6 |
| mem_sol | 84.4 |
| occ | 53.0 |
| l2amp | 49.8 |
| comp_sol | 44.9 |
| s_mem | 43.6 |
| dsa | 29.5 |
| lfmr | 25.0 |
| s_core | 24.4 |
| log_sect | 5.8 |
| qsr | 0.3 |

k-means ablation (does adding candidates sharpen unsupervised recovery of the classes?):

| feature set | n | purity@k=8 | silhouette |
|---|---|---|---|
| BASE (current classifier inputs) | 307 | 0.717 | 0.278 |
| BASE + QSR + L2AMP (new static) | 307 | 0.635 | 0.304 |
| BASE + response (S_core,S_mem) | 293 | 0.747 | 0.291 |
| ALL | 293 | 0.696 | 0.322 |
