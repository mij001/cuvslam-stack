# GPU-DAMOV validation — the DAMOV robustness checks, on our data

## §8.3 analog — cross-microarchitecture agreement
Same workload (TUM office), two GPUs (sm_75 laptop vs sm_89 workstation): **36/47 kernels same class (76.6%)**; excluding launch-tax G0 kernels **36/45 (80.0%)**. The class is a property of the kernel's data movement, not the microarchitecture. -> `reports/2026-07-09_damov_validation/cross_device_agreement.csv`

## §4.1 analog — independent algorithm (Ward hierarchical clustering)
Pooled feature cloud (47+ kernels x 4 device reports, features: mem_sol, comp_sol, dram_sol, lfmr, occupancy, sectors_req, stall_long_sb, stall_wait), Ward dendrogram cut at k=8: **ARI 0.31, purity 0.675** vs the decision-tree labels (k-means gave purity 0.68 — two independent algorithms see the same structure). -> `reports/2026-07-09_damov_validation/hierarchical_agreement.csv`

## §3.5 analog — ground-truth calibration (designed kernels, classified blind)
**8/8 archetypes recovered** with frozen thresholds:

| archetype | designed | classified | match |
|---|---|---|---|
| g1_triad | G1-bandwidth | G1-bandwidth | yes |
| g2_gather | G2-coalescing | G2-coalescing | yes |
| g3_l2 | G3-l2-reuse | G3-l2-reuse | yes |
| g4_chase | G4-latency | G4-latency | yes |
| g5_fma | G5-compute | G5-compute | yes |
| g6_shared | G6-onchip | G6-onchip | yes |
| g7_dep | G7-dependency | G7-dependency | yes |
| g0_tiny | screened-Step1 | screened(0.0070ms<0.05) | yes |

## Step-3 analog — clock-domain intervention (real-hardware response test)
**7/7 classes respond as the taxonomy predicts** (core-clock vs memory-clock sensitivity):

| archetype | S_core | S_mem | predicted | observed | verdict |
|---|---|---|---|---|---|
| g1_triad | 1.06 | 1.35 | core∈[0.9,1.2] mem∈[1.2,1.45] — bus-bound: tracks MEM period (<=1.4x), core-insensitive | core=1.06 mem=1.35 | OK |
| g2_gather | 1.18 | 0.97 | core∈[1.05,1.5] mem∈[0.9,1.1] — request-concurrency-bound: MSHR/LSU are core-domain; bus NOT saturated | core=1.18 mem=0.97 | OK |
| g3_l2 | 1.80 | 1.00 | core∈[1.5,2.1] mem∈[0.9,1.1] — L2-resident: the L2 is core-domain | core=1.80 mem=1.00 | OK |
| g4_chase | 1.36 | 1.12 | core∈[1.2,1.6] mem∈[1.02,1.2] — latency = core-domain L2/NoC traversal + mem-domain CAS: mixed, core-leaning | core=1.36 mem=1.12 | OK |
| g5_fma | 2.00 | 1.00 | core∈[1.7,2.1] mem∈[0.9,1.1] — compute: tracks core period (2.0x) | core=2.00 mem=1.00 | OK |
| g6_shared | 2.00 | 1.00 | core∈[1.7,2.1] mem∈[0.9,1.1] — shared memory is core-domain | core=2.00 mem=1.00 | OK |
| g7_dep | 2.00 | 1.00 | core∈[1.7,2.1] mem∈[0.9,1.1] — FMA latency chain is core-domain | core=2.00 mem=1.00 | OK |

