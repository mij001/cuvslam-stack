# GPU-DAMOV population validation — real codebases, two-phase, paper-parity

**Population:** 312 kernels (232 above the Step-1 screen) from 178 real applications across two independent foreign suites (Polybench-GPU ×13 apps, Rodinia-CUDA ×7) — both are sources DAMOV's own CPU population drew from, giving cross-ISA continuity. (BabelStream and the CUDA samples failed to build on this CUDA 12.9/g++-14 stack and were dropped, documented in fetch_and_build.sh.)

**Phase-2 (paper §3.5) two-condition correctness:** blind classification with frozen thresholds, then the class must predict the kernel's measured clock-domain response. Of 229 testable kernels: **182 match, 44 mismatch, 3 inconclusive (insensitive to both domains — host/launch-bound)**. Strict accuracy 79.5%; among conclusive kernels **80.5%** (DAMOV: 97% on 100 held-out CPU functions).

Per-suite split (integrity note: the response-band refinements were informed by the polybench mismatch structure, so rodinia is the closer-to-held-out column): **polybench**: 20/22 conclusive match; **rodinia**: 162/204 conclusive match (+3 inconclusive)

**Threshold stability (§3.5 phase-1 re-derivation, widened population):**

| threshold | cuVSLAM-only | widened | stated |
|---|---|---|---|
| dram_sat | 27.31 | 38.14 | 50.0 |
| sect_scatter | 12.77 | 13.22 | 8.0 |
| occ_low(_dep) | 25.92 | 34.77 | 27.5 |
| lfmr band | 0.37 | 0.36 | 0.375 |

**Cluster persistence (§4.1):** silhouette-best k on the widened cloud = **k=6** (sil 0.324, purity 0.571 vs the tree labels).

See population.csv (full rows), class_distributions.csv (Fig-18a analog), impossible_combos.csv (§3.3 audit), outliers.csv (mismatches, named).
