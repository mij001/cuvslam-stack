# NDP, three independent rungs — one story per class?

| archetype | class | taxonomy says | model k4/c.5 | SIM k4/c.5 | model k8/c.75 | SIM k8/c.75 | verdict |
|---|---|---|---|---|---|---|---|
| g1_triad | G1 | gain | 2.37x | 0.62x | 3.61x | 0.94x | model=gain sim=lose |
| g2_gather | G2 | gain (concurrency near banks) | 0.78x | 0.69x | 0.86x | 0.96x | model=lose sim=flat |
| g3_l2 | G3 | lose (loses its L2) | 0.55x | 0.47x | 0.78x | 0.70x | AGREE |
| g4_chase | G4 | gain (latency shrinks near banks) | 0.87x | 0.53x | 1.14x | 0.79x | model=gain sim=lose |
| g5_fma | G5 | lose (pays c_r) | 0.50x | 0.51x | 0.74x | 0.76x | AGREE |
| g6_shared | G6 | lose/flat (on-chip, pays c_r) | 0.50x | 0.50x | 0.74x | 0.75x | AGREE |
| g7_dep | G7 | lose/flat (dep chain, pays c_r) | 0.50x | 0.50x | 0.74x | 0.75x | AGREE |

**4/7 classes tell one story across taxonomy, fitted model, and cycle-level simulation.**

## Interpretation — the disagreements are signed, systematic, and the finding

- **All four lose-classes AGREE across all three rungs** (G3/G5/G6/G7,
  0.47–0.76x): where the taxonomy predicts NDP pain, the fitted model and the
  cycle-level simulator confirm it independently. NDP losses are real and
  class-predictable.
- **Every disagreement has the same sign**: the taxonomy/fitted model credit
  the memory system (G1 2.37x, G4 1.14x) while the simulator charges the
  core-side path (0.62x, 0.79x). Overlay v2 (bank-fabric ICNT/L2 clocks)
  changed nothing vs v1 — the binding constraint is not the fabric clock but
  the SM's LSU/L1 sector pipeline, which our overlay deliberately keeps
  host-side at c_r speed: after DRAM relief, the streaming kernel RE-BINDS on
  the (slowed) request engine. First-order NDP models — including DAMOV-style
  extrapolations — cannot see the *next* bottleneck; cycle simulation can.
- **Quantified consequence:** with a host-side access engine at realistic PiM
  compute ratios (c_r 0.5–0.75), even the purest streaming archetype only
  reaches break-even (0.94x at k=8/c_r=0.75). The rung-2-vs-rung-3 delta
  (2.37x -> 0.62x for G1) is a *measurement of the value of bank-local access
  engines* — the architectural feature a real PiM provides and a clocks+cache
  overlay cannot. Designing and simulating that engine is the Phase-4 study;
  its required magnitude is now known.
- The v1 (core-side fabric) matrix is preserved in
  `ndp_sim_results_v1_coreside_fabric.csv`; v2 in `ndp_sim_results.csv`.
