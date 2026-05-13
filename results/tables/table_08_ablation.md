## Table 8. Ablation Checklist

| Component | Variant tested | Effect on σ | Effect on purity_gain | Conclusion |
|---|---|---|---|---|
| PH filtration | VR vs Alpha | Median diff < 0.02 | < 0.005 | VR and Alpha strongly correlated; VR-H₀ chosen as primary |
| Homology dim | H₀ vs H₁ | H₁ lower (~0.3×) | Smaller | H₀ captures more signal; H₁ adds diagnostic value |
| Subsample size | N=200 vs 400 vs 800 | < 3% variation | < 0.01 | N=400 stable; larger N adds cost with minimal gain |
| Random-label ctrl | 1 run vs 3 runs (mean) | < 0.01 | — | Single run sufficient; stochasticity in σ is low |
| MLP depth | 2-layer vs 3-layer | +0.05 median σ | +0.01 | 3-layer gives richer topological trajectory |
| BN removal | No BatchNorm | +0.12 on amplifiers | −0.02 | BN stabilises topology; critical for Archetype III |
| Architecture | Same 3-layer for all | Domain variation seen | Preserved | Fixed arch allows fair cross-domain comparison |
