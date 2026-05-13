## Table 7. Seven Topology Archetypes

| Archetype | σ range | purity_gain | TDI scale | H₁/H₀ | n in atlas | Key recommendation |
|---|---|---|---|---|---|---|
| I — Pre-Separated | < 0.35 | ≈ 0 | low | low | 6 | Linear model sufficient |
| II — Compact Cluster | 0.9–1.0 | ≈ 0 | very low | low | 37 | Shallow net or kNN |
| III — Topological Amp. | ≥ 1.45 | > 0.03 | moderate | low | 4 | Deep MLP + BN + GCN lift |
| IV — High-D Sparse | 0.8–1.1 | > 0.18 | very high | moderate | 8 | Dim-reduce first, then MLP |
| V — Loop-Rich | 0.7–1.1 | variable | moderate | ≥ 0.35 | 10 | RBF/Fourier features + Alpha |
| VI — Curse-of-Dim | ≈ 1.0 | ≈ 0 | extreme | low | 0 | Feature selection critical |
| VII — Noisy Isotropic | 0.92–1.18 | < 0.015 | variable | low | 34 | Feature engineering over arch. |
| Other | — | — | — | — | 273 | — |
