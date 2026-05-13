# The TDI Atlas

**A Cross-Domain Empirical Study of Topological Deformation in Neural Network Representations**

David Vesterlund · Independent Research · VesterlundCoder · May 2026

---

## Overview

The TDI Atlas maps the **Topological Deformation Index (TDI)** across **373 datasets spanning 25 scientific domains** — biology, physics, finance, NLP, ecology, robotics, vision, speech, software engineering, and mathematics. For each dataset, a standardised 3-layer MLP is trained and the topological structure of its layer-wise representations is measured using persistent homology (Vietoris-Rips and Alpha complex, H₀ and H₁).

The result is the first large-scale cross-domain topology fingerprint database, enabling systematic comparison of how much different types of data force neural networks to reshape geometric structure during learning.

### Key metrics per dataset
| Metric | Description |
|---|---|
| `tdi_vr_h0` | Wasserstein TDI summed across layers, VR filtration, H₀ |
| `tdi_alpha_h0` | Same, Alpha complex |
| `signal_ratio` σ | `TDI_rand / TDI_trained` — how much label structure drives topology |
| `purity_gain` | `final_kNN_purity − input_kNN_purity` — topology learned vs raw |
| `input_entropy_h0/h1` | Persistence entropy of raw input |

---

## Main Findings

### signal_ratio distribution (373 datasets)
- **Median σ = 0.87** — most networks *simplify* input topology rather than amplify it
- **σ < 0.3** (pre-separated datasets): classes are near-linearly separable in raw feature space; linear models sufficient
- **σ > 1.5** (topological amplifiers): network must *generate* structure absent in the input — depth and batch normalisation critical
- **66%** of all datasets have σ < 1.0

### Top purity_gain datasets
| Dataset | Domain | purity_gain | Accuracy |
|---|---|---|---|
| isolet | speech | 0.394 | 0.963 |
| cnae | nlp_features | 0.343 | 0.919 |
| gina_agnostic | synthetic | 0.272 | 0.896 |
| mnist_784 | vision | 0.250 | 0.942 |
| har | robotics | 0.233 | 0.984 |

### Domain-level signal_ratio (selected)
| Domain | Avg σ | Interpretation |
|---|---|---|
| mathematics | 0.636 | Highly natural geometry |
| biology | 0.718 | Pre-structured features |
| nlp_features | 0.811 | Distributional requires reorganisation |
| vision | 1.038 | Moderate amplification |
| robotics | **1.272** | Strongest topological amplification |

---

## Seven Topology Archetypes

From the atlas we identified seven canonical geometric patterns that recur across unrelated domains. Each implies a specific strategy for small-model architecture selection:

| Archetype | σ range | purity_gain | Key recommendation |
|---|---|---|---|
| I — Pre-Separated Manifold | < 0.3 | ≈ 0 | Linear model sufficient |
| II — Compact Low-Dim Cluster | 0.9–1.0 | ≈ 0 | Shallow network or kNN |
| III — Topological Amplifier | > 1.5 | > 0.04 | Deep MLP + BN + GCN lift |
| IV — High-D Sparse Manifold | 0.8–1.1 | > 0.20 | Dim-reduce first, then MLP |
| V — Loop-Rich Geometry | 0.7–1.1 | variable | RBF/Fourier features, Alpha-complex |
| VI — Curse-of-Dim Blob | ≈ 1.0 | ≈ 0 | Feature selection critical |
| VII — Noisy Isotropic | 0.95–1.15 | < 0.01 | Feature engineering over architecture |

See **Section 6.7** of `paper.md` for full descriptions, atlas examples, and detailed small-model guidance for each archetype.

---

## Repository Structure

```
tdi-atlas/
├── paper.md                    # Full research paper
├── tdi_atlas.py                # Main atlas pipeline (all 373 datasets)
├── visualize_atlas.py          # PCA/UMAP/histogram figures
├── download_tda_datasets.py    # Dataset downloader (OpenML + sklearn)
├── scan_topological_datasets.py # Dataset scanner / metadata collector
├── requirements.txt
└── results/
    ├── tdi_atlas_400.json      # Full 373-dataset atlas (JSONL, one record per line)
    ├── tdi_atlas_400.csv       # Flat CSV version
    └── figures/
        ├── fig_domain_signal_ratio.png
        ├── fig_signal_ratio_hist.png
        ├── fig_acc_vs_signal.png
        └── fig_top_purity_gain.png
```

---

## Quickstart

```bash
git clone https://github.com/VesterlundCoder/tdi-atlas.git
cd tdi-atlas
pip install -r requirements.txt

# Fast smoke test (~5 min, 10 datasets)
python tdi_atlas.py --fast --out results/tdi_atlas_fast.csv

# Full atlas (~2h, requires openml)
python tdi_atlas.py --out results/tdi_atlas.csv

# Generate figures
python visualize_atlas.py --atlas results/tdi_atlas.csv
```

---

## Atlas JSON Format

Each line of `results/tdi_atlas_400.json` is a JSON record:

```json
{
  "dataset": "isolet",
  "domain": "speech",
  "n_samples": 7797,
  "n_features": 617,
  "n_classes": 26,
  "accuracy": 0.963,
  "tdi_vr_h0": 6571.4,
  "tdi_vr_h1": 312.8,
  "tdi_alpha_h0": 5103.2,
  "tdi_alpha_h1": 241.6,
  "tdi_random_label": 7218.5,
  "signal_ratio": 1.099,
  "input_entropy_h0": 5.94,
  "input_entropy_h1": 5.23,
  "input_knn_purity": 0.569,
  "final_knn_purity": 0.963,
  "purity_gain": 0.394,
  "elapsed_s": 121.3
}
```

---

## Domains Covered

biology · physics · finance · nlp_features · ecology · software · vision · speech · robotics · medicine · engineering · chemistry · materials · social · psychology · education · neuroscience · environment · energy · aerospace · food · synthetic · digits · openml_catalog · mathematics

---

## Citation

```bibtex
@misc{vesterlund2026tdi,
  title  = {The TDI Atlas: A Cross-Domain Empirical Study of Topological Deformation in Neural Network Representations},
  author = {Vesterlund, David},
  year   = {2026},
  url    = {https://github.com/VesterlundCoder/tdi-atlas}
}
```

---

## Requirements

```
torch>=2.0.0
scikit-learn>=1.3.0
ripser>=0.6.0
persim>=0.3.0
gudhi>=3.8.0
umap-learn>=0.5.0
openml>=0.14.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```
