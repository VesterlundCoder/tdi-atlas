## Table 2. Model Architecture and Training Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | FC(D,64)→BN→ReLU → FC(64,32)→BN→ReLU → FC(32,C) |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-4) |
| LR schedule | Cosine decay over 80 epochs |
| Batch size | 256 (full-batch for N≤256) |
| Epochs | 80 |
| PH subsample | 400 points (stratified random, seed=42) |
| Filtrations | Vietoris-Rips (ripser) + Alpha complex (gudhi) |
| Homology dims | H₀, H₁ |
| Distance metric | Wasserstein-2 (persim) |
| Random-label runs | 1 per dataset (same seed, labels permuted) |
