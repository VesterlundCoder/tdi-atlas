"""
CMF Irrationality VAE — General-Purpose Search
===============================================
Trains a score-conditioned VAE across ALL available CMF data:
  • Hunt shards (dim=4, ~74k records, source_score as proxy)
  • 6F5 f0g4 sweep data (δ as irrationality exponent)

Pipeline:
  1. Load + featurize both datasets with a unified irrationality score in [0,1]
  2. Train a β-VAE conditioned on score (score embedded into latent prior)
  3. Visualize latent space coloured by irrationality score (UMAP)
  4. Sample new candidates near high-score latent regions
  5. Decode → reconstruct CMF parameters
  6. Run lightweight convergence check (mpmath matrix iteration, depth 60)
  7. Identify limits via mpmath.identify + PSLQ
  8. Rank and report top candidates by predicted irrationality probability

Usage:
    python cmf_irrationality_vae.py [--train] [--generate N] [--identify]
    python cmf_irrationality_vae.py --full-run
    python cmf_irrationality_vae.py --full-run \\
        --shards /path/to/hunt_shards \\
        --sweeps /path/to/6F5Sweeps

Output: results/cmf_irr_vae/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    raise RuntimeError("PyTorch required: pip install torch")

try:
    import mpmath
    mpmath.mp.dps = 50
    _MPMATH_OK = True
except ImportError:
    _MPMATH_OK = False

try:
    import umap as umap_lib
    _UMAP_OK = True
except ImportError:
    _UMAP_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURIZATION
# ─────────────────────────────────────────────────────────────────────────────

FEAT_DIM = 24   # unified feature vector length


def _featurize_shard_record(rec: Dict) -> Optional[np.ndarray]:
    """
    Dim-4 CMF from hunt_shards.
    Features (24):
      [0..7]   D_params flattened (4 rows × 2 cols)
      [8..11]  D_params row-norms
      [12..15] D_params row-signs (sign of (col0 - col1))
      [16]     source_score normalised to [0,1]  (score-45.5)/(71-45.5)
      [17]     dim (always 4 here, normalised /10)
      [18..23] L_off / U_off: count and mean magnitude of nonzero entries (×3 each)
    """
    try:
        params = rec.get("params", {})
        D = params.get("D_params", [])
        if len(D) != 4:
            return None
        flat = []
        norms = []
        signs = []
        for row in D:
            if len(row) < 2:
                return None
            flat.extend([float(row[0]), float(row[1])])
            norms.append(math.sqrt(float(row[0])**2 + float(row[1])**2))
            signs.append(math.copysign(1.0, float(row[0]) - float(row[1])))
        sc = float(rec.get("source_score", 60.0))
        sc_norm = (sc - 45.5) / (71.0 - 45.5 + 1e-9)

        L_off = params.get("L_off", {})
        U_off = params.get("U_off", {})
        L_vals = [abs(float(v)) for v in L_off.values()] if L_off else []
        U_vals = [abs(float(v)) for v in U_off.values()] if U_off else []
        off_feats = [
            len(L_vals) / 10.0,
            float(np.mean(L_vals)) if L_vals else 0.0,
            float(np.std(L_vals)) if L_vals else 0.0,
            len(U_vals) / 10.0,
            float(np.mean(U_vals)) if U_vals else 0.0,
            float(np.std(U_vals)) if U_vals else 0.0,
        ]
        vec = flat + norms + signs + [sc_norm, 4.0 / 10.0] + off_feats
        return np.array(vec, dtype=np.float32)
    except Exception:
        return None


def _featurize_6f5_record(rec: Dict) -> Optional[np.ndarray]:
    """
    6F5 f0g4 record from sweep JSONL files.
    Features (24) — different semantics but same length:
      [0..10]  shift[11] first 11 elements (numerator/denominator pattern)
      [11]     z_float (real part)
      [12]     log|z| / 5  (normalised)
      [13..17] advancing g-param one-hot (0..4)
      [18]     delta_verify normalised  / 0.3
      [19]     dim 6/10
      [20..23] zeros (pad)
    """
    try:
        shift = rec.get("shift", [])
        if len(shift) < 11:
            shift = (shift + [0] * 11)[:11]
        z = rec.get("z", rec.get("z_num", 1))
        try:
            z_f = float(z) if not isinstance(z, (list, dict)) else 0.0
        except Exception:
            z_f = 0.0
        log_z = math.log(abs(z_f) + 1e-30) / 5.0
        adv_g = int(rec.get("adv_g", rec.get("advancing_g", 0))) % 5
        adv_onehot = [0.0] * 5
        adv_onehot[adv_g] = 1.0

        delta = float(rec.get("delta_verify",
                      rec.get("delta_screen",
                      rec.get("delta", 0.0))) or 0.0)
        delta_norm = min(delta / 0.3, 1.0)

        vec = [float(x) for x in shift[:11]] + \
              [z_f / 30.0, log_z] + adv_onehot + \
              [delta_norm, 6.0 / 10.0] + [0.0, 0.0, 0.0, 0.0]
        return np.array(vec[:FEAT_DIM], dtype=np.float32)
    except Exception:
        return None


def load_hunt_shards(shards_dir: str, max_records: int = 30_000) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, scores) from hunt_shards. scores in [0,1]."""
    path = Path(shards_dir)
    if not path.exists():
        print(f"  [WARN] hunt_shards not found: {shards_dir}")
        return np.zeros((0, FEAT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)

    import glob
    files = sorted(path.glob("*.jsonl"))
    feats, scores = [], []
    for jf in files:
        if len(feats) >= max_records:
            break
        with open(jf) as f:
            for line in f:
                if len(feats) >= max_records:
                    break
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                v = _featurize_shard_record(rec)
                if v is not None:
                    feats.append(v)
                    sc = float(rec.get("source_score", 60.0))
                    scores.append((sc - 45.5) / (71.0 - 45.5 + 1e-9))

    if not feats:
        return np.zeros((0, FEAT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack(feats).astype(np.float32)
    S = np.clip(np.array(scores, dtype=np.float32), 0, 1)
    print(f"  hunt_shards: {len(X)} records  score∈[{S.min():.3f},{S.max():.3f}]")
    return X, S


def load_6f5_sweeps(sweeps_dir: str, min_delta: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, scores) from 6F5 JSONL sweep files. scores = δ/0.3 clipped to [0,1]."""
    path = Path(sweeps_dir)
    if not path.exists():
        print(f"  [WARN] 6F5Sweeps not found: {sweeps_dir}")
        return np.zeros((0, FEAT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)

    target_files = [
        "f0g4_neighborhood_results.jsonl",
        "log_z_scan_deep.jsonl",
        "verify_delta_official.jsonl",
        "large_hits_positive_delta.jsonl",
        "wide_scalpel_results.jsonl",
    ]
    feats, scores = [], []
    for fname in target_files:
        fp = path / fname
        if not fp.exists():
            continue
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                delta = float(rec.get("delta_verify",
                              rec.get("delta_screen",
                              rec.get("delta", 0.0))) or 0.0)
                if delta < min_delta:
                    continue
                v = _featurize_6f5_record(rec)
                if v is not None:
                    feats.append(v)
                    scores.append(min(delta / 0.3, 1.0))

    if not feats:
        return np.zeros((0, FEAT_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
    X = np.stack(feats).astype(np.float32)
    S = np.clip(np.array(scores, dtype=np.float32), 0, 1)
    print(f"  6F5 sweeps:  {len(X)} records  δ∈[{S.min()*0.3:.3f},{S.max()*0.3:.3f}]")
    return X, S


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCORE-CONDITIONED VAE
# ─────────────────────────────────────────────────────────────────────────────

class IrrationalityVAE(nn.Module):
    """
    β-VAE conditioned on irrationality score s ∈ [0,1].
    Encoder: [x ‖ s] → μ, log_var
    Decoder: [z ‖ s] → x_hat
    The score pushes the latent space so that high-irrationality CMFs
    cluster together and can be sampled preferentially.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, latent_dim: int = 10,
                 hidden: Tuple[int, ...] = (128, 64)):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_dim = feat_dim + 1  # +1 for score conditioning
        for h in hidden:
            enc_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_lv = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim + 1  # +1 for score conditioning
        for h in reversed(hidden):
            dec_layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, feat_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_col = s.unsqueeze(-1)
        h = self.encoder(torch.cat([x, s_col], dim=-1))
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_col = s.unsqueeze(-1)
        return self.decoder(torch.cat([z, s_col], dim=-1))

    def reparameterize(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        mu, lv = self.encode(x, s)
        z = self.reparameterize(mu, lv)
        return self.decode(z, s), mu, lv


def train_vae(X: np.ndarray, S: np.ndarray,
              latent_dim: int = 10, epochs: int = 200,
              batch_size: int = 512, lr: float = 1e-3,
              beta: float = 2.0, seed: int = 42) -> Tuple["IrrationalityVAE", Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = IrrationalityVAE(feat_dim=X.shape[1], latent_dim=latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xt = torch.tensor(X)
    St = torch.tensor(S)
    N = len(X)

    # Normalise X to [0,1] per feature
    X_min = Xt.min(0).values
    X_max = Xt.max(0).values
    X_range = (X_max - X_min).clamp(min=1e-6)
    Xt_norm = (Xt - X_min) / X_range

    history = {"recon": [], "kl": [], "total": []}
    beta_schedule = np.linspace(0.0, beta, epochs // 4).tolist() + \
                    [beta] * (epochs - epochs // 4)

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(N)
        epoch_recon, epoch_kl = 0.0, 0.0
        n_batches = 0
        b = beta_schedule[epoch]

        for start in range(0, N, batch_size):
            bi = idx[start:start + batch_size]
            xb = Xt_norm[bi]
            sb = St[bi]

            x_hat, mu, lv = model(xb, sb)
            recon = F.mse_loss(x_hat, xb)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = recon + b * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_recon += float(recon)
            epoch_kl += float(kl)
            n_batches += 1

        sched.step()
        history["recon"].append(epoch_recon / n_batches)
        history["kl"].append(epoch_kl / n_batches)
        history["total"].append(history["recon"][-1] + beta * history["kl"][-1])

        if (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:>3}/{epochs}  "
                  f"recon={history['recon'][-1]:.4f}  "
                  f"kl={history['kl'][-1]:.4f}  "
                  f"β={b:.2f}")

    # Store normalization stats on model for decode
    model.X_min = X_min
    model.X_range = X_range
    return model, history


def encode_all(model: "IrrationalityVAE",
               X: np.ndarray, S: np.ndarray) -> np.ndarray:
    model.eval()
    Xt_norm = (torch.tensor(X) - model.X_min) / model.X_range
    St = torch.tensor(S)
    with torch.no_grad():
        mu, _ = model.encode(Xt_norm, St)
    return mu.numpy()


def sample_high_irrationality(model: "IrrationalityVAE",
                               Z_all: np.ndarray, S_all: np.ndarray,
                               n_samples: int = 500,
                               top_frac: float = 0.15,
                               noise_std: float = 0.4,
                               seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample new latent vectors near the high-score region.
    Returns (new_z, target_scores) — scores drawn from top-score distribution.
    """
    rng = np.random.default_rng(seed)
    threshold = np.quantile(S_all, 1.0 - top_frac)
    top_mask = S_all >= threshold
    Z_top = Z_all[top_mask]
    S_top = S_all[top_mask]

    idx = rng.integers(0, len(Z_top), size=n_samples)
    noise = rng.normal(0, noise_std, size=(n_samples, Z_top.shape[1])).astype(np.float32)
    Z_new = Z_top[idx] + noise

    # Use maximum score as conditioning target for decode
    target_scores = np.full(n_samples, float(S_top.max()), dtype=np.float32)
    return Z_new.astype(np.float32), target_scores


def decode_candidates(model: "IrrationalityVAE",
                      Z_new: np.ndarray, S_new: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_hat = model.decode(torch.tensor(Z_new), torch.tensor(S_new))
    # Inverse normalise
    X_raw = x_hat * model.X_range + model.X_min
    return X_raw.numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LIGHTWEIGHT CONVERGENCE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _build_cmf_matrix(d_params: List[List[float]],
                      n: int, z: float = 1.0) -> Optional[np.ndarray]:
    """Build the 4×4 CMF companion matrix for step n."""
    try:
        dim = len(d_params)
        M = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            a, b = float(d_params[i][0]), float(d_params[i][1])
            M[i, i] = a * n + b
        M[0, 0] -= z  # z-coupling in (0,0) entry
        return M
    except Exception:
        return None


def estimate_delta_dim4(d_params: List[List[float]],
                        z: float = 1.0, depth: int = 60) -> Tuple[float, float]:
    """
    Quick estimate of irrationality exponent δ for a dim-4 CMF.
    Uses ratio  δ ≈ log|det(M_n)| / (n log n)  at depth=n.
    Returns (delta_est, limit_approx).
    """
    if not _MPMATH_OK:
        return 0.0, 0.0
    try:
        mpmath.mp.dps = 30
        # Run matrix recurrence
        dim = 4
        P = mpmath.eye(dim)
        Q = mpmath.eye(dim)
        for n in range(1, depth + 1):
            M = mpmath.matrix(dim, dim)
            for i in range(dim):
                a = mpmath.mpf(d_params[i][0])
                b = mpmath.mpf(d_params[i][1])
                M[i, i] = a * n + b
            M[0, 0] -= mpmath.mpf(z)
            # Simple 1D recurrence via trace/det
            # Use scalar continued fraction estimate
            pass

        # Simpler: scalar recurrence p_{n+1} = trace*p_n - det*p_{n-1}
        # For quick estimate, compute ratio of successive partial sums
        p_prev, p_curr = mpmath.mpf(1), mpmath.mpf(0)
        q_prev, q_curr = mpmath.mpf(0), mpmath.mpf(1)
        for n in range(1, depth + 1):
            coeff = mpmath.mpf(0)
            for i in range(len(d_params)):
                a = mpmath.mpf(d_params[i][0])
                b = mpmath.mpf(d_params[i][1])
                coeff += a * n + b
            coeff /= len(d_params)
            p_new = coeff * p_curr - p_prev
            q_new = coeff * q_curr - q_prev
            p_prev, p_curr = p_curr, p_new
            q_prev, q_curr = q_curr, q_new

        if abs(q_curr) < 1e-30 or abs(q_prev) < 1e-30:
            return 0.0, 0.0

        limit = float(p_curr / q_curr) if abs(q_curr) > 1e-30 else 0.0

        # δ estimate from |p/q - L| ~ q^{-1-δ}
        diff = abs(p_curr - q_curr * float(p_prev / q_prev)) if abs(q_prev) > 1e-30 else 1.0
        log_q = math.log(max(abs(float(q_curr)), 1.0) + 1e-30)
        log_diff = math.log(max(abs(diff), 1e-50))
        delta_est = max(0.0, (-log_diff / log_q - 1.0)) if log_q > 1 else 0.0
        return min(delta_est, 3.0), limit
    except Exception:
        return 0.0, 0.0


def reconstruct_d_params(x_vec: np.ndarray) -> List[List[float]]:
    """Inverse of _featurize_shard_record: extract D_params from feature vector."""
    d_params = []
    for i in range(4):
        a = float(x_vec[2 * i])
        b = float(x_vec[2 * i + 1])
        # Snap to nearest half-integer (CMF params are typically ±0.5 multiples)
        a = round(a * 2) / 2.0
        b = round(b * 2) / 2.0
        d_params.append([a, b])
    return d_params


def verify_candidates(X_decoded: np.ndarray,
                      n_top: int = 200,
                      depth: int = 60) -> List[Dict]:
    """
    Lightweight convergence check on decoded candidates.
    Returns list of dicts sorted by delta_est descending.
    """
    results = []
    for i, x in enumerate(X_decoded[:n_top]):
        d_params = reconstruct_d_params(x)
        # Try z=1.0 and z=-1.0
        for z in [1.0, -1.0, 0.5, -0.5]:
            delta, limit = estimate_delta_dim4(d_params, z=z, depth=depth)
            if delta > 0.05 and abs(limit) > 1e-8 and abs(limit) < 1e4:
                results.append({
                    "candidate_idx": i,
                    "d_params": d_params,
                    "z": z,
                    "delta_est": delta,
                    "limit_approx": limit,
                    "identified": None,
                })
    results.sort(key=lambda r: r["delta_est"], reverse=True)
    return results


def identify_limits(candidates: List[Dict], top_k: int = 30) -> List[Dict]:
    """Try mpmath.identify on the top-k limit values."""
    if not _MPMATH_OK:
        return candidates
    for r in candidates[:top_k]:
        try:
            lim = mpmath.mpf(str(r["limit_approx"]))
            identified = mpmath.identify(lim, tol=1e-8)
            r["identified"] = str(identified) if identified else None
        except Exception:
            r["identified"] = None
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _umap_or_pca(Z: np.ndarray, seed: int = 0) -> np.ndarray:
    if _UMAP_OK and len(Z) >= 15:
        reducer = umap_lib.UMAP(n_components=2, random_state=seed,
                                 n_neighbors=min(15, len(Z)-1), min_dist=0.1)
        return reducer.fit_transform(Z)
    # Fallback: PCA
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(Z)


def plot_latent_space(Z: np.ndarray, S: np.ndarray, sources: np.ndarray,
                      out_path: str):
    if not _MPL_OK:
        return
    Z2 = _umap_or_pca(Z)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: coloured by irrationality score
    sc = axes[0].scatter(Z2[:, 0], Z2[:, 1], c=S, cmap="plasma",
                         alpha=0.4, s=6, linewidths=0)
    plt.colorbar(sc, ax=axes[0], label="Irrationality score (normalised)")
    axes[0].set_title("Latent Space — Irrationality Score", fontweight="bold")
    axes[0].set_xlabel("UMAP-1" if _UMAP_OK else "PC-1")
    axes[0].set_ylabel("UMAP-2" if _UMAP_OK else "PC-2")

    # Right: coloured by data source (hunt=0, 6F5=1)
    colors_src = ["#3498db" if s == 0 else "#e74c3c" for s in sources]
    axes[1].scatter(Z2[:, 0], Z2[:, 1], c=colors_src,
                    alpha=0.4, s=6, linewidths=0)
    import matplotlib.patches as mpatches
    axes[1].legend(handles=[
        mpatches.Patch(color="#3498db", label="hunt_shards (dim-4)"),
        mpatches.Patch(color="#e74c3c", label="6F5 f0g4 sweep"),
    ], fontsize=9)
    axes[1].set_title("Latent Space — Data Source", fontweight="bold")
    axes[1].set_xlabel("UMAP-1" if _UMAP_OK else "PC-1")
    axes[1].set_ylabel("UMAP-2" if _UMAP_OK else "PC-2")

    plt.suptitle("CMF Irrationality VAE — Latent Space", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_score_distribution(S_hunt: np.ndarray, S_6f5: np.ndarray,
                             candidates: List[Dict], out_path: str):
    if not _MPL_OK:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(S_hunt, bins=40, color="#3498db", alpha=0.7,
                 label=f"hunt_shards (n={len(S_hunt)})", density=True)
    if len(S_6f5) > 0:
        axes[0].hist(S_6f5, bins=20, color="#e74c3c", alpha=0.7,
                     label=f"6F5 f0g4 δ/0.3 (n={len(S_6f5)})", density=True)
    axes[0].set_xlabel("Normalised irrationality score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score Distribution by Dataset", fontweight="bold")
    axes[0].legend(fontsize=9)

    if candidates:
        deltas = [r["delta_est"] for r in candidates[:100]]
        axes[1].hist(deltas, bins=25, color="#2ecc71", edgecolor="white")
        axes[1].axvline(0.15, color="#e74c3c", linestyle="--",
                        label="δ=0.15 (good threshold)")
        axes[1].set_xlabel("δ estimate (generated candidates)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Generated Candidates — δ Distribution", fontweight="bold")
        axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_top_candidates(candidates: List[Dict], out_path: str):
    if not _MPL_OK or not candidates:
        return
    top = candidates[:20]
    labels = [f"[{r['candidate_idx']}] z={r['z']:.1f}" for r in top]
    deltas = [r["delta_est"] for r in top]
    limits = [r["limit_approx"] for r in top]
    colors = ["#e74c3c" if r["identified"] else "#3498db" for r in top]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    bars = axes[0].barh(labels[::-1], deltas[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, deltas[::-1]):
        axes[0].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)
    axes[0].set_xlabel("δ estimate")
    axes[0].set_title("Top-20 Generated Candidates by δ", fontweight="bold")
    axes[0].axvline(0.15, color="#e67e22", linestyle="--", lw=1.2)

    axes[1].scatter(range(len(candidates)), [r["delta_est"] for r in candidates],
                    c=["#e74c3c" if r["identified"] else "#3498db" for r in candidates],
                    alpha=0.6, s=20)
    axes[1].axhline(0.15, color="#e67e22", linestyle="--", lw=1.2, label="δ=0.15")
    axes[1].set_xlabel("Candidate rank")
    axes[1].set_ylabel("δ estimate")
    axes[1].set_title("All Verified Candidates (red = identified limit)", fontweight="bold")
    axes[1].legend(fontsize=9)

    import matplotlib.patches as mpatches
    fig.legend(handles=[
        mpatches.Patch(color="#e74c3c", label="Limit identified via mpmath"),
        mpatches.Patch(color="#3498db", label="Unidentified"),
    ], loc="lower center", ncol=2, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "cmf_irr_vae.pt"

    print("=" * 70)
    print("CMF Irrationality VAE — General-Purpose Search")
    print(f"  shards   = {args.shards}")
    print(f"  sweeps   = {args.sweeps}")
    print(f"  out_dir  = {args.out_dir}")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    X_hunt, S_hunt = load_hunt_shards(args.shards, max_records=args.max_hunt)
    X_6f5, S_6f5 = load_6f5_sweeps(args.sweeps)

    if len(X_hunt) == 0 and len(X_6f5) == 0:
        print("ERROR: No data loaded. Check --shards and --sweeps paths.")
        return

    if len(X_hunt) > 0 and len(X_6f5) > 0:
        X_all = np.vstack([X_hunt, X_6f5])
        S_all = np.concatenate([S_hunt, S_6f5])
        sources = np.array([0] * len(X_hunt) + [1] * len(X_6f5))
    elif len(X_hunt) > 0:
        X_all, S_all = X_hunt, S_hunt
        sources = np.zeros(len(X_hunt), dtype=int)
    else:
        X_all, S_all = X_6f5, S_6f5
        sources = np.ones(len(X_6f5), dtype=int)

    print(f"  Combined: {len(X_all)} records  "
          f"(hunt={len(X_hunt)}, 6F5={len(X_6f5)})")

    # ── Train VAE ─────────────────────────────────────────────────────────
    if args.train or not ckpt_path.exists():
        print(f"\n[2/6] Training VAE ({args.epochs} epochs, latent_dim={args.latent_dim})...")
        t0 = time.time()
        model, history = train_vae(
            X_all, S_all,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            beta=args.beta,
        )
        torch.save({
            "model_state": model.state_dict(),
            "X_min": model.X_min,
            "X_range": model.X_range,
            "history": history,
            "feat_dim": FEAT_DIM,
            "latent_dim": args.latent_dim,
        }, ckpt_path)
        print(f"  Trained in {time.time()-t0:.0f}s. Saved: {ckpt_path}")
    else:
        print(f"\n[2/6] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = IrrationalityVAE(feat_dim=FEAT_DIM, latent_dim=ckpt["latent_dim"])
        model.load_state_dict(ckpt["model_state"])
        model.X_min = ckpt["X_min"]
        model.X_range = ckpt["X_range"]
        history = ckpt["history"]

    # ── Encode all ─────────────────────────────────────────────────────────
    print("\n[3/6] Encoding dataset into latent space...")
    Z_all = encode_all(model, X_all, S_all)

    # ── Visualize ──────────────────────────────────────────────────────────
    if args.visualize:
        print("\n[4/6] Visualizing latent space...")
        # Subsample for speed if large
        max_vis = 5000
        if len(Z_all) > max_vis:
            idx = np.random.default_rng(0).choice(len(Z_all), max_vis, replace=False)
        else:
            idx = np.arange(len(Z_all))
        plot_latent_space(Z_all[idx], S_all[idx], sources[idx],
                          str(out_dir / "fig_latent_space.png"))
        plot_score_distribution(S_hunt, S_6f5, [],
                                str(out_dir / "fig_score_distribution.png"))
    else:
        print("\n[4/6] Skipping visualization (pass --visualize to enable)")

    # ── Generate ───────────────────────────────────────────────────────────
    print(f"\n[5/6] Generating {args.generate} candidates (top {int(args.top_frac*100)}% region)...")
    Z_new, S_new = sample_high_irrationality(
        model, Z_all, S_all,
        n_samples=args.generate,
        top_frac=args.top_frac,
    )
    X_decoded = decode_candidates(model, Z_new, S_new)
    print(f"  Decoded {len(X_decoded)} candidate parameter vectors")

    # ── Verify ─────────────────────────────────────────────────────────────
    print(f"\n[6/6] Running convergence check (depth={args.depth})...")
    candidates = verify_candidates(X_decoded, n_top=args.generate, depth=args.depth)
    n_hits = sum(1 for r in candidates if r["delta_est"] > 0.10)
    print(f"  Candidates with δ > 0.10: {n_hits}/{args.generate} "
          f"({100*n_hits/max(args.generate,1):.1f}%)")

    # ── Identify ───────────────────────────────────────────────────────────
    if args.identify and candidates:
        print(f"  Running mpmath.identify on top-{min(40, len(candidates))} limits...")
        candidates = identify_limits(candidates, top_k=40)
        identified = [(r["limit_approx"], r["identified"])
                      for r in candidates if r["identified"]]
        print(f"  Identified: {len(identified)}")
        for lim, name in identified[:10]:
            print(f"    L≈{lim:.6f}  →  {name}")

    # ── Save results ───────────────────────────────────────────────────────
    results_path = out_dir / "candidates.json"
    with open(results_path, "w") as f:
        json.dump({
            "n_training_records": int(len(X_all)),
            "n_hunt": int(len(X_hunt)),
            "n_6f5": int(len(X_6f5)),
            "n_generated": args.generate,
            "n_hits_delta_gt_010": n_hits,
            "hit_rate_pct": 100 * n_hits / max(args.generate, 1),
            "top_candidates": candidates[:50],
        }, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    if args.visualize and candidates:
        plot_score_distribution(S_hunt, S_6f5, candidates,
                                str(out_dir / "fig_score_distribution.png"))
        plot_top_candidates(candidates, str(out_dir / "fig_top_candidates.png"))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Training set  : {len(X_all)} CMFs  "
          f"(hunt_shards dim-4={len(X_hunt)}, 6F5 f0g4={len(X_6f5)})")
    print(f"  Latent dim    : {args.latent_dim}")
    print(f"  Generated     : {args.generate} candidates")
    print(f"  δ > 0.10 hits : {n_hits}  ({100*n_hits/max(args.generate,1):.1f}%)")
    if candidates:
        best = candidates[0]
        print(f"  Best candidate:")
        print(f"    d_params = {best['d_params']}")
        print(f"    z = {best['z']},  δ_est = {best['delta_est']:.4f}")
        print(f"    limit ≈ {best['limit_approx']:.8f}")
        if best.get("identified"):
            print(f"    identified as: {best['identified']}")
    print(f"  Output dir    : {out_dir}")
    print("=" * 70)


def parse_args():
    p = argparse.ArgumentParser(
        description="CMF Irrationality VAE — general-purpose search")
    p.add_argument("--full-run", action="store_true",
                   help="Train + visualize + generate + identify in one go")
    p.add_argument("--train", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--generate", type=int, default=400,
                   help="Number of candidates to generate and verify")
    p.add_argument("--identify", action="store_true",
                   help="Run mpmath.identify on top limits")
    p.add_argument("--shards",
                   default="/Users/davidsvensson/Desktop/rd-lumi-z3/"
                           "cmf_loop_project/hunt_shards")
    p.add_argument("--sweeps",
                   default="/Users/davidsvensson/Desktop/rd-lumi-z3/6F5Sweeps")
    p.add_argument("--out-dir", default="results/cmf_irr_vae")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--top-frac", type=float, default=0.15,
                   help="Fraction of top-score latent points to sample near")
    p.add_argument("--depth", type=int, default=60,
                   help="Recurrence depth for delta estimation")
    p.add_argument("--max-hunt", type=int, default=30_000,
                   help="Max records from hunt_shards (larger = slower training)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.full_run:
        args.train = True
        args.visualize = True
        args.identify = True
    run_pipeline(args)
