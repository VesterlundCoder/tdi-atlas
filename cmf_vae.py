"""
CMF Variational Autoencoder — Latent Space Navigator for Mathematical Constant Discovery.

Trains a VAE on CMF hunt_shards (74k+ records) to learn a smooth latent manifold
of CMF coefficient space. Uses this manifold to:
  1. Visualise the structure of known CMF families
  2. Sample new candidate CMF coefficients from unexplored regions
  3. Interpolate between known formulas to discover intermediate ones
  4. Target-conditioned generation: propose CMFs near a given limit value

Usage:
    python cmf_vae.py --train                         # train VAE
    python cmf_vae.py --train --epochs 200
    python cmf_vae.py --visualize                     # UMAP of latent space
    python cmf_vae.py --generate 1000                 # sample 1000 candidates
    python cmf_vae.py --interpolate                   # walk between two CMFs
    python cmf_vae.py --target 1.2020569              # target ζ(3) ~ Apéry const
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")


# ── Architecture ──────────────────────────────────────────────────────────────

class CMF_VAE(nn.Module):
    """Variational Autoencoder for CMF coefficient space.

    Encoder: X (d_in) → [μ (latent_dim), log_σ² (latent_dim)]
    Decoder: z (latent_dim) → X̂ (d_in)

    Loss = MSE(X, X̂) + β * KL(N(μ,σ²) || N(0,1))
    β anneals from 0 → 1 over warmup_epochs for stable training.
    """

    def __init__(self, in_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        enc = []
        prev = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            prev = h
        self.encoder_body = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.LeakyReLU(0.1)]
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cmf_data(shards_dir: str,
                  max_per_class: int = 3000) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder, List[str]]:
    """Load CMF hunt shards. Returns (X_scaled, y_encoded, scaler, le, feature_names)."""
    path = Path(shards_dir)
    features, labels = [], []
    feature_names = None

    for jf in sorted(path.glob("*.jsonl")):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                params = rec.get("params", {})
                d_raw = params.get("D_params", [])
                d_flat = []
                for entry in d_raw:
                    if isinstance(entry, list):
                        d_flat.extend(float(v) for v in entry)
                    else:
                        d_flat.append(float(entry))
                if len(d_flat) < 2:
                    continue
                score = float(rec.get("source_score", 0.0))
                def _s(d):
                    vals = [float(v) for v in d.values() if v is not None] if d else []
                    if not vals:
                        return [0.] * 5
                    return [np.mean(vals), np.std(vals), np.min(vals), np.max(vals), float(len(vals))]
                l_s = _s(params.get("L_off", {}))
                u_s = _s(params.get("U_off", {}))
                try:
                    lim = abs(float(str(rec.get("source_limit", "0"))[:30]))
                except Exception:
                    lim = 0.0
                fv = d_flat + [score] + l_s + u_s + [lim]
                if feature_names is None:
                    feature_names = (
                        [f"D_{i}" for i in range(len(d_flat))]
                        + ["score"]
                        + [f"L_{s}" for s in ("mean","std","min","max","n")]
                        + [f"U_{s}" for s in ("mean","std","min","max","n")]
                        + ["limit_abs"]
                    )
                features.append(np.array(fv, dtype=np.float32))
                labels.append(str(rec.get("source_tier", "?")))

    if not features:
        raise FileNotFoundError(f"No CMF data found in {shards_dir}")

    max_len = max(len(f) for f in features)
    X_all = np.stack([np.pad(f, (0, max_len - len(f))) for f in features])
    le = LabelEncoder()
    y_all = le.fit_transform(labels).astype(np.int64)

    # Stratified cap
    rng = np.random.default_rng(42)
    keep = []
    for c in np.unique(y_all):
        idx = np.where(y_all == c)[0]
        n = min(len(idx), max_per_class)
        keep.extend(rng.choice(idx, n, replace=False).tolist())
    keep = np.array(keep); rng.shuffle(keep)
    X_all = X_all[keep]; y_all = y_all[keep]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all).astype(np.float32)
    return X_scaled, y_all, scaler, le, (feature_names or [])


# ── Training ──────────────────────────────────────────────────────────────────

def train_vae(X: np.ndarray,
              latent_dim: int = 16,
              hidden_dims: List[int] = None,
              epochs: int = 150,
              batch_size: int = 512,
              lr: float = 1e-3,
              beta_max: float = 0.5,
              warmup_epochs: int = 30,
              seed: int = 42,
              verbose: bool = True) -> CMF_VAE:
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CMF_VAE(X.shape[1], hidden_dims, latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(X)
    N = len(X)

    for epoch in range(epochs):
        model.train()
        beta = beta_max * min(1.0, epoch / max(warmup_epochs, 1))
        perm = rng.permutation(N)
        epoch_losses = []
        for start in range(0, N, batch_size):
            bidx = perm[start:start + batch_size]
            xb = Xt[bidx]
            x_hat, mu, logvar = model(xb)
            loss, recon, kl = model.loss(xb, x_hat, mu, logvar, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_losses.append(float(loss))
        scheduler.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  loss={np.mean(epoch_losses):.4f}  β={beta:.3f}")

    return model


# ── Latent space utilities ────────────────────────────────────────────────────

def encode_all(model: CMF_VAE, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Encode full dataset to mean latent vectors."""
    model.eval()
    Xt = torch.tensor(X)
    mus = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            mu, _ = model.encode(Xt[start:start + batch_size])
            mus.append(mu.numpy())
    return np.concatenate(mus, axis=0)


def sample_from_tier(model: CMF_VAE,
                     Z: np.ndarray, y: np.ndarray,
                     tier_label: int, n_samples: int,
                     noise_scale: float = 0.5,
                     seed: int = 0) -> np.ndarray:
    """Sample new CMF coefficient vectors near a specific tier cluster."""
    rng = np.random.default_rng(seed)
    tier_idx = np.where(y == tier_label)[0]
    tier_z = Z[tier_idx]
    mu = tier_z.mean(axis=0)
    sigma = tier_z.std(axis=0) * noise_scale

    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            z_new = torch.tensor((mu + rng.normal(0, 1, mu.shape) * sigma).astype(np.float32)).unsqueeze(0)
            x_hat = model.decode(z_new).squeeze(0).numpy()
            samples.append(x_hat)
    return np.stack(samples)


def interpolate_formulas(model: CMF_VAE, X: np.ndarray,
                         idx_a: int, idx_b: int, n_steps: int = 10) -> np.ndarray:
    """Interpolate in latent space between two CMF records."""
    model.eval()
    Xa = torch.tensor(X[idx_a:idx_a+1])
    Xb = torch.tensor(X[idx_b:idx_b+1])
    with torch.no_grad():
        za, _ = model.encode(Xa)
        zb, _ = model.encode(Xb)
        alphas = np.linspace(0, 1, n_steps)
        results = []
        for a in alphas:
            z_mid = (1 - a) * za + a * zb
            x_hat = model.decode(z_mid).squeeze(0).numpy()
            results.append(x_hat)
    return np.stack(results)


def target_conditioned_search(model: CMF_VAE,
                               Z: np.ndarray,
                               X_orig: np.ndarray,
                               limits: List[float],
                               target_value: float,
                               scaler: StandardScaler,
                               feature_names: List[str],
                               n_candidates: int = 200,
                               tol: float = 0.05,
                               seed: int = 0) -> List[Dict]:
    """Find CMF candidates whose limit value is near target_value.

    Strategy: weight the latent sampling by proximity of the original
    limit_abs feature to the target value, then decode and report top hits.
    """
    rng = np.random.default_rng(seed)
    limits_arr = np.array(limits)
    dists = np.abs(limits_arr - target_value)
    weights = np.exp(-dists / (tol + 1e-10))
    weights /= weights.sum()

    chosen = rng.choice(len(Z), size=min(n_candidates, len(Z)),
                         replace=False, p=weights)
    z_seed = Z[chosen]
    noise = rng.normal(0, 0.3, z_seed.shape).astype(np.float32)
    z_perturb = torch.tensor(z_seed + noise)

    model.eval()
    candidates = []
    with torch.no_grad():
        x_hats = model.decode(z_perturb).numpy()
    x_hats_unscaled = scaler.inverse_transform(x_hats)

    for i, row in enumerate(x_hats_unscaled):
        rec = {fn: float(v) for fn, v in zip(feature_names, row)}
        rec["source_z"] = chosen[i]
        rec["search_target"] = target_value
        candidates.append(rec)
    return candidates


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_latent(Z: np.ndarray, y: np.ndarray, le: LabelEncoder,
                     out_dir: Path, method: str = "umap"):
    """Project latent space to 2D and plot tier clusters."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if method == "umap":
        try:
            from umap import UMAP
            Z2 = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(Z)
            label = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
            label = "PCA (umap not installed)"
    else:
        from sklearn.decomposition import PCA
        Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
        label = "PCA"

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"A": "#e74c3c", "B": "#2ecc71", "C": "#3498db"}
    tier_names = le.classes_

    for c, tier in enumerate(tier_names):
        mask = y == c
        col = colors.get(tier, "#888888")
        ax.scatter(Z2[mask, 0], Z2[mask, 1], c=col, s=8, alpha=0.5, rasterized=True)

    patches = [mpatches.Patch(color=colors.get(t, "#888"), label=f"Tier {t}")
               for t in tier_names]
    ax.legend(handles=patches, fontsize=11)
    ax.set_title(f"CMF Latent Space — {label}\n"
                 f"Tier A = best convergence, C = weaker convergence", fontsize=12)
    ax.set_xlabel(f"{label}-1"); ax.set_ylabel(f"{label}-2")
    ax.grid(True, alpha=0.2)
    out = out_dir / f"cmf_latent_{label.lower().split()[0]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

KNOWN_CONSTANTS = {
    "apery_zeta3":      1.2020569031595942,
    "zeta_5":           1.0369277551433699,
    "pi":               3.141592653589793,
    "1_over_pi":        0.3183098861837907,
    "ln2":              0.6931471805599453,
    "catalan":          0.9159655941772190,
    "euler_gamma":      0.5772156649015329,
    "sqrt2":            1.4142135623730951,
}


def parse_args():
    p = argparse.ArgumentParser(description="CMF VAE — Latent Space Navigator")
    p.add_argument("--train",       action="store_true", help="Train the VAE")
    p.add_argument("--visualize",   action="store_true", help="Generate latent space plot")
    p.add_argument("--generate",    type=int, default=0,
                   help="Sample N new CMF candidates from tier-A cluster")
    p.add_argument("--interpolate", action="store_true",
                   help="Interpolate between two tier-A formulas")
    p.add_argument("--target",      type=float, default=None,
                   help="Target limit value for conditioned search (e.g. 1.2020569 for ζ(3))")
    p.add_argument("--target-name", type=str, default=None,
                   help=f"Named constant: {list(KNOWN_CONSTANTS.keys())}")
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--latent-dim",  type=int, default=16)
    p.add_argument("--checkpoint",  type=str, default="results/cmf_vae.pt")
    p.add_argument("--out-dir",     type=str, default="results")
    p.add_argument("--cmf-shards",  type=str,
                   default="/Users/davidsvensson/Desktop/rd-lumi-z3/cmf_loop_project/hunt_shards")
    p.add_argument("--max-per-class", type=int, default=3000)
    p.add_argument("--quiet",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    print("Loading CMF data...")
    X, y, scaler, le, feature_names = load_cmf_data(
        args.cmf_shards, max_per_class=args.max_per_class
    )
    print(f"  N={len(X)}  D={X.shape[1]}  classes={le.classes_}")

    # Train or load
    if args.train or not ckpt_path.exists():
        print(f"\nTraining VAE  (epochs={args.epochs}  latent_dim={args.latent_dim})")
        model = train_vae(X, latent_dim=args.latent_dim,
                          epochs=args.epochs, verbose=verbose)
        torch.save({"model_state": model.state_dict(),
                    "in_dim": X.shape[1],
                    "latent_dim": args.latent_dim,
                    "feature_names": feature_names,
                    "tier_classes": list(le.classes_)},
                   ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")
    else:
        print(f"\nLoading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = CMF_VAE(ckpt["in_dim"], [128, 64, 32], ckpt["latent_dim"])
        model.load_state_dict(ckpt["model_state"])

    # Encode full dataset
    print("\nEncoding all CMFs to latent space...")
    Z = encode_all(model, X)
    print(f"  Latent shape: {Z.shape}")

    # Visualise
    if args.visualize:
        print("\nGenerating latent space visualisation...")
        visualize_latent(Z, y, le, out_dir)

    # Generate candidates
    if args.generate > 0:
        tier_a_label = list(le.classes_).index("A") if "A" in le.classes_ else 0
        print(f"\nSampling {args.generate} new candidates near Tier-A cluster...")
        candidates = sample_from_tier(model, Z, y, tier_a_label, args.generate)
        candidates_unscaled = scaler.inverse_transform(candidates)
        out_json = out_dir / "cmf_candidates.json"
        recs = [{fn: float(v) for fn, v in zip(feature_names, row)}
                for row in candidates_unscaled]
        with open(out_json, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  {args.generate} candidates → {out_json}")

    # Interpolate
    if args.interpolate:
        tier_a_idx = np.where(y == (list(le.classes_).index("A") if "A" in le.classes_ else 0))[0]
        idx_a, idx_b = tier_a_idx[0], tier_a_idx[min(50, len(tier_a_idx)-1)]
        print(f"\nInterpolating between CMF[{idx_a}] and CMF[{idx_b}] in latent space...")
        path_raw = interpolate_formulas(model, X, idx_a, idx_b, n_steps=12)
        path_unscaled = scaler.inverse_transform(path_raw)
        out_path_json = out_dir / "cmf_interpolation.json"
        recs = [{fn: float(v) for fn, v in zip(feature_names, row)}
                for row in path_unscaled]
        with open(out_path_json, "w") as f:
            json.dump(recs, f, indent=2)
        print(f"  12-step path → {out_path_json}")

    # Target-conditioned search
    target = args.target
    if args.target_name and args.target_name in KNOWN_CONSTANTS:
        target = KNOWN_CONSTANTS[args.target_name]
        print(f"\n  Target: {args.target_name} = {target:.10f}")
    if target is not None:
        # Extract raw limit values
        limit_idx = feature_names.index("limit_abs") if "limit_abs" in feature_names else -1
        X_raw = scaler.inverse_transform(X)
        limits = X_raw[:, limit_idx].tolist() if limit_idx >= 0 else [0.0] * len(X)
        print(f"\nTarget-conditioned search: limit ≈ {target:.8f}")
        hits = target_conditioned_search(
            model, Z, X, limits, target, scaler, feature_names,
            n_candidates=500
        )
        out_hits = out_dir / f"cmf_target_{target:.6f}.json"
        with open(out_hits, "w") as f:
            json.dump(hits[:100], f, indent=2)
        print(f"  Top 100 candidates → {out_hits}")

    print("\nDone.")


if __name__ == "__main__":
    main()
