"""
CMF 6F5 Explorer — VAE-guided search for high-δ f0g4 stratum formulas.

The f0g4 stratum is the most productive class of 6-dimensional Confluent
Hypergeometric CMFs known to yield positive irrationality exponent δ > 0.
This module:

  1. Loads all 6F5 scan data from 6F5Sweeps/ JSONL files
  2. Featurises each record (shift, z, advancing g-direction)
  3. Trains a Variational Autoencoder on the f0g4-stratum data
  4. Projects the latent space to 2D (UMAP or PCA) coloured by δ value
  5. Samples new candidates from the high-δ neighbourhood in latent space
  6. Runs the actual 6F5 matrix recurrence (with mpmath) to verify δ
  7. Pipes verified high-δ candidates to a PSLQ/mpmath identifier
  8. Saves all results + generates a delta-profile report

Background (from 6F5Sweeps/ research):
  - f0g4 stratum: 4 of 5 g-params fixed at 0, one active g drives recurrence
  - Confirmed: 10 trajectories with δ > 0.10, best δ = 0.208 (Rank 1: z=-1/20)
  - Delta ladder:  f0g4 (δ≈0.15) < f1g4 (δ≈0.22) < f2g4 (δ≈0.32)
                   < f3g4 (δ≈0.50, proven irrational) < f4g4 (δ≈1.0)
  - Log-z scan: 7020 hits across |z| zones; Zone V (|z|<10⁻⁵) gives δ→0.205

Usage:
    python cmf_6f5_explorer.py --train                    # train VAE on 6F5 data
    python cmf_6f5_explorer.py --visualize                # UMAP latent space (δ coloured)
    python cmf_6f5_explorer.py --generate 200 --verify    # generate + δ-verify candidates
    python cmf_6f5_explorer.py --identify                 # PSLQ on best verified L values
    python cmf_6f5_explorer.py --full-run                 # all of the above
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

DEFAULT_SWEEPS_DIR = Path(
    "/Users/davidsvensson/Desktop/rd-lumi-z3/6F5Sweeps"
)

DATA_FILES = [
    "f0g4_neighborhood_results.jsonl",
    "log_z_scan_deep.jsonl",
    "verify_delta_official.jsonl",
    "large_hits_positive_delta.jsonl",
    "wide_scalpel_results.jsonl",
    "znegsweep_verified.jsonl",
    "f0g4_lirec_results.jsonl",
]

# ── 6F5 Matrix recurrence ─────────────────────────────────────────────────────

def _esym_all(v: List) -> List:
    """Elementary symmetric polynomials e_0, e_1, ..., e_n for a list v."""
    import mpmath as mp
    n = len(v)
    e = [mp.mpf(0)] * (n + 1)
    e[0] = mp.mpf(1)
    for i in range(n):
        for k in range(i + 1, 0, -1):
            e[k] = e[k] + v[i] * e[k - 1]
    return e


def _build_matrix_6f5(n: int, shift: List[int], direction: List[int],
                       z_num: int, z_den: int):
    """Companion matrix B(n) for a 6F5 CMF recurrence at step n."""
    import mpmath as mp
    h_f = mp.mpf(-z_num)
    h_g = mp.mpf(z_den)
    f = [mp.mpf(shift[i] + n * direction[i] + 1) for i in range(6)]
    g = [mp.mpf(shift[6 + j] + n * direction[6 + j] + 2) for j in range(5)]
    ef = _esym_all(f)
    eg = _esym_all(g)
    while len(eg) < 7:
        eg.append(mp.mpf(0))
    c = [mp.mpf(0)] * 6
    c[0] = ef[6] * h_f
    for k in range(1, 6):
        c[k] = eg[6 - k] * h_g + ef[6 - k] * h_f
    M = mp.zeros(6)
    for row in range(1, 6):
        M[row, row - 1] = mp.mpf(1)
    for row in range(6):
        M[row, 5] = c[row]
    return M


def compute_delta_6f5(
    shift: List[int],
    direction: List[int],
    z_num: int,
    z_den: int,
    depth_n: int = 100,
    depth_2n: int = 200,
    num_row: int = 0,
    den_row: int = 1,
    dps: int = 400,
) -> Dict:
    """
    Compute the irrationality exponent δ(n, 2n) for a 6F5 trajectory.

    Returns dict with keys: delta, L_n, L_2n, q_digits, converged.
    Returns None if the walk fails (zero denominator, numerical issue).
    """
    try:
        import mpmath as mp
        mp.mp.dps = dps

        def _walk(d):
            P = mp.eye(6)
            for step in range(1, d + 1):
                P = P * _build_matrix_6f5(step, shift, direction, z_num, z_den)
            return P

        P_n = _walk(depth_n)
        P_2n = _walk(depth_2n)

        p_n = P_n[num_row, 5]
        q_n = P_n[den_row, 5]
        p_2n = P_2n[num_row, 5]
        q_2n = P_2n[den_row, 5]

        if abs(q_n) < 1e-100 or abs(q_2n) < 1e-100:
            return None

        L_n = float(p_n / q_n)
        L_2n = float(p_2n / q_2n)
        diff = abs(float(p_n / q_n - p_2n / q_2n))

        if diff < 1e-300 or abs(float(q_n)) < 1e-300:
            return None

        q_digits = float(mp.log10(abs(q_n)))
        log_diff = math.log10(diff) if diff > 0 else -300
        delta = -(1.0 + log_diff / q_digits) if q_digits > 0 else 0.0

        return {
            "delta": delta,
            "L_n": L_n,
            "L_2n": L_2n,
            "q_digits": q_digits,
            "log_diff": log_diff,
            "converged": diff < 1e-5,
        }
    except Exception as e:
        return None


def identify_constant(L: float, n_terms: int = 8, tol: int = 10) -> Optional[str]:
    """Try mpmath.identify on a limit value."""
    try:
        import mpmath as mp
        mp.mp.dps = 50
        result = mp.identify(mp.mpf(str(L)), tol=tol, maxcoeff=500)
        return str(result)
    except Exception:
        return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_6f5_data(sweeps_dir: Path,
                   min_delta: float = 0.05,
                   require_f0g4: bool = True) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Load 6F5 scan results and featurise for VAE training.

    Features (18 dims):
      shift[0:11]     — 11 integer shift parameters (f[0:6] and g[6:11])
      z_float         — z as a float (z_num/z_den)
      log_abs_z       — log10(|z|), sensitive to small-z regime
      adv_g_onehot[5] — one-hot encoding of which g[6..10] is advancing (0-4)

    Target: delta_verify (higher = better irrationality exponent).
    """
    records = []
    seen = set()

    for fname in DATA_FILES:
        fpath = sweeps_dir / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                shift = rec.get("shift")
                direction = rec.get("dir")
                if not shift or not direction or len(shift) != 11 or len(direction) != 11:
                    continue

                # f0g4 stratum: exactly 4 g-params have direction=0, 1 has direction=1
                # among indices 6-10 (the g-parameters)
                g_dirs = direction[6:11]
                n_active_g = sum(g_dirs)
                if require_f0g4 and n_active_g != 1:
                    continue

                # Delta filter
                delta = rec.get("delta_verify") or rec.get("delta_screen") or 0.0
                if delta < min_delta:
                    continue

                # Dedup by (shift, z_num, z_den, direction)
                z_str = rec.get("z", "0/1")
                key = (tuple(shift), tuple(direction), z_str)
                if key in seen:
                    continue
                seen.add(key)

                # Parse z
                z_num = rec.get("z_num")
                z_den = rec.get("z_den")
                if z_num is None or z_den is None:
                    try:
                        frac = Fraction(z_str)
                        z_num, z_den = frac.numerator, frac.denominator
                    except Exception:
                        continue
                if z_den == 0:
                    continue

                z_float = z_num / z_den
                log_abs_z = math.log10(abs(z_float)) if abs(z_float) > 1e-15 else -15.0

                # Advancing g index (0-4, mapping to g[6]-g[10])
                adv_g_onehot = [0.0] * 5
                adv_idx = next((i for i, d in enumerate(g_dirs) if d == 1), 0)
                adv_g_onehot[adv_idx] = 1.0

                feat = (
                    [float(s) for s in shift]
                    + [z_float, log_abs_z]
                    + adv_g_onehot
                )

                records.append({
                    "shift": shift,
                    "direction": direction,
                    "z_num": int(z_num),
                    "z_den": int(z_den),
                    "z_float": z_float,
                    "log_abs_z": log_abs_z,
                    "adv_g": adv_idx,
                    "delta": delta,
                    "feat": feat,
                    "source_file": fname,
                    "L": rec.get("L"),
                })

    if not records:
        raise ValueError(f"No f0g4 records found in {sweeps_dir} with δ > {min_delta}")

    X = np.array([r["feat"] for r in records], dtype=np.float32)
    y = np.array([r["delta"] for r in records], dtype=np.float32)
    return X, y, records


# ── VAE ───────────────────────────────────────────────────────────────────────

class F0G4_VAE(nn.Module):
    """VAE for the 6F5 f0g4 CMF coefficient space."""

    def __init__(self, in_dim: int = 18, hidden: List[int] = None, latent_dim: int = 8):
        super().__init__()
        if hidden is None:
            hidden = [64, 32]
        # Encoder
        enc_layers = []
        prev = in_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec_layers)
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def elbo(self, x, x_hat, mu, logvar, beta: float = 0.5):
        recon = F.mse_loss(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl


def train_f0g4_vae(
    X: np.ndarray, y: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta_max: float = 0.5,
    warmup: int = 40,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[F0G4_VAE, StandardScaler]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    Xt = torch.tensor(Xs)
    N = len(Xs)

    model = F0G4_VAE(in_dim=X.shape[1], latent_dim=latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        beta = beta_max * min(1.0, epoch / max(warmup, 1))
        perm = rng.permutation(N)
        losses = []
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xt[idx]
            x_hat, mu, logvar = model(xb)
            loss, _, _ = model.elbo(xb, x_hat, mu, logvar, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        sched.step()
        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  loss={mean_loss:.4f}  β={beta:.3f}")

    return model, scaler


# ── Candidate generation ──────────────────────────────────────────────────────

def encode_all(model: F0G4_VAE, Xs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(Xs))
    return mu.numpy()


def sample_near_high_delta(
    model: F0G4_VAE,
    scaler: StandardScaler,
    Z: np.ndarray,
    y: np.ndarray,
    n_candidates: int = 200,
    delta_threshold: float = 0.12,
    noise_scale: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Sample new candidates in the neighbourhood of high-δ records."""
    rng = np.random.default_rng(seed)
    high_idx = np.where(y >= delta_threshold)[0]
    if len(high_idx) == 0:
        high_idx = np.argsort(y)[-max(10, len(y) // 5):]
        print(f"  Warning: no records with δ≥{delta_threshold}. "
              f"Using top {len(high_idx)} records instead.")

    z_high = Z[high_idx]
    mu_ref = z_high.mean(axis=0)
    sigma_ref = z_high.std(axis=0) * noise_scale

    z_samples = (mu_ref + rng.normal(0, 1, (n_candidates, Z.shape[1])) * sigma_ref).astype(np.float32)

    model.eval()
    with torch.no_grad():
        x_hat = model.decode(torch.tensor(z_samples)).numpy()
    return scaler.inverse_transform(x_hat)


def decode_to_params(x_raw: np.ndarray) -> Optional[Dict]:
    """Convert a raw decoded feature vector back to (shift, direction, z_num, z_den)."""
    try:
        shift = [int(round(x_raw[i])) for i in range(11)]
        z_float = float(x_raw[11])
        # log_abs_z = x_raw[12]  # ignored — derived from z_float
        adv_g_logits = x_raw[13:18]
        adv_g_idx = int(np.argmax(adv_g_logits))

        direction = [0] * 11
        direction[6 + adv_g_idx] = 1

        # Convert z_float to a nearby simple fraction
        frac = Fraction(z_float).limit_denominator(100)
        if frac.denominator == 0:
            return None
        z_num, z_den = frac.numerator, frac.denominator

        return {
            "shift": shift,
            "direction": direction,
            "z_num": z_num,
            "z_den": z_den,
            "z_float": float(frac),
            "adv_g": adv_g_idx,
        }
    except Exception:
        return None


# ── Verification pipeline ─────────────────────────────────────────────────────

def verify_candidates(
    candidates_raw: np.ndarray,
    n_depth: int = 80,
    dps: int = 300,
    verbose: bool = True,
) -> List[Dict]:
    """Decode raw vectors → params → run δ check. Return verified hits."""
    results = []
    t0 = time.time()
    for i, row in enumerate(candidates_raw):
        params = decode_to_params(row)
        if params is None:
            continue

        res = compute_delta_6f5(
            shift=params["shift"],
            direction=params["direction"],
            z_num=params["z_num"],
            z_den=params["z_den"],
            depth_n=n_depth,
            depth_2n=n_depth * 2,
            dps=dps,
        )
        if res is None:
            continue

        delta = res["delta"]
        entry = {**params, **res, "candidate_idx": i}

        if verbose and delta > 0.05:
            print(f"  [{i:>4}] shift={params['shift'][:6]}...  "
                  f"z={params['z_num']}/{params['z_den']}  "
                  f"g[{6+params['adv_g']}]↑  "
                  f"δ={delta:.4f}")

        results.append(entry)

    results.sort(key=lambda r: r["delta"], reverse=True)
    elapsed = time.time() - t0
    if verbose:
        pos = sum(1 for r in results if r["delta"] > 0.1)
        print(f"\n  Verified {len(results)} candidates in {elapsed:.1f}s — "
              f"{pos} with δ > 0.10")
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualize_latent_delta(
    Z: np.ndarray, y: np.ndarray, records: List[Dict],
    out_dir: Path, method: str = "umap",
):
    """Project latent space to 2D, colour by δ value."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if method == "umap":
        try:
            from umap import UMAP
            Z2 = UMAP(n_components=2, random_state=42,
                      n_neighbors=min(15, len(Z) - 1)).fit_transform(Z)
            label = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
            label = "PCA"
    else:
        from sklearn.decomposition import PCA
        Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
        label = "PCA"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: coloured by δ
    sc = axes[0].scatter(Z2[:, 0], Z2[:, 1], c=y, cmap="plasma",
                          s=10, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label="δ (irrationality exponent)")
    axes[0].set_title(f"6F5 f0g4 Latent Space — {label}\nColoured by δ", fontsize=11)
    axes[0].set_xlabel(f"{label}-1"); axes[0].set_ylabel(f"{label}-2")
    axes[0].grid(True, alpha=0.2)

    # Right: coloured by advancing g-index
    adv_colors = [r["adv_g"] for r in records]
    cmap_g = cm.get_cmap("tab10", 5)
    sc2 = axes[1].scatter(Z2[:, 0], Z2[:, 1], c=adv_colors, cmap=cmap_g,
                           s=10, alpha=0.7, rasterized=True, vmin=0, vmax=4)
    cb = plt.colorbar(sc2, ax=axes[1], label="Advancing g-index (0=g[6]...4=g[10])")
    cb.set_ticks([0.4, 1.2, 2.0, 2.8, 3.6])
    cb.set_ticklabels(["g[6]", "g[7]", "g[8]", "g[9]", "g[10]"])
    axes[1].set_title(f"6F5 f0g4 Latent Space — {label}\nColoured by advancing g-param", fontsize=11)
    axes[1].set_xlabel(f"{label}-1"); axes[1].set_ylabel(f"{label}-2")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    out = out_dir / f"fig_6f5_latent_{label.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_delta_ladder(out_dir: Path):
    """Bar chart of the f0g4 → f4g4 delta ladder from the Degeneracy Atlas."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    strata = ["f0g4\n(full 6F5)", "f1g4\n(5-active)", "f2g4\n(4-active)",
              "f3g4\n(3F2)", "f4g4\n(2-active)"]
    delta_canonical = [0.148, 0.218, 0.324, 0.506, 1.006]
    delta_best = [0.208, None, None, 0.50, None]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6"]
    proof_status = ["Conjectured", "Atlas", "Atlas", "Proven ✓", "Atlas"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(5), delta_canonical, color=colors, alpha=0.8, width=0.5,
                  label="Canonical δ (Degeneracy Atlas)")
    for i, (bd, ps) in enumerate(zip(delta_best, proof_status)):
        if bd is not None:
            ax.scatter([i], [bd], color=colors[i], s=120, zorder=5,
                       marker="*", edgecolors="black", linewidth=0.5)
        ax.text(i, delta_canonical[i] + 0.02, ps,
                ha="center", fontsize=8, color=colors[i], fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_xticklabels(strata, fontsize=10)
    ax.set_ylabel("Irrationality Exponent δ", fontsize=11)
    ax.set_title("The 6F5 f0g4→f4g4 Delta Ladder\n"
                 "As g-zeros increase, active dimension decreases, δ increases",
                 fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="δ = 1.0 (strong irrationality)")
    star_patch = mpatches.Patch(color="gray", label="★ = Best sweep hit")
    ax.legend(handles=[ax.get_legend_handles_labels()[0][0], star_patch], fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = out_dir / "fig_delta_ladder.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_log_z_scan(out_dir: Path, sweeps_dir: Path):
    """Plot δ vs log|z| for all log-z scan hits."""
    import matplotlib.pyplot as plt

    fpath = sweeps_dir / "log_z_scan_deep.jsonl"
    if not fpath.exists():
        print("  log_z_scan_deep.jsonl not found, skipping plot")
        return

    log_z_vals, deltas = [], []
    with open(fpath) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            z_num = rec.get("z_num")
            z_den = rec.get("z_den")
            delta = rec.get("delta_verify") or rec.get("delta_screen")
            if z_num and z_den and delta and z_den != 0 and abs(z_num) > 0:
                z = abs(z_num / z_den)
                log_z_vals.append(math.log10(z))
                deltas.append(float(delta))

    if not log_z_vals:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(log_z_vals, deltas, s=4, alpha=0.3, color="#e74c3c", rasterized=True)

    # Zone boundaries
    for x, label in [(-2, "Zone I/II"), (-3, "Zone II/III"),
                     (-4, "Zone III/IV"), (-5, "Zone IV/V")]:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.text(x + 0.05, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.22,
                label, fontsize=7, color="gray", rotation=90)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("log₁₀|z|  (z-value of the CMF)")
    ax.set_ylabel("δ  (irrationality exponent)")
    ax.set_title("f0g4 Delta vs log|z| — 7020 Hits from Log-Z Scan\n"
                 "Small-|z| converges to δ ≈ 0.205 asymptote", fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "fig_log_z_delta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CMF 6F5 Explorer — f0g4 VAE search")
    p.add_argument("--sweeps-dir", type=str, default=str(DEFAULT_SWEEPS_DIR))
    p.add_argument("--out-dir",    type=str, default="results/6f5_explorer")
    p.add_argument("--checkpoint", type=str, default="results/6f5_vae.pt")

    p.add_argument("--train",      action="store_true", help="Train VAE on f0g4 data")
    p.add_argument("--visualize",  action="store_true", help="Generate latent space plots")
    p.add_argument("--generate",   type=int, default=0,
                   help="Generate N candidates from high-δ neighbourhood")
    p.add_argument("--verify",     action="store_true",
                   help="Run δ-verification on generated candidates")
    p.add_argument("--identify",   action="store_true",
                   help="Run mpmath.identify on best verified L values")
    p.add_argument("--full-run",   action="store_true",
                   help="Train + visualize + generate 300 + verify + identify")
    p.add_argument("--delta-plots", action="store_true",
                   help="Generate delta-ladder and log-z scan plots")

    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--latent-dim",  type=int, default=8)
    p.add_argument("--min-delta",   type=float, default=0.05)
    p.add_argument("--n-depth",     type=int, default=80,
                   help="Recurrence depth for δ-verification")
    p.add_argument("--quiet",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.full_run:
        args.train = True
        args.visualize = True
        args.generate = 300
        args.verify = True
        args.identify = True
        args.delta_plots = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    sweeps_dir = Path(args.sweeps_dir)
    verbose = not args.quiet

    print("=" * 65)
    print("CMF 6F5 EXPLORER — f0g4 Stratum VAE Search")
    print(f"  sweeps_dir = {sweeps_dir}")
    print(f"  out_dir    = {out_dir}")
    print("=" * 65)

    # Load data
    print("\nLoading 6F5 f0g4 data...")
    X, y, records = load_6f5_data(sweeps_dir, min_delta=args.min_delta)
    print(f"  Records: {len(records)}  δ range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  Features: {X.shape[1]} dims  "
          f"(shift×11 + z_float + log_abs_z + adv_g_onehot×5)")

    # Delta ladder + log-z plots
    if args.delta_plots:
        print("\nGenerating delta plots...")
        plot_delta_ladder(out_dir)
        plot_log_z_scan(out_dir, sweeps_dir)

    # Train or load VAE
    if args.train or not ckpt_path.exists():
        print(f"\nTraining VAE (epochs={args.epochs}  latent_dim={args.latent_dim})")
        model, scaler = train_f0g4_vae(
            X, y, latent_dim=args.latent_dim,
            epochs=args.epochs, verbose=verbose
        )
        torch.save({
            "model_state": model.state_dict(),
            "in_dim": X.shape[1],
            "latent_dim": args.latent_dim,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }, ckpt_path)
        print(f"  Checkpoint → {ckpt_path}")
    else:
        print(f"\nLoading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = F0G4_VAE(in_dim=ckpt["in_dim"], latent_dim=ckpt["latent_dim"])
        model.load_state_dict(ckpt["model_state"])
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt["scaler_mean"])
        scaler.scale_ = np.array(ckpt["scaler_scale"])

    # Encode all records
    print("\nEncoding all records to latent space...")
    Xs = scaler.transform(X).astype(np.float32)
    Z = encode_all(model, Xs)
    print(f"  Latent shape: {Z.shape}")

    # Visualise
    if args.visualize:
        print("\nGenerating latent space visualisation...")
        visualize_latent_delta(Z, y, records, out_dir)

    # Generate candidates
    if args.generate > 0:
        print(f"\nSampling {args.generate} candidates from high-δ region (δ≥0.12)...")
        candidates_raw = sample_near_high_delta(
            model, scaler, Z, y,
            n_candidates=args.generate,
            delta_threshold=0.12,
        )

        # Verify with real δ computation
        if args.verify:
            print(f"\nVerifying {args.generate} candidates "
                  f"(n={args.n_depth}, dps=300)...")
            print("  [shift[:6]...   z_num/z_den   g-dir]  →  δ")
            verified = verify_candidates(
                candidates_raw,
                n_depth=args.n_depth,
                dps=300,
                verbose=verbose,
            )

            # Save results
            out_json = out_dir / "verified_candidates.json"
            with open(out_json, "w") as f:
                json.dump(
                    [
                        {k: v for k, v in r.items()
                         if k not in ("shift_f", "dir_f")}
                        for r in verified[:100]
                    ],
                    f, indent=2,
                    default=lambda x: float(x) if hasattr(x, "__float__") else str(x),
                )
            print(f"\n  Top verified candidates → {out_json}")

            # Print top-10
            top10 = [r for r in verified if r["delta"] > 0.05][:10]
            if top10:
                print("\n  ── Top-10 new candidates ──")
                for r in top10:
                    print(f"  δ={r['delta']:.4f}  "
                          f"z={r['z_num']}/{r['z_den']}  "
                          f"g[{6+r['adv_g']}]↑  "
                          f"shift={r['shift']}")

            # Identify best L values
            if args.identify:
                print("\nRunning mpmath.identify on top candidates...")
                id_results = []
                for r in verified[:20]:
                    if r.get("L_n") is None:
                        continue
                    L = r["L_n"]
                    ident = identify_constant(L)
                    status = ident if ident else "not identified"
                    print(f"  δ={r['delta']:.4f}  L={L:.12f}  → {status}")
                    id_results.append({**r, "identification": status})

                out_id = out_dir / "identified_constants.json"
                with open(out_id, "w") as f:
                    json.dump(id_results, f, indent=2,
                              default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
                print(f"\n  Identification results → {out_id}")

        else:
            # Just save raw decoded params without δ check
            params_list = [decode_to_params(row) for row in candidates_raw]
            params_list = [p for p in params_list if p is not None]
            out_raw = out_dir / "raw_candidates.json"
            with open(out_raw, "w") as f:
                json.dump(params_list, f, indent=2,
                          default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
            print(f"  {len(params_list)} raw candidates (no δ check) → {out_raw}")

    print(f"\n{'='*65}")
    print(f"DONE  Results → {out_dir}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
