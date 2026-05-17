#!/usr/bin/env python3
"""
Astronomical Table Model Trainer — Vat.gr.1291 Handy Tables (Ptolemy / Oblique Ascensions)
=============================================================================
Trains two small MLP models on the transcribed zodiac table data:

  FLAT model  — uses linear ecliptic longitude feature λ/360 ∈ [0,1]
                topology: maps S¹ to interval ℝ → discontinuity at 360°→0°

  TOPO model  — uses Fourier (circular) features [sin(kλ), cos(kλ)] on S¹
                topology: preserves the circular S¹ structure of the zodiac
                natural basis for periodic functions (hour_dec is cos-like)

Key motivation for topological enrichment:
  The zodiac is homeomorphic to S¹ (circle), not to [0,1] (interval).
  The hour column (seasonal daylength variation) is ~ A + B·cos(2πλ/360),
  which Fourier features represent exactly with k=1, while the flat feature
  requires the MLP to learn a non-smooth V-shape near the 360°/0° boundary.
  This matters especially for predicting Capricorn–Pisces (λ 271–360°),
  which are missing from our training data but adjacent on S¹ to Aries.

Targets (3 columns per table row):
  anaph  — cumulative oblique ascension (degrees, sexagesimal X;Y → float)
  hour   — seasonal hours (integer, periodic with period 360°)
  chron  — time-degrees / chronoi (integer 0–59, sub-hour)

Usage:
  python train_astro_models.py                  # train + eval, 2000 epochs
  python train_astro_models.py --epochs 5000    # longer run
  python train_astro_models.py --eval-only      # load + evaluate only
  python train_astro_models.py --harmonics 4    # more Fourier modes
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────

ASTRO_DIR = Path(__file__).parent
TSV_FILES = sorted(ASTRO_DIR.glob("vat_gr_1291_*.tsv"))
CKPT_DIR  = ASTRO_DIR / "astro_ckpt"

SIGN_ORDER = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_BASE = {s: i * 30 for i, s in enumerate(SIGN_ORDER)}   # sign → λ offset

COL_NAMES  = ["anaph", "hour", "chron"]
COL_LABELS = ["anaph (°)", "hour (h)", "chron (min)"]

# ── Data loading ──────────────────────────────────────────────────────────────

def _strip_uncertain(s: str):
    """Remove [?] marker and return (value_str, is_uncertain)."""
    uncertain = "[?]" in s
    return s.replace("[?]", "").strip(), uncertain


def _parse_sexagesimal(s: str) -> float:
    """Parse Ptolemaic sexagesimal notation 'X;Y' → X + Y/60 degrees."""
    s, _ = _strip_uncertain(s)
    if ";" in s:
        d, m = s.split(";", 1)
        return float(d) + float(m) / 60.0
    return float(s)


def load_tsv_files(paths):
    """
    Parse all TSV files and return two lists of dicts:
      rows_clean    — no uncertain cells (used for training)
      rows_uncertain — at least one [?] cell (held out for evaluation)
    """
    rows_clean, rows_uncertain = [], []

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("sign"):   # header row
                    continue

                parts = line.split("\t")
                if len(parts) < 10:
                    continue

                sign      = parts[0].strip()
                deg_raw   = parts[6].strip()   # deg_dec (decimal int 1-30)
                anaph_raw = parts[7].strip()   # anaph_dec  "X;Y"
                hour_raw  = parts[8].strip()   # hour_dec   integer
                chron_raw = parts[9].strip()   # chron_dec  integer

                if sign not in SIGN_BASE:
                    continue

                # Degree within sign
                deg_str, unc_deg = _strip_uncertain(deg_raw)
                try:
                    deg = int(deg_str)
                except ValueError:
                    continue

                lam = SIGN_BASE[sign] + deg     # ecliptic longitude 1..360

                # Targets
                unc_any = unc_deg
                try:
                    anaph_str, unc_a = _strip_uncertain(anaph_raw)
                    anaph = _parse_sexagesimal(anaph_raw)
                    unc_any = unc_any or unc_a
                except (ValueError, IndexError):
                    continue

                try:
                    hour_str, unc_h = _strip_uncertain(hour_raw)
                    hour = float(hour_str)
                    unc_any = unc_any or unc_h
                except ValueError:
                    continue

                try:
                    chron_str, unc_c = _strip_uncertain(chron_raw)
                    chron = float(chron_str)
                    unc_any = unc_any or unc_c
                except ValueError:
                    continue

                record = dict(
                    sign=sign, deg=deg, lam=lam,
                    anaph=anaph, hour=hour, chron=chron,
                    uncertain=unc_any, source=path.name,
                )
                (rows_uncertain if unc_any else rows_clean).append(record)

    return rows_clean, rows_uncertain


# ── Feature engineering ───────────────────────────────────────────────────────

def flat_features(lam: np.ndarray) -> np.ndarray:
    """
    Flat (interval) feature: λ/360 ∈ [0,1].
    Topology: maps the zodiac circle to a line segment [0,1].
    Discontinuity: Pisces (λ≈360) → Aries (λ≈1) are far apart (distance ≈1).
    Shape: (N, 1).
    """
    return (lam / 360.0).reshape(-1, 1).astype(np.float32)


def topo_features(lam: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    """
    Topological (circular S¹) Fourier features:
      [sin(kθ), cos(kθ)]  for k = 1 .. n_harmonics,  θ = 2π·λ/360.

    Topology: the zodiac maps to S¹; these are eigenfunctions of the
    Laplacian on S¹ — the natural orthonormal basis for periodic functions.
    Discontinuity at 360°→0° is eliminated: sin/cos wrap smoothly.

    Why this matters for 'hour':
      hour_dec ≈ A + B·cos(θ)   (peaks at Cancer θ≈π/2, troughs at Capricorn)
      This is exactly represented with k=1 Fourier mode.
      A flat model requires the MLP to approximate this with a V-shape.

    Shape: (N, 2·n_harmonics).
    """
    theta = 2.0 * math.pi * lam / 360.0
    cols  = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))
    return np.stack(cols, axis=1).astype(np.float32)


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_stats(rows):
    """Compute mean + std of the three target columns over rows."""
    arr = np.array([[r["anaph"], r["hour"], r["chron"]] for r in rows],
                   dtype=np.float32)
    mu    = arr.mean(axis=0)
    sigma = arr.std(axis=0) + 1e-8
    return {"mu": mu.tolist(), "sigma": sigma.tolist()}


def to_tensors(rows, stats, device):
    arr   = np.array([[r["anaph"], r["hour"], r["chron"]] for r in rows],
                     dtype=np.float32)
    mu    = np.array(stats["mu"],    dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    Y_raw = arr.copy()
    Y_norm = (arr - mu) / sigma
    return (torch.tensor(Y_norm, device=device),
            torch.tensor(Y_raw,  device=device))


def denorm(Y_norm: np.ndarray, stats) -> np.ndarray:
    mu    = np.array(stats["mu"],    dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    return Y_norm * sigma + mu


# ── Model ─────────────────────────────────────────────────────────────────────

class AstroMLP(nn.Module):
    """
    Small 3-layer MLP for astronomical table value regression.
    in_dim=1  → flat model (1 linear feature)
    in_dim=2K → topo model (K Fourier sine+cosine pairs on S¹)
    """
    def __init__(self, in_dim: int, hidden: int = 64, n_out: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X: np.ndarray, Y_norm_t: torch.Tensor,
                epochs: int, lr: float, weight_decay: float,
                device: str, label: str) -> list:
    model.to(device)
    Xt  = torch.tensor(X, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    log = []
    for ep in range(1, epochs + 1):
        model.train()
        pred = model(Xt)
        loss = nn.functional.mse_loss(pred, Y_norm_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            log.append((ep, loss.item()))
            print(f"  [{label}] ep {ep:5d}/{epochs}  loss={loss.item():.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

    return log


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X: np.ndarray, Y_raw: np.ndarray,
             stats: dict, device: str):
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(X, device=device)).cpu().numpy()
    pred = denorm(pred_norm, stats)
    rmse = np.sqrt(((pred - Y_raw) ** 2).mean(axis=0))
    mae  = np.abs(pred - Y_raw).mean(axis=0)
    return pred, rmse, mae


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs",     type=int,   default=2000)
    ap.add_argument("--hidden",     type=int,   default=64)
    ap.add_argument("--harmonics",  type=int,   default=3,
                    help="Fourier harmonics for topo model (default 3 → 6D input)")
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--wd",         type=float, default=1e-4,
                    help="Weight decay (L2 regularisation)")
    ap.add_argument("--dropout",    type=float, default=0.0)
    ap.add_argument("--device",     type=str,   default="cpu")
    ap.add_argument("--eval-only",  action="store_true",
                    help="Skip training, load saved checkpoints and evaluate")
    args = ap.parse_args()

    CKPT_DIR.mkdir(exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"Loading TSV files from {ASTRO_DIR} …")
    if not TSV_FILES:
        raise FileNotFoundError(f"No vat_gr_1291_*.tsv found in {ASTRO_DIR}")

    rows_clean, rows_unc = load_tsv_files(TSV_FILES)
    print(f"  Loaded {len(rows_clean)} clean rows, {len(rows_unc)} uncertain rows")
    print(f"  Signs in data: {sorted({r['sign'] for r in rows_clean})}")
    print(f"  λ range (clean): {min(r['lam'] for r in rows_clean)}°"
          f" – {max(r['lam'] for r in rows_clean)}°")

    if not rows_clean:
        raise ValueError("No clean rows found — check TSV format")

    # ── Build features + targets ───────────────────────────────────────────
    lam_clean = np.array([r["lam"] for r in rows_clean], dtype=np.float32)

    X_flat  = flat_features(lam_clean)
    X_topo  = topo_features(lam_clean, args.harmonics)

    stats = compute_stats(rows_clean)
    with open(CKPT_DIR / "scaler.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    Y_norm_t, Y_raw_t = to_tensors(rows_clean, stats, args.device)
    Y_raw = Y_raw_t.cpu().numpy()

    flat_in = X_flat.shape[1]   # 1
    topo_in = X_topo.shape[1]   # 2 × harmonics

    print(f"\nFlat model  input dim: {flat_in}")
    print(f"Topo model  input dim: {topo_in}  ({args.harmonics} harmonics × 2)")

    flat_model = AstroMLP(flat_in, args.hidden, dropout=args.dropout)
    topo_model = AstroMLP(topo_in, args.hidden, dropout=args.dropout)

    n_flat = sum(p.numel() for p in flat_model.parameters())
    n_topo = sum(p.numel() for p in topo_model.parameters())
    print(f"Flat params: {n_flat}   Topo params: {n_topo}")

    # ── Train ──────────────────────────────────────────────────────────────
    if not args.eval_only:
        print(f"\n── Training FLAT model ({'CPU' if args.device=='cpu' else args.device}) ──")
        train_model(flat_model, X_flat, Y_norm_t,
                    args.epochs, args.lr, args.wd, args.device, "FLAT")
        torch.save(flat_model.state_dict(), CKPT_DIR / "flat_model.pt")
        print(f"  Saved → {CKPT_DIR / 'flat_model.pt'}")

        print(f"\n── Training TOPO model ──")
        train_model(topo_model, X_topo, Y_norm_t,
                    args.epochs, args.lr, args.wd, args.device, "TOPO")
        torch.save(topo_model.state_dict(), CKPT_DIR / "topo_model.pt")
        print(f"  Saved → {CKPT_DIR / 'topo_model.pt'}")

    else:
        flat_model.load_state_dict(
            torch.load(CKPT_DIR / "flat_model.pt", map_location="cpu"))
        topo_model.load_state_dict(
            torch.load(CKPT_DIR / "topo_model.pt", map_location="cpu"))
        print("  Loaded saved checkpoints.")

    # ── Evaluate on training data ──────────────────────────────────────────
    pred_flat, rmse_flat, mae_flat = evaluate(
        flat_model, X_flat, Y_raw, stats, args.device)
    pred_topo, rmse_topo, mae_topo = evaluate(
        topo_model, X_topo, Y_raw, stats, args.device)

    print("\n── Training-set metrics ──────────────────────────────────────────")
    print(f"  {'Column':<14} {'FLAT RMSE':>10} {'FLAT MAE':>10} "
          f"{'TOPO RMSE':>10} {'TOPO MAE':>10}")
    for i, col in enumerate(COL_LABELS):
        print(f"  {col:<14} {rmse_flat[i]:>10.4f} {mae_flat[i]:>10.4f}"
              f" {rmse_topo[i]:>10.4f} {mae_topo[i]:>10.4f}")

    # ── Evaluate on uncertain cells ────────────────────────────────────────
    if rows_unc:
        lam_unc  = np.array([r["lam"] for r in rows_unc], dtype=np.float32)
        X_fu     = flat_features(lam_unc)
        X_tu     = topo_features(lam_unc, args.harmonics)
        Y_unc_raw = np.array(
            [[r["anaph"], r["hour"], r["chron"]] for r in rows_unc],
            dtype=np.float32)

        pred_fu, rmse_fu, mae_fu = evaluate(
            flat_model, X_fu, Y_unc_raw, stats, args.device)
        pred_tu, rmse_tu, mae_tu = evaluate(
            topo_model, X_tu, Y_unc_raw, stats, args.device)

        print("\n── Uncertain-cell metrics (held-out, marked [?] in source) ──────")
        print(f"  {'Column':<14} {'FLAT RMSE':>10} {'FLAT MAE':>10} "
              f"{'TOPO RMSE':>10} {'TOPO MAE':>10}")
        for i, col in enumerate(COL_LABELS):
            print(f"  {col:<14} {rmse_fu[i]:>10.4f} {mae_fu[i]:>10.4f}"
                  f" {rmse_tu[i]:>10.4f} {mae_tu[i]:>10.4f}")

        print("\n── Per-cell predictions on uncertain rows ────────────────────────")
        hdr = (f"  {'Sign':<12} {'λ':>4} "
               f"{'transcr_anaph':>14} {'flat':>8} {'topo':>8} | "
               f"{'transcr_hour':>13} {'flat':>6} {'topo':>6} | "
               f"{'transcr_chron':>13} {'flat':>6} {'topo':>6}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, r in enumerate(rows_unc):
            print(
                f"  {r['sign']:<12} {r['lam']:>4}  "
                f"{r['anaph']:>13.3f}  {pred_fu[i,0]:>7.3f}  {pred_tu[i,0]:>7.3f} | "
                f"{r['hour']:>12.0f}  {pred_fu[i,1]:>5.1f}  {pred_tu[i,1]:>5.1f} | "
                f"{r['chron']:>12.0f}  {pred_fu[i,2]:>5.1f}  {pred_tu[i,2]:>5.1f}"
            )

    # ── Out-of-range prediction: Capricorn–Pisces ─────────────────────────
    print("\n── Extrapolation: Capricorn–Pisces (λ 271–360°, not in training) ─")
    lam_ext  = np.arange(271, 361, dtype=np.float32)
    X_fe     = flat_features(lam_ext)
    X_te     = topo_features(lam_ext, args.harmonics)

    flat_model.eval(); topo_model.eval()
    with torch.no_grad():
        pred_fe_n = flat_model(torch.tensor(X_fe)).numpy()
        pred_te_n = topo_model(torch.tensor(X_te)).numpy()

    pred_fe = denorm(pred_fe_n, stats)
    pred_te = denorm(pred_te_n, stats)

    sign_labels = []
    for l in lam_ext:
        s_idx  = min(int((l - 1) // 30), 11)
        sign_labels.append(SIGN_ORDER[s_idx])

    print(f"  {'Sign':<12} {'λ':>4}  {'flat_anaph':>11}  {'topo_anaph':>11} | "
          f"{'flat_hour':>10}  {'topo_hour':>10} | "
          f"{'flat_chron':>11}  {'topo_chron':>11}")
    for i in range(0, len(lam_ext), 10):
        l = int(lam_ext[i])
        print(f"  {sign_labels[i]:<12} {l:>4}  "
              f"{pred_fe[i,0]:>11.3f}  {pred_te[i,0]:>11.3f} | "
              f"{pred_fe[i,1]:>10.2f}  {pred_te[i,1]:>10.2f} | "
              f"{pred_fe[i,2]:>11.2f}  {pred_te[i,2]:>11.2f}")

    print(f"\n  Note: TOPO predictions use circular Fourier features and can")
    print(f"  interpolate around S¹; FLAT extrapolates beyond λ=270° linearly.")
    print(f"\nCheckpoints saved to {CKPT_DIR}/")
    print("  flat_model.pt, topo_model.pt, scaler.json")
    print("\nNext: python predict_table.py --input <new_table.tsv>")


if __name__ == "__main__":
    main()
