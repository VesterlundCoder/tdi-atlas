#!/usr/bin/env python3
"""
Astronomical Table Predictor — fill uncertain cells in new manuscript tables
=============================================================================
Takes a TSV file (same format as vat_gr_1291_*.tsv) with cells marked [?],
runs both trained models (flat + topo), and outputs:
  1. A corrected TSV with model predictions substituted for [?] cells
  2. A human-readable comparison table printed to stdout

Usage:
  python predict_table.py --input new_table.tsv
  python predict_table.py --input new_table.tsv --out corrected.tsv --model topo
  python predict_table.py --klima <lat_deg>       # predict full table for a klima
  python predict_table.py --list-signs            # show zodiac λ mapping

Supported --model values: flat | topo | both (default: both)

Training the models first:
  python train_astro_models.py

Input TSV format (tab-separated, same as training data):
  sign  deg  anaph_deg_raw  anaph_min_raw  hour_raw  chron_raw  deg_dec  anaph_dec  hour_dec  chron_dec
  Aries  α  ō  λβ  ιε  δ  1  0;32  15  4
  Aries  β  α  δ  ιε  η  2  [?]  15  8     ← uncertain anaph_dec
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Shared definitions (mirrors train_astro_models.py) ────────────────────────

ASTRO_DIR = Path(__file__).parent
CKPT_DIR  = ASTRO_DIR / "astro_ckpt"

SIGN_ORDER = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_BASE = {s: i * 30 for i, s in enumerate(SIGN_ORDER)}
COL_NAMES = ["anaph", "hour", "chron"]


class AstroMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=3, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def _load_model(path: Path, in_dim: int, hidden: int = 64) -> AstroMLP:
    m = AstroMLP(in_dim, hidden)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def flat_features(lam: np.ndarray) -> np.ndarray:
    return (lam / 360.0).reshape(-1, 1).astype(np.float32)


def topo_features(lam: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    import math
    theta = 2.0 * math.pi * lam / 360.0
    cols  = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))
    return np.stack(cols, axis=1).astype(np.float32)


def denorm(Y_norm: np.ndarray, stats: dict) -> np.ndarray:
    mu    = np.array(stats["mu"],    dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    return Y_norm * sigma + mu


def _predict(model, X: np.ndarray, stats: dict) -> np.ndarray:
    with torch.no_grad():
        pred_norm = model(torch.tensor(X)).numpy()
    return denorm(pred_norm, stats)


def _strip(s: str):
    return s.replace("[?]", "").strip(), "[?]" in s


def _sexag_to_float(s: str) -> float:
    s, _ = _strip(s)
    if ";" in s:
        d, m = s.split(";", 1)
        return float(d) + float(m) / 60.0
    return float(s)


def _float_to_sexag(v: float) -> str:
    d = int(v)
    m = round((v - d) * 60)
    if m == 60:
        d += 1; m = 0
    return f"{d};{m:02d}"


# ── TSV parsing ───────────────────────────────────────────────────────────────

def parse_tsv(path: Path):
    """
    Returns list of dicts with fields:
      sign, deg, lam, anaph, hour, chron,
      unc_anaph, unc_hour, unc_chron,
      raw_parts (original tab-split line)
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("sign"):
                rows.append({"_meta": "skip", "_raw": line})
                continue

            parts = line.split("\t")
            if len(parts) < 10:
                rows.append({"_meta": "skip", "_raw": line})
                continue

            sign      = parts[0].strip()
            deg_raw   = parts[6].strip()
            anaph_raw = parts[7].strip()
            hour_raw  = parts[8].strip()
            chron_raw = parts[9].strip()

            if sign not in SIGN_BASE:
                rows.append({"_meta": "skip", "_raw": line})
                continue

            deg_str, unc_deg = _strip(deg_raw)
            try:
                deg = int(deg_str)
            except ValueError:
                rows.append({"_meta": "skip", "_raw": line})
                continue

            lam = SIGN_BASE[sign] + deg

            try:
                anaph = _sexag_to_float(anaph_raw)
                hour  = float(_strip(hour_raw)[0])
                chron = float(_strip(chron_raw)[0])
            except (ValueError, IndexError):
                rows.append({"_meta": "skip", "_raw": line})
                continue

            _, unc_a = _strip(anaph_raw)
            _, unc_h = _strip(hour_raw)
            _, unc_c = _strip(chron_raw)

            rows.append(dict(
                _meta="data", _raw=line, _parts=parts,
                sign=sign, deg=deg, lam=lam,
                anaph=anaph, hour=hour, chron=chron,
                unc_anaph=unc_a, unc_hour=unc_h, unc_chron=unc_c,
            ))
    return rows


# ── Prediction + output ────────────────────────────────────────────────────────

def predict_and_fill(rows, flat_model, topo_model, stats, harmonics,
                     use_model="both"):
    """
    For each uncertain cell, predict using selected model(s).
    Returns updated rows with _pred_anaph, _pred_hour, _pred_chron injected.
    """
    data_rows = [r for r in rows if r.get("_meta") == "data"]
    if not data_rows:
        return rows

    lam_arr = np.array([r["lam"] for r in data_rows], dtype=np.float32)
    X_f = flat_features(lam_arr)
    X_t = topo_features(lam_arr, harmonics)

    pred_f = _predict(flat_model, X_f, stats) if use_model in ("flat", "both") else None
    pred_t = _predict(topo_model, X_t, stats) if use_model in ("topo", "both") else None

    data_idx = 0
    for r in rows:
        if r.get("_meta") != "data":
            continue
        r["_pred_flat"]  = pred_f[data_idx] if pred_f is not None else None
        r["_pred_topo"]  = pred_t[data_idx] if pred_t is not None else None
        data_idx += 1

    return rows


def write_corrected_tsv(rows, out_path: Path, use_model: str):
    """Write TSV with [?] cells replaced by model prediction."""
    model_key = "_pred_topo" if use_model == "topo" else "_pred_flat"

    lines = []
    for r in rows:
        if r.get("_meta") != "data":
            lines.append(r["_raw"])
            continue

        pred = r.get(model_key)
        parts = list(r["_parts"])   # copy

        if pred is not None:
            if r["unc_anaph"]:
                parts[7] = _float_to_sexag(float(pred[0])) + "[pred]"
            if r["unc_hour"]:
                parts[8] = str(round(float(pred[1]))) + "[pred]"
            if r["unc_chron"]:
                parts[9] = str(round(float(pred[2]))) + "[pred]"

        lines.append("\t".join(parts))

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  Corrected TSV written → {out_path}")


def print_summary(rows, use_model: str):
    """Print a human-readable table of uncertain cells + predictions."""
    uncertain = [r for r in rows
                 if r.get("_meta") == "data"
                 and (r["unc_anaph"] or r["unc_hour"] or r["unc_chron"])]

    if not uncertain:
        print("  No uncertain cells found in input table.")
        return

    print(f"\n── Uncertain cells: {len(uncertain)} rows ──────────────────────────────────")
    print(f"  {'Sign':<12} {'λ':>4}  "
          f"{'anaph(t)':>9} {'F':>8} {'T':>8}  "
          f"{'hour(t)':>7} {'F':>6} {'T':>6}  "
          f"{'chron(t)':>8} {'F':>6} {'T':>6}  uncertain cols")
    print("  " + "-" * 90)

    for r in uncertain:
        pf = r.get("_pred_flat")
        pt = r.get("_pred_topo")

        def fmt(val, pred, flag, fmt_v=".3f", fmt_p=".2f"):
            marker = " [?]" if flag else "     "
            pf_s = f"{pred[0]:{fmt_p}}" if (pred is not None and flag) else "  ---"
            pt_s = f"{pred[1]:{fmt_p}}" if (pred is not None and flag) else "  ---"
            return f"{val:{fmt_v}}{marker}", pf_s, pt_s

        a_v, af, at = fmt(r["anaph"], [pf[0] if pf is not None else None,
                                        pt[0] if pt is not None else None],
                          r["unc_anaph"])
        h_v, hf, ht = fmt(r["hour"],  [pf[1] if pf is not None else None,
                                        pt[1] if pt is not None else None],
                          r["unc_hour"], ".0f", ".1f")
        c_v, cf, ct = fmt(r["chron"], [pf[2] if pf is not None else None,
                                        pt[2] if pt is not None else None],
                          r["unc_chron"], ".0f", ".1f")

        unc_cols = "+".join(
            c for c, f in [("anaph", r["unc_anaph"]),
                            ("hour",  r["unc_hour"]),
                            ("chron", r["unc_chron"])] if f)

        print(f"  {r['sign']:<12} {r['lam']:>4}  "
              f"{a_v:>9} {af:>8} {at:>8}  "
              f"{h_v:>7} {hf:>6} {ht:>6}  "
              f"{c_v:>8} {cf:>6} {ct:>6}  {unc_cols}")


# ── Full-klima generation ──────────────────────────────────────────────────────

def generate_full_klima(flat_model, topo_model, stats, harmonics):
    """
    Generate predicted full oblique-ascension table for all 360° of the zodiac
    using both models. Useful for comparing against a new manuscript klima table.
    """
    lam = np.arange(1, 361, dtype=np.float32)
    X_f = flat_features(lam)
    X_t = topo_features(lam, harmonics)
    pred_f = _predict(flat_model, X_f, stats)
    pred_t = _predict(topo_model, X_t, stats)

    print("\n── Full zodiac table (predicted) ────────────────────────────────────")
    print(f"  {'Sign':<12} {'λ':>4}  "
          f"{'flat_anaph':>11}  {'topo_anaph':>11}  "
          f"{'flat_h':>8}  {'topo_h':>8}  "
          f"{'flat_c':>8}  {'topo_c':>8}")
    for i in range(360):
        s_idx = min(int(lam[i] - 1) // 30, 11)
        sign  = SIGN_ORDER[s_idx]
        if (int(lam[i]) - 1) % 30 == 0 or i == 0:  # sign boundary
            print(f"  {'─'*90}")
        print(f"  {sign:<12} {int(lam[i]):>4}  "
              f"{pred_f[i,0]:>11.3f}  {pred_t[i,0]:>11.3f}  "
              f"{pred_f[i,1]:>8.2f}  {pred_t[i,1]:>8.2f}  "
              f"{pred_f[i,2]:>8.2f}  {pred_t[i,2]:>8.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",       type=Path, help="Input TSV with [?] cells")
    ap.add_argument("--out",         type=Path, help="Output corrected TSV path")
    ap.add_argument("--model",       default="both",
                    choices=["flat", "topo", "both"],
                    help="Which model to use for correction (default: both)")
    ap.add_argument("--harmonics",   type=int, default=3,
                    help="Fourier harmonics (must match training, default 3)")
    ap.add_argument("--hidden",      type=int, default=64,
                    help="MLP hidden size (must match training, default 64)")
    ap.add_argument("--full-klima",  action="store_true",
                    help="Print full predicted 360° table (no input TSV needed)")
    ap.add_argument("--list-signs",  action="store_true",
                    help="Print zodiac sign → ecliptic longitude mapping and exit")
    args = ap.parse_args()

    if args.list_signs:
        print("Zodiac sign → ecliptic longitude mapping:")
        for s in SIGN_ORDER:
            b = SIGN_BASE[s]
            print(f"  {s:<12} λ = {b+1:3d} – {b+30:3d}°")
        return

    # Load checkpoints
    scaler_path = CKPT_DIR / "scaler.json"
    flat_path   = CKPT_DIR / "flat_model.pt"
    topo_path   = CKPT_DIR / "topo_model.pt"

    for p in [scaler_path, flat_path, topo_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run train_astro_models.py first.", file=sys.stderr)
            sys.exit(1)

    with open(scaler_path) as fh:
        stats = json.load(fh)

    flat_in = 1
    topo_in = 2 * args.harmonics

    flat_model = _load_model(flat_path, flat_in, args.hidden)
    topo_model = _load_model(topo_path, topo_in, args.hidden)
    print(f"Loaded flat ({flat_in}D) + topo ({topo_in}D) models from {CKPT_DIR}/")

    if args.full_klima:
        generate_full_klima(flat_model, topo_model, stats, args.harmonics)
        return

    if not args.input:
        ap.print_help()
        print("\nERROR: --input required (or use --full-klima).", file=sys.stderr)
        sys.exit(1)

    if not args.input.exists():
        print(f"ERROR: {args.input} not found.", file=sys.stderr)
        sys.exit(1)

    # Parse + predict
    rows = parse_tsv(args.input)
    data_rows = [r for r in rows if r.get("_meta") == "data"]
    unc_rows  = [r for r in data_rows
                 if r["unc_anaph"] or r["unc_hour"] or r["unc_chron"]]
    print(f"  {len(data_rows)} data rows, {len(unc_rows)} with uncertain cells")

    rows = predict_and_fill(rows, flat_model, topo_model, stats,
                            args.harmonics, args.model)
    print_summary(rows, args.model)

    out_path = args.out or args.input.with_suffix(".predicted.tsv")
    write_corrected_tsv(rows, out_path, use_model="topo" if args.model == "topo" else "flat")


if __name__ == "__main__":
    main()
