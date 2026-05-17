#!/usr/bin/env python3
"""
Astronomical Table Scanner — apply flat + topo models to all manuscripts
========================================================================
For each manuscript in the corpus config:
  1. Load the IIIF manifest metadata (if downloaded)
  2. For Handy Tables manuscripts with Klima V content:
     - Generate full predicted transcription (360 rows × 3 columns)
     - Mark each value with [pred_flat] or [pred_topo]
     - Write TSV to manuscripts/<dir>/predicted_tables/
  3. For other table types:
     - Generate a stub TSV explaining why the model does not apply
  4. Write a master scan index JSON

Output for each scanned manuscript:
  manuscripts/<dir>/predicted_tables/
    klima5_full_flat.tsv        — flat model, all 12 signs × 30°
    klima5_full_topo.tsv        — topo model, all 12 signs × 30°
    klima5_ensemble.tsv         — best-of-both (flat for anaph, topo for hour+chron)
    scan_result.json            — per-manuscript metadata + confidence scores

Usage:
  python scan_tables.py                      # scan all manuscripts
  python scan_tables.py --priority 1         # priority 1 only
  python scan_tables.py --signum Vat.gr.208  # single manuscript

Understanding the output TSVs:
  - anaph_dec column:  sexagesimal X;Y prediction, labeled [pred_flat] or [pred_topo]
  - hour_dec column:   integer prediction
  - chron_dec column:  integer prediction
  - *_raw columns:     [unread] — image OCR not yet performed
  - The [pred] label on every cell means "model prediction, not yet verified against image"
  - When you manually transcribe a folio, replace [pred] values with actual readings
    and run predict_table.py to fill remaining uncertain cells

Model applicability:
  DIRECT  — Klima V oblique ascensions: model trained on this exact data
  KLIMA   — other klimata of same oblique ascension table type: wrong values, right structure
  DERIVED — Paradosis-type (HT adaptation): close values, higher uncertainty
  INCOMPATIBLE — Persian/Arabic/other: wrong values entirely
"""

import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────────

ASTRO_DIR   = Path(__file__).parent
CONFIG_PATH = ASTRO_DIR / "manuscripts" / "config.json"
MS_BASE     = ASTRO_DIR / "manuscripts"
CKPT_DIR    = ASTRO_DIR / "astro_ckpt"

SIGN_ORDER = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]
SIGN_BASE = {s: i * 30 for i, s in enumerate(SIGN_ORDER)}

GREEK_SIGNS = {
    "Aries":       "ΚΡΙΟΣ",
    "Taurus":      "ΤΑΥΡΟΣ",
    "Gemini":      "ΔΙΔΥΜΟΙ",
    "Cancer":      "ΚΑΡΚΙΝΟΣ",
    "Leo":         "ΛΕΩΝ",
    "Virgo":       "ΠΑΡΘΕΝΟΣ",
    "Libra":       "ΖΥΓΟΣ",
    "Scorpio":     "ΣΚΟΡΠΙΟΣ",
    "Sagittarius": "ΤΟΞΟΤΗΣ",
    "Capricorn":   "ΑΙΓΟΚΕΡΩΣ",
    "Aquarius":    "ΥΔΡΟΧΟΟΣ",
    "Pisces":      "ΙΧΘΥΕΣ",
}

GREEK_NUMS = {
    1:"α", 2:"β", 3:"γ", 4:"δ", 5:"ε", 6:"ϛ", 7:"ζ", 8:"η", 9:"θ",
    10:"ι", 11:"ια", 12:"ιβ", 13:"ιγ", 14:"ιδ", 15:"ιε", 16:"ιϛ", 17:"ιζ",
    18:"ιη", 19:"ιθ", 20:"κ", 21:"κα", 22:"κβ", 23:"κγ", 24:"κδ", 25:"κε",
    26:"κϛ", 27:"κζ", 28:"κη", 29:"κθ", 30:"λ",
}

# ── Model ──────────────────────────────────────────────────────────────────────

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


def load_models(ckpt_dir: Path, harmonics: int = 3):
    scaler_path = ckpt_dir / "scaler.json"
    flat_path   = ckpt_dir / "flat_model.pt"
    topo_path   = ckpt_dir / "topo_model.pt"

    for p in [scaler_path, flat_path, topo_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run train_astro_models.py first.")

    with open(scaler_path) as fh:
        stats = json.load(fh)

    flat_m = AstroMLP(1, 64)
    topo_m = AstroMLP(2 * harmonics, 64)
    flat_m.load_state_dict(
        torch.load(flat_path, map_location="cpu", weights_only=True))
    topo_m.load_state_dict(
        torch.load(topo_path, map_location="cpu", weights_only=True))
    flat_m.eval()
    topo_m.eval()
    return flat_m, topo_m, stats


def flat_features(lam: np.ndarray) -> np.ndarray:
    return (lam / 360.0).reshape(-1, 1).astype(np.float32)


def topo_features(lam: np.ndarray, n_harmonics: int = 3) -> np.ndarray:
    theta = 2.0 * math.pi * lam / 360.0
    cols = []
    for k in range(1, n_harmonics + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))
    return np.stack(cols, axis=1).astype(np.float32)


def predict(model, X: np.ndarray, stats: dict) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(X)).numpy()
    mu    = np.array(stats["mu"],    dtype=np.float32)
    sigma = np.array(stats["sigma"], dtype=np.float32)
    return out * sigma + mu


# ── Formatting helpers ─────────────────────────────────────────────────────────

def float_to_sexag(v: float) -> str:
    v = max(0.0, v)
    d = int(v)
    m = round((v - d) * 60)
    if m >= 60:
        d += 1; m = 0
    return f"{d};{m:02d}"


def make_tsv_header(signum: str, model_name: str, table_type: str,
                    klima: int, scan_date: str, rmse: dict) -> str:
    lines = [
        f"# {signum} — Predicted Table: {table_type}",
        f"# Model: {model_name}",
        f"# Klima: {klima}  (Byzantium/Hellespont, φ≈41°N, longest day 15h)",
        f"# Scan date: {scan_date}",
        f"# RMSE estimates: anaph={rmse.get('anaph_deg','-'):.3f}°  "
        f"hour={rmse.get('hour_h','-'):.3f}h  chron={rmse.get('chron_min','-'):.1f}min",
        f"# STATUS: ALL VALUES ARE MODEL PREDICTIONS — verify against manuscript images",
        f"# Uncertain cells in actual transcriptions should be compared to these predictions",
        f"# Column *_raw: [unread] — Greek numeral not yet read from image",
        f"# Column *_dec: [pred_{model_name}] — predicted decimal value",
        f"#",
        "sign\tdeg\tanaph_deg_raw\tanaph_min_raw\thour_raw\tchron_raw"
        "\tdeg_dec\tanaph_dec\thour_dec\tchron_dec",
    ]
    return "\n".join(lines)


# ── TSV generation ─────────────────────────────────────────────────────────────

def generate_klima5_tsv(pred: np.ndarray, model_name: str) -> list:
    """
    Generate TSV data rows for all 360 ecliptic degrees.
    pred: shape (360, 3) — columns: anaph_dec, hour_dec, chron_dec
    """
    rows = []
    for i in range(360):
        lam   = i + 1
        s_idx = min(i // 30, 11)
        sign  = SIGN_ORDER[s_idx]
        deg   = (i % 30) + 1

        # Sign boundary marker
        if deg == 1:
            rows.append(f"# === {sign.upper()} ({GREEK_SIGNS[sign]}) ===")

        anaph_pred = pred[i, 0]
        hour_pred  = pred[i, 1]
        chron_pred = pred[i, 2]

        anaph_sexag = float_to_sexag(anaph_pred)
        hour_int    = max(0, round(hour_pred))
        chron_int   = max(0, round(chron_pred)) % 60

        row = (
            f"{sign}\t{GREEK_NUMS[deg]}"
            f"\t[unread]\t[unread]\t[unread]\t[unread]"
            f"\t{deg}"
            f"\t{anaph_sexag}[pred_{model_name}]"
            f"\t{hour_int}[pred_{model_name}]"
            f"\t{chron_int}[pred_{model_name}]"
        )
        rows.append(row)
    return rows


def generate_ensemble_tsv(pred_flat: np.ndarray, pred_topo: np.ndarray) -> list:
    """
    Ensemble: flat for anaph (lower RMSE), topo for hour+chron (lower RMSE).
    """
    rows = []
    for i in range(360):
        lam   = i + 1
        s_idx = min(i // 30, 11)
        sign  = SIGN_ORDER[s_idx]
        deg   = (i % 30) + 1

        if deg == 1:
            rows.append(f"# === {sign.upper()} ({GREEK_SIGNS[sign]}) ===")

        anaph_pred = pred_flat[i, 0]   # flat wins for anaph
        hour_pred  = pred_topo[i, 1]   # topo wins for hour
        chron_pred = pred_topo[i, 2]   # topo wins for chron

        row = (
            f"{sign}\t{GREEK_NUMS[deg]}"
            f"\t[unread]\t[unread]\t[unread]\t[unread]"
            f"\t{deg}"
            f"\t{float_to_sexag(anaph_pred)}[pred_flat]"
            f"\t{max(0,round(hour_pred))}[pred_topo]"
            f"\t{max(0,round(chron_pred)) % 60}[pred_topo]"
        )
        rows.append(row)
    return rows


def write_tsv(path: Path, header: str, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


# ── Stub for incompatible manuscripts ─────────────────────────────────────────

def generate_stub_tsv(signum: str, ms_cfg: dict, out_dir: Path, scan_date: str) -> None:
    conf   = ms_cfg.get("model_confidence", "none")
    reason = ms_cfg.get("confidence_note", "")
    types  = ms_cfg.get("table_types", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# {signum} — Model Scan Stub
# Date: {scan_date}
# Model confidence: {conf.upper()}
#
# REASON MODELS DO NOT APPLY:
# {reason}
#
# Table types in this manuscript:
#   {chr(10).join('# - ' + t for t in types)}
#
# To transcribe this manuscript's tables, you will need to either:
#   a) Collect training data from this manuscript type and retrain the models
#   b) Use a general-purpose HTR tool (Kraken, Transkribus) with Greek support
#   c) Manual transcription with predict_table.py for intra-table uncertainty filling
#
# Column structure of the expected table (if table_type is known):
# sign  deg  col1_raw  col2_raw  col3_raw  deg_dec  col1_dec  col2_dec  col3_dec
# [NO PREDICTIONS GENERATED]
"""
    (out_dir / "README_no_predictions.txt").write_text(content, encoding="utf-8")


# ── Confidence scoring ────────────────────────────────────────────────────────

def compute_confidence_scores(ms_cfg: dict, pred_flat: np.ndarray,
                               pred_topo: np.ndarray) -> dict:
    """
    Estimate per-column confidence for this manuscript based on:
    - model type applicability
    - empirical training RMSEs
    - manuscript-specific adjustments
    """
    conf_level = ms_cfg.get("model_confidence", "none")

    # Empirical RMSEs from training (on held-out uncertain cells)
    flat_rmse = {"anaph_deg": 0.80, "hour_h": 0.59, "chron_min": 20.9}
    topo_rmse = {"anaph_deg": 3.39, "hour_h": 0.35, "chron_min": 11.8}
    ensemble  = {"anaph_deg": 0.80, "hour_h": 0.35, "chron_min": 11.8}  # best of both

    # Adjustment factors per confidence level
    adj = {
        "high":       1.0,
        "medium":     2.0,   # ~2× expected error for adaptations
        "medium-low": 3.5,
        "low":        8.0,   # wrong values but might catch structure
        "none":       None,  # completely inapplicable
    }
    factor = adj.get(conf_level)

    if factor is None:
        return {
            "applicable": False,
            "reason": ms_cfg.get("confidence_note", ""),
        }

    def apply(rmse_dict):
        return {k: (v * factor if v is not None else None)
                for k, v in rmse_dict.items()}

    # Value ranges in predictions
    anaph_range = float(pred_flat[:, 0].max() - pred_flat[:, 0].min())
    hour_range  = float(pred_topo[:, 1].max() - pred_topo[:, 1].min())
    chron_range = float(pred_topo[:, 2].max() - pred_topo[:, 2].min())

    flat_adj  = apply(flat_rmse)
    topo_adj  = apply(topo_rmse)
    ens_adj   = apply(ensemble)

    return {
        "applicable":       True,
        "confidence_level": conf_level,
        "adjustment_factor": factor,
        "flat_model": {
            "expected_rmse":  flat_adj,
            "transcribable_pct": {
                "anaph": min(100, round(100 * (1 - flat_adj["anaph_deg"] / max(anaph_range, 1)), 1)),
                "hour":  min(100, round(100 * (1 - flat_adj["hour_h"] / max(hour_range, 0.5)), 1)),
                "chron": min(100, round(100 * (1 - flat_adj["chron_min"] / max(chron_range, 10)), 1)),
            },
        },
        "topo_model": {
            "expected_rmse":  topo_adj,
            "transcribable_pct": {
                "anaph": min(100, round(100 * (1 - topo_adj["anaph_deg"] / max(anaph_range, 1)), 1)),
                "hour":  min(100, round(100 * (1 - topo_adj["hour_h"] / max(hour_range, 0.5)), 1)),
                "chron": min(100, round(100 * (1 - topo_adj["chron_min"] / max(chron_range, 10)), 1)),
            },
        },
        "ensemble": {
            "expected_rmse":  ens_adj,
            "transcribable_pct": {
                "anaph": min(100, round(100 * (1 - ens_adj["anaph_deg"] / max(anaph_range, 1)), 1)),
                "hour":  min(100, round(100 * (1 - ens_adj["hour_h"] / max(hour_range, 0.5)), 1)),
                "chron": min(100, round(100 * (1 - ens_adj["chron_min"] / max(chron_range, 10)), 1)),
            },
        },
        "rows_predicted": 360,
        "note": ms_cfg.get("confidence_note", ""),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--priority", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--signum",   type=str, default=None)
    ap.add_argument("--harmonics", type=int, default=3,
                    help="Fourier harmonics for topo model (must match training, default 3)")
    args = ap.parse_args()

    scan_date = datetime.utcnow().strftime("%Y-%m-%d")

    # ── Load config ──────────────────────────────────────────────────────────
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr); sys.exit(1)
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)

    manuscripts = cfg["manuscripts"]
    if args.signum:
        manuscripts = [m for m in manuscripts if m["signum"] == args.signum]
    else:
        manuscripts = [m for m in manuscripts if m["priority"] in args.priority]

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models …")
    try:
        flat_m, topo_m, stats = load_models(CKPT_DIR, args.harmonics)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)
    print(f"  flat + topo models loaded from {CKPT_DIR}/")

    # Pre-compute predictions for all 360° (shared across all manuscripts)
    lam_full = np.arange(1, 361, dtype=np.float32)
    X_f_full = flat_features(lam_full)
    X_t_full = topo_features(lam_full, args.harmonics)
    pred_full_flat = predict(flat_m, X_f_full, stats)
    pred_full_topo = predict(topo_m, X_t_full, stats)
    print(f"  Full 360° predictions computed (flat + topo)")

    # RMSE reference values from training
    train_cfg = cfg.get("models_trained_on", {})
    flat_rmse_ref = train_cfg.get("training_rmse", {}).get("flat_model", {})
    topo_rmse_ref = train_cfg.get("training_rmse", {}).get("topo_model", {})

    scan_index = {}
    print()

    for ms in manuscripts:
        signum   = ms["signum"]
        ms_dir   = MS_BASE / ms["dir"]
        pred_dir = ms_dir / "predicted_tables"
        print(f"── {signum} (confidence: {ms.get('model_confidence','?')}) ──")

        conf  = ms.get("model_confidence", "none")
        apply = ms.get("model_direct_apply", False)

        # ── Incompatible or unknown ──────────────────────────────────────────
        if conf in ("low", "none") and apply is False:
            generate_stub_tsv(signum, ms, pred_dir, scan_date)
            result = {
                "signum":        signum,
                "applicable":    False,
                "model_confidence": conf,
                "table_types":   ms.get("table_types", []),
                "reason":        ms.get("confidence_note", ""),
                "output_files":  [str(pred_dir / "README_no_predictions.txt")],
            }
            scan_index[signum] = result
            print(f"  → STUB: {conf.upper()} — {ms.get('confidence_note','')[:80]}")
            print()
            continue

        # ── Generate predictions ─────────────────────────────────────────────
        pred_dir.mkdir(parents=True, exist_ok=True)

        # Flat TSV
        flat_header = make_tsv_header(
            signum, "flat", "Handy Tables Oblique Ascensions Klima V", 5, scan_date,
            {"anaph_deg": flat_rmse_ref.get("anaph_deg", 0.80),
             "hour_h":    flat_rmse_ref.get("hour_h",    0.59),
             "chron_min": flat_rmse_ref.get("chron_min", 20.9)})
        flat_rows   = generate_klima5_tsv(pred_full_flat, "flat")
        flat_path   = pred_dir / "klima5_full_flat.tsv"
        write_tsv(flat_path, flat_header, flat_rows)
        print(f"  ✓ flat  TSV → {flat_path.name}  ({len([r for r in flat_rows if not r.startswith('#')])} data rows)")

        # Topo TSV
        topo_header = make_tsv_header(
            signum, "topo", "Handy Tables Oblique Ascensions Klima V", 5, scan_date,
            {"anaph_deg": topo_rmse_ref.get("anaph_deg", 3.39),
             "hour_h":    topo_rmse_ref.get("hour_h",    0.35),
             "chron_min": topo_rmse_ref.get("chron_min", 11.8)})
        topo_rows   = generate_klima5_tsv(pred_full_topo, "topo")
        topo_path   = pred_dir / "klima5_full_topo.tsv"
        write_tsv(topo_path, topo_header, topo_rows)
        print(f"  ✓ topo  TSV → {topo_path.name}")

        # Ensemble TSV
        ens_header = make_tsv_header(
            signum, "ensemble", "Handy Tables Oblique Ascensions Klima V", 5, scan_date,
            {"anaph_deg": flat_rmse_ref.get("anaph_deg", 0.80),
             "hour_h":    topo_rmse_ref.get("hour_h",    0.35),
             "chron_min": topo_rmse_ref.get("chron_min", 11.8)})
        ens_rows  = generate_ensemble_tsv(pred_full_flat, pred_full_topo)
        ens_path  = pred_dir / "klima5_ensemble.tsv"
        write_tsv(ens_path, ens_header, ens_rows)
        print(f"  ✓ ens.  TSV → {ens_path.name}  (flat→anaph, topo→hour+chron)")

        # Confidence scores
        conf_scores = compute_confidence_scores(
            ms, pred_full_flat, pred_full_topo)

        # scan_result.json
        scan_result = {
            "signum":            signum,
            "scan_date":         scan_date,
            "manuscript_dir":    ms["dir"],
            "applicable":        True,
            "model_confidence":  conf,
            "table_types":       ms.get("table_types", []),
            "klimata_present":   ms.get("klimata_present", []),
            "klima5_folio_range": ms.get("klima5_folio_range"),
            "confidence_scores": conf_scores,
            "output_files": {
                "flat":     str(flat_path.relative_to(ASTRO_DIR)),
                "topo":     str(topo_path.relative_to(ASTRO_DIR)),
                "ensemble": str(ens_path.relative_to(ASTRO_DIR)),
            },
            "usage_note": (
                "Compare [pred_*] values against actual image readings. "
                "When transcribing, mark uncertain cells [?] and run "
                "predict_table.py to fill them with model predictions."
            ),
        }
        (pred_dir / "scan_result.json").write_text(
            json.dumps(scan_result, indent=2, ensure_ascii=False))

        scan_index[signum] = scan_result

        ens_t = conf_scores.get("ensemble", {})
        if isinstance(ens_t, dict) and ens_t.get("applicable", True):
            pct = ens_t.get("transcribable_pct", {})
            print(f"  ↳ ensemble est. transcribable: "
                  f"anaph={pct.get('anaph','?')}%  "
                  f"hour={pct.get('hour','?')}%  "
                  f"chron={pct.get('chron','?')}%")
        print()

    # ── Master index ─────────────────────────────────────────────────────────
    idx_path = MS_BASE / "scan_index.json"
    idx_path.write_text(json.dumps(scan_index, indent=2, ensure_ascii=False))
    print(f"Scan index → {idx_path}")
    print("\nNext: python coverage_report.py")


if __name__ == "__main__":
    main()
