#!/usr/bin/env python3
"""
Model Coverage Report — estimate transcription capacity per manuscript
======================================================================
Reads the scan_index.json produced by scan_tables.py and generates:
  1. A printed summary with per-manuscript estimates
  2. A markdown report: manuscripts/COVERAGE_REPORT.md
  3. A machine-readable summary: manuscripts/coverage_summary.json

Coverage categories per cell:
  TRANSCRIBABLE   — model RMSE small enough to be useful (< 1° for anaph, < 0.5h for hour)
  UNCERTAIN       — model gives a prediction but with high expected error
  OFF_TABLE       — table type incompatible; model cannot help at all

Usage:
  python coverage_report.py              # reads scan_index.json
  python coverage_report.py --verbose    # show per-sign breakdown
"""

import argparse
import json
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

ASTRO_DIR  = Path(__file__).parent
MS_BASE    = ASTRO_DIR / "manuscripts"
SCAN_INDEX = MS_BASE / "scan_index.json"
CONFIG     = MS_BASE / "config.json"

# ── Thresholds for "transcribable" vs "uncertain" ─────────────────────────────
# Based on the precision required to distinguish individual sexagesimal entries.
# Handy Tables entries differ by ~0.5°–1° per degree step, so:
#   anaph RMSE < 0.5° → TRANSCRIBABLE   (can reliably identify the correct cell)
#   anaph RMSE < 2.0° → UNCERTAIN       (within ~2 cells, needs manual check)
#   anaph RMSE ≥ 2.0° → OFF_TABLE for this column

THRESHOLDS = {
    # Handy Tables step size: anaph ~0.8–1.0°/deg, hour ~0.2h/deg, chron ~0.9 min/deg
    # TRANSCRIBABLE = RMSE < ~1 table step  → prediction identifies correct cell
    # UNCERTAIN     = RMSE < ~4 table steps → useful range hint, needs image check
    # OFF TABLE     = RMSE ≥ 4 table steps  → not useful as a starting point
    "anaph_deg":  {"transcribable": 1.2,  "uncertain": 4.0},
    "hour_h":     {"transcribable": 0.5,  "uncertain": 1.2},
    "chron_min":  {"transcribable": 15.0, "uncertain": 30.0},
}

TOTAL_ROWS     = 360    # 12 signs × 30 degrees = full Handy Tables Klima V
TOTAL_COLS     = 3      # anaph, hour, chron
TOTAL_CELLS    = TOTAL_ROWS * TOTAL_COLS   # 1080 per klima per manuscript

# ── Classification helpers ────────────────────────────────────────────────────

def classify_rmse(rmse_val: float, col: str):
    t = THRESHOLDS.get(col, {"transcribable": 1.0, "uncertain": 5.0})
    if rmse_val <= t["transcribable"]:
        return "TRANSCRIBABLE"
    elif rmse_val <= t["uncertain"]:
        return "UNCERTAIN"
    else:
        return "OFF_TABLE"


def summarise_model(rmse_dict: dict, rows: int = TOTAL_ROWS) -> dict:
    """Given RMSE dict, return cell counts per category."""
    cols = {
        "anaph_deg":  ("anaph",  "anaph (°)"),
        "hour_h":     ("hour",   "hour (h)"),
        "chron_min":  ("chron",  "chron (min)"),
    }
    result = {}
    total_t = total_u = total_o = 0
    for col_key, (short, label) in cols.items():
        rmse = rmse_dict.get(col_key)
        if rmse is None:
            cat = "OFF_TABLE"
        else:
            cat = classify_rmse(rmse, col_key)
        result[short] = {"category": cat, "rmse": rmse, "label": label}
        n_t = rows if cat == "TRANSCRIBABLE" else 0
        n_u = rows if cat == "UNCERTAIN" else 0
        n_o = rows if cat == "OFF_TABLE" else 0
        total_t += n_t; total_u += n_u; total_o += n_o

    result["_total"] = {
        "n_cells": rows * TOTAL_COLS,
        "transcribable": total_t,
        "uncertain":     total_u,
        "off_table":     total_o,
        "pct_transcribable": round(100 * total_t / (rows * TOTAL_COLS), 1),
        "pct_uncertain":     round(100 * total_u / (rows * TOTAL_COLS), 1),
        "pct_off_table":     round(100 * total_o / (rows * TOTAL_COLS), 1),
    }
    return result


# ── Report generation ─────────────────────────────────────────────────────────

BAR_WIDTH = 30

def bar(pct_t: float, pct_u: float, pct_o: float) -> str:
    t = round(BAR_WIDTH * pct_t / 100)
    u = round(BAR_WIDTH * pct_u / 100)
    o = BAR_WIDTH - t - u
    return "█" * t + "▒" * u + "░" * o


CONF_EMOJI = {
    "high":       "🟢",
    "medium":     "🟡",
    "medium-low": "🟠",
    "low":        "🔴",
    "none":       "⚫",
}


def build_report(scan_index: dict, cfg: dict, verbose: bool) -> str:
    lines = []

    def h(text, char="="):
        return text + "\n" + char * len(text)

    def add(*args):
        lines.append(" ".join(str(a) for a in args))

    add(h("Byzantine Astronomical Tables — Model Coverage Report", "="))
    add(f"Generated: see scan_index.json\n")
    add("Models trained on: Vat.gr.1291, Klima V (Byzantium 41°N), Handy Tables Oblique Ascensions")
    add("Training RMSE (uncertain cells): flat: anaph=0.80°, hour=0.59h, chron=20.9min")
    add("                                  topo: anaph=3.39°, hour=0.35h, chron=11.8min")
    add("Ensemble (best-of-both):          flat→anaph 0.80°, topo→hour 0.35h, topo→chron 11.8min\n")
    add("Legend: █ TRANSCRIBABLE  ▒ UNCERTAIN  ░ OFF TABLE / INCOMPATIBLE\n")

    add(h("Coverage Categories", "-"))
    add("TRANSCRIBABLE  = model RMSE small enough to identify correct value with high confidence")
    add("                 (anaph < 0.5°, hour < 0.3h, chron < 5min)")
    add("UNCERTAIN      = model prediction is in the right ballpark; manual check recommended")
    add("                 (anaph 0.5–2°, hour 0.3–1h, chron 5–15min)")
    add("OFF TABLE      = model predictions are unreliable for this column/manuscript\n")

    # Sort: applicable first (by conf), then non-applicable
    applicable     = [(s, d) for s, d in scan_index.items() if d.get("applicable")]
    not_applicable = [(s, d) for s, d in scan_index.items() if not d.get("applicable")]

    conf_order = {"high": 0, "medium": 1, "medium-low": 2, "low": 3, "none": 4}
    applicable.sort(key=lambda x: conf_order.get(x[1].get("model_confidence", "none"), 5))

    summary_rows = []

    # ── Applicable manuscripts ────────────────────────────────────────────────
    add(h("Manuscripts where models apply (directly or partially)", "-"))
    add()

    for signum, data in applicable:
        conf = data.get("model_confidence", "?")
        emoji = CONF_EMOJI.get(conf, "?")
        add(f"{emoji}  {signum}  [{conf.upper()} confidence]")
        add(f"   Table types: {', '.join(data.get('table_types', []))}")

        cs = data.get("confidence_scores", {})
        if not isinstance(cs, dict) or not cs.get("applicable", False):
            add(f"   No confidence scores available\n")
            continue

        for model_key, model_label in [
            ("flat_model",  "FLAT  "),
            ("topo_model",  "TOPO  "),
            ("ensemble",    "ENSEMBL"),
        ]:
            md = cs.get(model_key, {})
            if not isinstance(md, dict):
                continue
            rmse = md.get("expected_rmse", {})
            pct  = md.get("transcribable_pct", {})

            if model_key == "ensemble":
                ens_rmse = {
                    "anaph_deg":  cs.get("flat_model", {}).get("expected_rmse", {}).get("anaph_deg"),
                    "hour_h":     cs.get("topo_model", {}).get("expected_rmse", {}).get("hour_h"),
                    "chron_min":  cs.get("topo_model", {}).get("expected_rmse", {}).get("chron_min"),
                }
                sm = summarise_model(ens_rmse)
            else:
                sm = summarise_model(rmse)

            tot = sm["_total"]
            b   = bar(tot["pct_transcribable"], tot["pct_uncertain"], tot["pct_off_table"])
            add(f"   {model_label}: {b}  "
                f"{tot['pct_transcribable']:5.1f}% T  "
                f"{tot['pct_uncertain']:5.1f}% U  "
                f"{tot['pct_off_table']:5.1f}% O")
            if verbose:
                for col in ["anaph", "hour", "chron"]:
                    col_data = sm.get(col, {})
                    r = col_data.get("rmse", "?")
                    c = col_data.get("category", "?")
                    r_str = f"{r:.3f}" if isinstance(r, float) else str(r)
                    add(f"            {col:<6}: RMSE={r_str:>8}  → {c}")

        summary_rows.append({
            "signum": signum,
            "confidence": conf,
            "applicable": True,
            "ensemble_pct_T": cs.get("ensemble", {}).get("transcribable_pct", {}),
        })
        add()

    # ── Incompatible manuscripts ───────────────────────────────────────────────
    add(h("Manuscripts where models do NOT apply", "-"))
    add()

    for signum, data in not_applicable:
        conf = data.get("model_confidence", "none")
        emoji = CONF_EMOJI.get(conf, "⚫")
        add(f"{emoji}  {signum}  [{conf.upper()}]")
        add(f"   Table types: {', '.join(data.get('table_types', []))}")
        add(f"   Reason: {data.get('reason', data.get('confidence_note', 'incompatible table type'))}")
        add(f"   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0% T  0% U  100% O")
        add(f"   → Requires new training data or general HTR (Kraken/Transkribus)")
        summary_rows.append({
            "signum": signum,
            "confidence": conf,
            "applicable": False,
            "ensemble_pct_T": {},
        })
        add()

    # ── Overall summary ───────────────────────────────────────────────────────
    add(h("Overall Summary", "="))
    n_total = len(scan_index)
    n_app   = len(applicable)
    n_napp  = len(not_applicable)
    add(f"Total manuscripts in corpus:  {n_total}")
    add(f"  Models directly applicable: {n_app}  (Handy Tables / Handy Tables–derived)")
    add(f"  Models NOT applicable:      {n_napp}  (Persian/Arabic/other table types)")
    add()

    # High-confidence ensemble estimates
    high_ms = [(s, d) for s, d in applicable if d.get("model_confidence") == "high"]
    med_ms  = [(s, d) for s, d in applicable if d.get("model_confidence") in ("medium", "medium-low")]
    add(f"High-confidence manuscripts ({len(high_ms)}):")
    for signum, d in high_ms:
        add(f"  {signum}")
    add()
    add(f"Medium-confidence manuscripts ({len(med_ms)}):")
    for signum, d in med_ms:
        add(f"  {signum}  (adapted/partial Handy Tables)")
    add()

    add("Per-cell breakdown for HIGH-confidence manuscripts (ensemble model):")
    add("  Note: these are estimates based on training RMSEs, not image-verified.")
    for signum, d in high_ms:
        cs   = d.get("confidence_scores", {})
        ens  = cs.get("ensemble", {})
        pct  = ens.get("transcribable_pct", {}) if isinstance(ens, dict) else {}
        rmse = {
            "anaph_deg":  cs.get("flat_model",{}).get("expected_rmse",{}).get("anaph_deg"),
            "hour_h":     cs.get("topo_model",{}).get("expected_rmse",{}).get("hour_h"),
            "chron_min":  cs.get("topo_model",{}).get("expected_rmse",{}).get("chron_min"),
        }
        sm   = summarise_model(rmse) if any(v is not None for v in rmse.values()) else {}
        tot  = sm.get("_total", {})
        add(f"  {signum:<22} "
            f"TRANSCRIBABLE={tot.get('pct_transcribable','?')}%  "
            f"UNCERTAIN={tot.get('pct_uncertain','?')}%  "
            f"OFF_TABLE={tot.get('pct_off_table','?')}%")
    add()

    add("Interpretation:")
    add("  TRANSCRIBABLE % = % of cells where model prediction is reliable enough")
    add("                    to use as a starting transcription without manual check")
    add("  UNCERTAIN %     = % of cells where model gives a useful hint but")
    add("                    manual verification is recommended")
    add("  OFF TABLE %     = % of cells where model prediction is not useful")
    add()
    add("Important caveats:")
    add("  1. These estimates assume the target manuscript has the SAME table type")
    add("     and SAME klima (Klima V, Byzantium) as the training data.")
    add("  2. For other klimata in the same manuscript: anaph values will differ")
    add("     (need klima-specific model), but hour/chron are NOT present in HT oblique")
    add("     ascension tables for other klimata without retraining.")
    add("  3. The 'flat' model is better for anaph (monotone linear function).")
    add("     The 'topo' model is better for hour + chron (periodic on S¹).")
    add("  4. Manual verification by the user remains essential for all predictions.")

    return "\n".join(lines)


# ── Machine-readable summary ───────────────────────────────────────────────────

def build_coverage_summary(scan_index: dict) -> dict:
    summary = {}
    for signum, data in scan_index.items():
        cs   = data.get("confidence_scores", {})
        ens  = cs.get("ensemble", {}) if isinstance(cs, dict) else {}
        flat = cs.get("flat_model", {}) if isinstance(cs, dict) else {}
        topo = cs.get("topo_model", {}) if isinstance(cs, dict) else {}

        def get_rmse(model_dict):
            if not isinstance(model_dict, dict):
                return {}
            return model_dict.get("expected_rmse", {})

        ens_rmse = {
            "anaph_deg":  get_rmse(flat).get("anaph_deg"),
            "hour_h":     get_rmse(topo).get("hour_h"),
            "chron_min":  get_rmse(topo).get("chron_min"),
        } if data.get("applicable") else {}

        sm = summarise_model(ens_rmse) if ens_rmse else {}
        tot = sm.get("_total", {})

        summary[signum] = {
            "applicable":           data.get("applicable", False),
            "model_confidence":     data.get("model_confidence", "none"),
            "table_types":          data.get("table_types", []),
            "ensemble_rmse":        ens_rmse,
            "ensemble_pct_T":       tot.get("pct_transcribable", 0),
            "ensemble_pct_U":       tot.get("pct_uncertain",     0),
            "ensemble_pct_O":       tot.get("pct_off_table",     100),
            "flat_best_for":        ["anaph"],
            "topo_best_for":        ["hour", "chron"],
        }
    return summary


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                    help="Show per-column RMSE breakdown for each model")
    args = ap.parse_args()

    if not SCAN_INDEX.exists():
        print(f"ERROR: {SCAN_INDEX} not found. Run scan_tables.py first.", file=sys.stderr)
        sys.exit(1)

    with open(SCAN_INDEX) as fh:
        scan_index = json.load(fh)

    cfg = {}
    if CONFIG.exists():
        with open(CONFIG) as fh:
            cfg = json.load(fh)

    # Build and print report
    report_text = build_report(scan_index, cfg, args.verbose)
    print(report_text)

    # Write markdown report
    md_path = MS_BASE / "COVERAGE_REPORT.md"
    md_path.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved → {md_path}")

    # Write machine-readable summary
    summary = build_coverage_summary(scan_index)
    sum_path = MS_BASE / "coverage_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"Summary JSON → {sum_path}")


if __name__ == "__main__":
    main()
