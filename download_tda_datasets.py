"""
download_tda_datasets.py
Clone top-scored TDA repos locally and to LaCie.
Reads results/tda_datasets.json produced by scan_topological_datasets.py.

Usage:
    python3 download_tda_datasets.py [--local-dir PATH] [--lacie-dir PATH]
                                     [--min-score 2.0] [--max-size-mb 200]
                                     [--dry-run]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

RESULTS_FILE  = "results/tda_datasets.json"
DEFAULT_LOCAL = "datasets/tda_repos"
DEFAULT_LACIE = "/Volumes/LaCie/tda_datasets"

# Repos we definitely want regardless of simple score heuristic (research value)
PRIORITY_REPOS = {
    "scikit-tda/tadasets",          # canonical synthetic TDA shapes
    "aidos-lab/mantra",             # manifold triangulations (ICLR 2025)
    "aidos-lab/mantra-benchmarks",  # benchmark suite for mantra
    "RabadanLab/ph_datasets",       # columbia PH datasets
    "netsci-rwth/ahorn",            # simplicial complex / cell complex / hypergraph
    "A-EL-YAAGOUBI/Dynamic-TDA",    # dynamic time-series TDA → directly analogous to side-channel trace analysis
    "aerojam95/mph-deep-learning-pipeline",  # multi-parameter PH → DL pipeline
    "eashwarsoma/TDA-benchmark",    # PH benchmark (R-journal)
    "degnbol/hyperTDA",             # PL curves → point clouds → PH → hypergraphs
    "LucaNyckees/zigzag-homology",  # zigzag persistence
}

# Repos to SKIP (noise / clearly irrelevant / too large with no research value)
SKIP_REPOS = {
    "Luckyaman/Sensor-Spoofing-and-Perception-Manipulation-in-Autonomous-Vehicles",  # empty repo
    "Luckyaman/Sensor-Spoofing-and-Perception-Manipulation-in-Autonomous-Vehicles",
    "Rafiyudheen04/POTHOLE-DETECTION",   # unrelated
    "Rashmi-S-Gowda/Spectral-Angular-Classification-of-Satellite-Image",  # unrelated
    "Eshan-Agarwal/MNIST-Dataset",       # just MNIST, no TDA content
}

# Crypto-relevance tags for ZeroTrace radar
CRYPTO_RELEVANT = {
    "A-EL-YAAGOUBI/Dynamic-TDA": (
        "HIGH",
        "Dynamic sliding-window PH on EEG time-series — exact same pipeline as ZeroTrace "
        "Study 7 (sliding-window RL on power traces). Adapt directly for NTT/RSA trace segments."
    ),
    "aerojam95/mph-deep-learning-pipeline": (
        "HIGH",
        "Multi-parameter PH landscapes → CNN/contrastive DL. The pipeline architecture maps "
        "directly to converting NTT butterfly traces into PH feature tensors for Study 11."
    ),
    "acharchan/tdaproject": (
        "MEDIUM",
        "PH on geometric Brownian motion (stochastic process analysis). Timing jitter in "
        "crypto hardware follows similar stochastic structure — useful for noise-floor calibration."
    ),
    "LucaNyckees/zigzag-homology": (
        "MEDIUM",
        "Zigzag persistence tracks topological changes across time — useful for detecting "
        "transition events in side-channel traces (RELIN/BOOT events in FHE, Study 10)."
    ),
    "aidos-lab/mantra": (
        "LOW",
        "Manifold triangulation dataset useful for pretraining the SNN simplicial backbone. "
        "Provides diverse simplicial complex structures beyond synthetic CMF data."
    ),
    "netsci-rwth/ahorn": (
        "LOW",
        "Cell complex and hypergraph datasets — diverse higher-order graph structures for "
        "backbone pretraining. Improves generalisation of the SNN100M k5 backbone."
    ),
    "scikit-tda/tadasets": (
        "LOW",
        "Canonical synthetic shapes (circles, tori, spheres, Klein bottles). Useful as "
        "negative-class examples in Study 8 ROCA detection (structured vs random key topology)."
    ),
    "degnbol/hyperTDA": (
        "LOW",
        "PL-curve → PH → hypergraph pipeline. NTT butterfly graphs have a natural PL-curve "
        "representation; this codebase shows how to lift it to a hypergraph for the backbone."
    ),
}


def free_gb(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def clone_repo(url: str, dest: str, dry_run: bool) -> bool:
    if os.path.exists(dest):
        print(f"    [SKIP] already exists: {dest}")
        return True
    if dry_run:
        print(f"    [DRY-RUN] would clone {url} → {dest}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, dest],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"    [OK] {dest}")
        return True
    else:
        print(f"    [FAIL] {result.stderr.strip()[:120]}")
        return False


def select_repos(results: list[dict], min_score: float, max_size_mb: float) -> list[dict]:
    selected = []
    for r in results:
        name     = r["name"]
        score    = r.get("score", 0)
        size_mb  = r.get("size_kb", 0) / 1024.0

        if name in SKIP_REPOS:
            continue
        if name in PRIORITY_REPOS:
            selected.append(r)
            continue
        if score >= min_score and size_mb <= max_size_mb:
            selected.append(r)

    # Deduplicate by name
    seen = set()
    out  = []
    for r in selected:
        if r["name"] not in seen:
            seen.add(r["name"])
            out.append(r)
    return out


def print_crypto_report(selected: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("  ZEROTRACE CRYPTO RELEVANCE ANALYSIS")
    print("=" * 68)
    print()
    print("  Direct side-channel (RSA/Kyber) datasets: NONE in this scan.")
    print("  Recommended external sources to add:")
    print("    • ASCAD  — https://github.com/ANSSI-FR/ASCAD  (AES power traces, 700k)")
    print("    • DPA Contest v4 — http://www.dpacontest.org  (AES, EM traces)")
    print("    • ChipWhisperer CHES CTF — https://github.com/newaetech/chipwhisperer")
    print("    • SCAAML — https://github.com/google/scaaml  (Google, Kyber/AES)")
    print()
    print("  From THIS scan — indirect crypto utility:\n")

    levels = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for r in selected:
        entry = CRYPTO_RELEVANT.get(r["name"])
        if entry:
            level, reason = entry
            levels[level].append((r["name"], reason))

    for level in ("HIGH", "MEDIUM", "LOW"):
        if not levels[level]:
            continue
        print(f"  [{level}]")
        for name, reason in levels[level]:
            print(f"    • {name}")
            # word-wrap reason at 60 chars
            words = reason.split()
            line = "      "
            for w in words:
                if len(line) + len(w) > 68:
                    print(line)
                    line = "      " + w + " "
                else:
                    line += w + " "
            if line.strip():
                print(line)
            print()

    print("  Recommendation for ZeroTrace:")
    print("    1. Add ASCAD + SCAAML as primary crypto side-channel data.")
    print("    2. Use Dynamic-TDA (EEG pipeline) as architecture template for")
    print("       sliding-window PH feature extraction on NTT traces (Study 11).")
    print("    3. Use AHORN + MANTRA for backbone (SNN100M k5) pretraining to")
    print("       improve generalisation on unseen simplicial structures.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",    default=RESULTS_FILE)
    parser.add_argument("--local-dir",  default=DEFAULT_LOCAL)
    parser.add_argument("--lacie-dir",  default=DEFAULT_LACIE)
    parser.add_argument("--min-score",  type=float, default=2.0)
    parser.add_argument("--max-size-mb", type=float, default=200.0)
    parser.add_argument("--dry-run",    action="store_true")
    args = parser.parse_args()

    # Load scan results
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)
    all_results = data["results"]

    selected = select_repos(all_results, args.min_score, args.max_size_mb)
    print(f"Selected {len(selected)} repos (score≥{args.min_score}, size≤{args.max_size_mb}MB, priority overrides included)\n")

    # Check LaCie
    lacie_ok = os.path.exists(args.lacie_dir) or os.path.exists(os.path.dirname(args.lacie_dir))
    if not os.path.exists(args.lacie_dir) and lacie_ok:
        if not args.dry_run:
            os.makedirs(args.lacie_dir, exist_ok=True)

    if os.path.exists("/Volumes/LaCie"):
        gb = free_gb("/Volumes/LaCie")
        print(f"LaCie free: {gb:.1f} GB\n")

    # Clone
    ok_local  = 0
    ok_lacie  = 0
    fail      = 0

    for r in selected:
        name    = r["name"]
        url     = r["url"]
        slug    = name.replace("/", "__")
        score   = r.get("score", 0)
        size_mb = r.get("size_kb", 0) / 1024.0
        crypto  = CRYPTO_RELEVANT.get(name, ("—", ""))[0]

        print(f"  {name}  [score={score} size={size_mb:.0f}MB crypto={crypto}]")

        local_dest = os.path.join(args.local_dir, slug)
        if clone_repo(url, local_dest, args.dry_run):
            ok_local += 1
        else:
            fail += 1

        if os.path.exists("/Volumes/LaCie"):
            lacie_dest = os.path.join(args.lacie_dir, slug)
            if clone_repo(url, lacie_dest, args.dry_run):
                ok_lacie += 1

    print(f"\n[DONE] local={ok_local} lacie={ok_lacie} failed={fail}")

    print_crypto_report(selected)

    # Write manifest
    manifest_path = "results/tda_download_manifest.json"
    manifest = {
        "downloaded_at": __import__("datetime").datetime.now().isoformat(),
        "total_selected": len(selected),
        "local_dir": args.local_dir,
        "lacie_dir": args.lacie_dir,
        "repos": [
            {
                "name":        r["name"],
                "url":         r["url"],
                "score":       r.get("score"),
                "size_kb":     r.get("size_kb"),
                "crypto_tag":  CRYPTO_RELEVANT.get(r["name"], ("—", ""))[0],
                "crypto_note": CRYPTO_RELEVANT.get(r["name"], ("—", ""))[1],
            }
            for r in selected
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[OK] Manifest saved → {manifest_path}")


if __name__ == "__main__":
    main()
