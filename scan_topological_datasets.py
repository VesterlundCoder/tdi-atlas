"""
scan_topological_datasets.py
Scans GitHub and HuggingFace for topological / TDA datasets.

Usage:
    python scan_topological_datasets.py [--token GITHUB_TOKEN] [--out results/tda_datasets.json]

Token is optional but strongly recommended (GitHub: 60 req/h unauthenticated vs 5000 req/h).
Set via:
    export GITHUB_TOKEN=ghp_...
    python scan_topological_datasets.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_QUERIES = [
    '"topological data analysis" dataset',
    '"persistent homology" benchmark',
    '"persistent homology" dataset',
    '"simplicial complex" dataset',
    '"betti numbers" dataset',
    '"point cloud" "persistent homology"',
    '"ripser" dataset',
    '"gudhi" dataset',
    '"vietoris-rips" data',
    '"mapper algorithm" dataset',
    'topic:topological-data-analysis topic:dataset',
    'topic:persistent-homology topic:dataset',
    'topic:tda dataset',
]

HF_SEARCH_TERMS = [
    "topological data analysis",
    "persistent homology",
    "simplicial complex",
    "point cloud topology",
    "betti numbers",
    "TDA benchmark",
]

# Formats that signal genuine topological data (boost score)
HIGH_VALUE_EXTENSIONS = {".npy", ".npz", ".h5", ".hdf5", ".vr", ".dgm", ".gudhi", ".csv", ".off"}

# Topics / keywords that signal real dataset repos (not just theory)
DATASET_SIGNAL_KEYWORDS = [
    "dataset", "benchmark", "data", "point-cloud", "pointcloud",
    "shapes", "meshes", "manifold", "simplicial", "filtration",
    "barcode", "diagram", "homology",
]

GITHUB_API = "https://api.github.com/search/repositories"
HF_API     = "https://huggingface.co/api/datasets"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def github_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def score_github_item(item: dict) -> float:
    """Heuristic quality score for a GitHub repository."""
    s = 0.0
    s += min(item.get("stargazers_count", 0) / 50.0, 5.0)   # up to +5 for stars
    s += min(item.get("forks_count", 0) / 20.0, 2.0)         # up to +2 for forks
    topics = [t.lower() for t in item.get("topics", [])]
    desc   = (item.get("description") or "").lower()
    for kw in DATASET_SIGNAL_KEYWORDS:
        if kw in topics or kw in desc:
            s += 0.5
    if item.get("has_downloads"):
        s += 0.5
    # Penalise archived repos
    if item.get("archived"):
        s -= 2.0
    return round(s, 2)


def search_github(token: str | None, verbose: bool = True) -> list[dict]:
    headers   = github_headers(token)
    seen_ids  = set()
    results   = []

    for query in GITHUB_QUERIES:
        encoded = quote_plus(query)
        url     = f"{GITHUB_API}?q={encoded}&sort=stars&order=desc&per_page=30"
        if verbose:
            print(f"  GitHub › {query!r}")

        try:
            resp = requests.get(url, headers=headers, timeout=15)
        except requests.RequestException as exc:
            print(f"    [WARN] request failed: {exc}")
            time.sleep(2)
            continue

        if resp.status_code == 403:
            print("    [WARN] GitHub rate-limit hit — sleeping 60s")
            time.sleep(60)
            continue
        if resp.status_code != 200:
            print(f"    [WARN] HTTP {resp.status_code}")
            time.sleep(2)
            continue

        data = resp.json()
        for item in data.get("items", []):
            rid = item["id"]
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            results.append({
                "source":       "github",
                "name":         item["full_name"],
                "url":          item["html_url"],
                "description":  item.get("description") or "",
                "stars":        item.get("stargazers_count", 0),
                "forks":        item.get("forks_count", 0),
                "topics":       item.get("topics", []),
                "language":     item.get("language") or "",
                "size_kb":      item.get("size", 0),
                "updated_at":   item.get("updated_at", ""),
                "archived":     item.get("archived", False),
                "license":      (item.get("license") or {}).get("spdx_id", ""),
                "score":        score_github_item(item),
                "matched_query": query,
            })

        # Respect secondary rate-limit (search API: 30 req/min authenticated)
        time.sleep(2.5 if token else 4.0)

    return results


def search_huggingface(verbose: bool = True) -> list[dict]:
    results = []
    seen    = set()

    for term in HF_SEARCH_TERMS:
        if verbose:
            print(f"  HuggingFace › {term!r}")
        url    = f"{HF_API}?search={quote_plus(term)}&limit=20"

        try:
            resp = requests.get(url, timeout=15)
        except requests.RequestException as exc:
            print(f"    [WARN] request failed: {exc}")
            continue

        if resp.status_code != 200:
            print(f"    [WARN] HTTP {resp.status_code}")
            time.sleep(1)
            continue

        for ds in resp.json():
            did = ds.get("id") or ds.get("_id", "")
            if did in seen:
                continue
            seen.add(did)

            tags = ds.get("tags") or []
            results.append({
                "source":      "huggingface",
                "name":        did,
                "url":         f"https://huggingface.co/datasets/{did}",
                "description": (ds.get("description") or ds.get("cardData", {}).get("description") or ""),
                "tags":        tags,
                "downloads":   ds.get("downloads", 0),
                "likes":       ds.get("likes", 0),
                "updated_at":  ds.get("lastModified", ""),
                "score":       round(
                    min(ds.get("downloads", 0) / 200.0, 5.0) +
                    min(ds.get("likes", 0) / 10.0, 3.0),
                    2
                ),
                "matched_query": term,
            })

        time.sleep(1.5)

    return results


def deduplicate(items: list[dict]) -> list[dict]:
    """Remove near-duplicate repos (same name, different matched_query)."""
    seen  = {}
    out   = []
    for item in items:
        key = item["name"].lower()
        if key not in seen:
            seen[key] = True
            out.append(item)
    return out


def print_summary(items: list[dict]) -> None:
    github_items = [i for i in items if i["source"] == "github"]
    hf_items     = [i for i in items if i["source"] == "huggingface"]

    print("\n" + "=" * 60)
    print(f"  TOPOLOGICAL DATASET SCAN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    print(f"  GitHub repos  : {len(github_items)}")
    print(f"  HuggingFace   : {len(hf_items)}")
    print(f"  Total unique  : {len(items)}")
    print()

    print("  TOP 15 by score:")
    top = sorted(items, key=lambda x: x["score"], reverse=True)[:15]
    for i, r in enumerate(top, 1):
        src  = "GH" if r["source"] == "github" else "HF"
        name = r["name"][:48]
        desc = (r.get("description") or "")[:60]
        print(f"  {i:>2}. [{src}] {name:<48}  ★{r['score']:.1f}  {desc}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scan GitHub + HuggingFace for TDA datasets")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub Personal Access Token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--out", default="results/tda_datasets.json",
                        help="Output JSON file (default: results/tda_datasets.json)")
    parser.add_argument("--no-hf", action="store_true",
                        help="Skip HuggingFace search")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    token   = args.token or None

    if not token:
        print("[WARN] No GitHub token — rate-limited to 60 req/h. Set GITHUB_TOKEN for best results.\n")
    else:
        print(f"[INFO] Using GitHub token ({token[:8]}...)\n")

    print("=== Scanning GitHub ===")
    gh_results = search_github(token, verbose)
    print(f"  → {len(gh_results)} raw hits\n")

    hf_results = []
    if not args.no_hf:
        print("=== Scanning HuggingFace ===")
        hf_results = search_huggingface(verbose)
        print(f"  → {len(hf_results)} raw hits\n")

    all_results = deduplicate(gh_results + hf_results)
    all_results.sort(key=lambda x: x["score"], reverse=True)

    print_summary(all_results)

    # Save
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "total": len(all_results),
                "github_count": sum(1 for r in all_results if r["source"] == "github"),
                "huggingface_count": sum(1 for r in all_results if r["source"] == "huggingface"),
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OK] Saved {len(all_results)} entries → {out_path}")


if __name__ == "__main__":
    main()
