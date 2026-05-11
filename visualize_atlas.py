"""
TDI Atlas Visualiser — PCA, UMAP, heatmap, and cross-domain plots.

Reads results/tdi_atlas.csv (or .json) produced by tdi_atlas.py and generates:
  1. TDI Heatmap Matrix     — datasets × filtrations, coloured by TDI value
  2. TDI PCA scatter        — datasets projected by full TDI fingerprint
  3. TDI UMAP scatter       — same, via UMAP (more nonlinear structure)
  4. Domain violin plot     — TDI distribution per domain
  5. Signal ratio bar chart — TDI_rand / TDI_trained per dataset
  6. Accuracy vs TDI scatter— does low TDI predict high accuracy?
  7. Purity gain heatmap    — how much each model improves kNN purity

Usage:
    python visualize_atlas.py                             # reads results/tdi_atlas.csv
    python visualize_atlas.py --atlas results/tdi_atlas.csv --out results/figures/atlas/
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm

warnings.filterwarnings("ignore")

DOMAIN_COLORS = {
    "biology":        "#2ecc71",
    "medicine":       "#e74c3c",
    "physics":        "#3498db",
    "chemistry":      "#9b59b6",
    "mathematics":    "#f39c12",
    "finance":        "#1abc9c",
    "social":         "#e67e22",
    "speech":         "#34495e",
    "cybersecurity":  "#c0392b",
    "ecology":        "#27ae60",
    "software":       "#7f8c8d",
    "synthetic":      "#bdc3c7",
    "vision":         "#8e44ad",
    "nlp_features":   "#2980b9",
    "remote_sensing": "#16a085",
    "energy":         "#d35400",
    "robotics":       "#2c3e50",
    "aerospace":      "#0984e3",
    "engineering":    "#6c5ce7",
    "games":          "#fd79a8",
    "education":      "#55efc4",
    "dentistry":      "#a29bfe",
    "psychology":     "#fdcb6e",
}


def load_atlas(path: str) -> List[Dict]:
    p = Path(path)
    if p.suffix == ".json":
        with open(p) as f:
            return json.load(f)
    # CSV
    import csv
    with open(p) as f:
        return list(csv.DictReader(f))


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_matrix(records: List[Dict]):
    """Build numeric arrays for plotting."""
    names = [r["dataset"] for r in records]
    domains = [r.get("domain", "unknown") for r in records]
    tdi_vr_h0 = [_float(r.get("tdi_vr_h0")) for r in records]
    tdi_vr_h1 = [_float(r.get("tdi_vr_h1")) for r in records]
    tdi_alpha_h0 = [_float(r.get("tdi_alpha_h0")) for r in records]
    tdi_alpha_h1 = [_float(r.get("tdi_alpha_h1")) for r in records]
    tdi_rand = [_float(r.get("tdi_random_label")) for r in records]
    signal_ratio = [_float(r.get("signal_ratio")) for r in records]
    accuracy = [_float(r.get("accuracy")) for r in records]
    purity_gain = [_float(r.get("purity_gain")) for r in records]
    entropy_h0 = [_float(r.get("input_entropy_h0")) for r in records]
    return {
        "names": names, "domains": domains,
        "tdi_vr_h0": tdi_vr_h0, "tdi_vr_h1": tdi_vr_h1,
        "tdi_alpha_h0": tdi_alpha_h0, "tdi_alpha_h1": tdi_alpha_h1,
        "tdi_rand": tdi_rand, "signal_ratio": signal_ratio,
        "accuracy": accuracy, "purity_gain": purity_gain,
        "entropy_h0": entropy_h0,
    }


def fig1_tdi_heatmap(data: Dict, out_dir: Path) -> Path:
    """TDI value matrix: datasets (rows) × filtration/dim (cols)."""
    keys = ["tdi_vr_h0", "tdi_vr_h1", "tdi_alpha_h0", "tdi_alpha_h1"]
    labels = ["VR H₀", "VR H₁", "Alpha H₀", "Alpha H₁"]
    N = len(data["names"])

    mat = np.zeros((N, len(keys)))
    for j, k in enumerate(keys):
        for i, v in enumerate(data[k]):
            mat[i, j] = v if v is not None else np.nan

    # Sort by VR H0
    order = np.argsort(mat[:, 0])
    mat = mat[order]
    names = [data["names"][i] for i in order]
    domains = [data["domains"][i] for i in order]

    fig_h = max(6, N * 0.22)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   norm=Normalize(vmin=np.nanmin(mat), vmax=np.nanpercentile(mat, 95)))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(N))
    ax.set_yticklabels(
        [f"{n} [{d[:4]}]" for n, d in zip(names, domains)],
        fontsize=7
    )
    plt.colorbar(im, ax=ax, label="TDI value", fraction=0.03, pad=0.02)
    ax.set_title("TDI Heatmap: Datasets × Filtration/Homology Dimension", fontsize=11)
    plt.tight_layout()
    out = out_dir / "fig1_tdi_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1: {out}")
    return out


def fig2_pca_scatter(data: Dict, out_dir: Path) -> Path:
    """PCA of TDI fingerprint vectors."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer

    keys = ["tdi_vr_h0", "tdi_vr_h1", "tdi_alpha_h0", "tdi_alpha_h1",
            "tdi_rand", "signal_ratio", "entropy_h0", "purity_gain"]
    X = np.column_stack([[v if v is not None else np.nan for v in data[k]] for k in keys])
    X = SimpleImputer(strategy="median").fit_transform(X)
    Z = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    seen = set()
    for i, (name, domain) in enumerate(zip(data["names"], data["domains"])):
        color = DOMAIN_COLORS.get(domain, "#95a5a6")
        label = domain if domain not in seen else ""
        seen.add(domain)
        ax.scatter(Z[i, 0], Z[i, 1], c=color, s=60, zorder=3, label=label)
        ax.annotate(name[:14], (Z[i, 0], Z[i, 1]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", alpha=0.7)
    ax.set_title("PCA of TDI Fingerprint — Dataset Topology Archetypes", fontsize=12)
    ax.set_xlabel("PC-1"); ax.set_ylabel("PC-2")
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "fig2_pca_fingerprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2: {out}")
    return out


def fig3_umap_scatter(data: Dict, out_dir: Path) -> Path:
    """UMAP of TDI fingerprint vectors."""
    from sklearn.impute import SimpleImputer

    keys = ["tdi_vr_h0", "tdi_vr_h1", "tdi_alpha_h0", "tdi_alpha_h1",
            "tdi_rand", "signal_ratio", "entropy_h0", "purity_gain"]
    X = np.column_stack([[v if v is not None else np.nan for v in data[k]] for k in keys])
    X = SimpleImputer(strategy="median").fit_transform(X)

    try:
        from umap import UMAP
        Z = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X)-1)).fit_transform(X)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//3)).fit_transform(X)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(10, 7))
    seen = set()
    for i, (name, domain) in enumerate(zip(data["names"], data["domains"])):
        color = DOMAIN_COLORS.get(domain, "#95a5a6")
        label = domain if domain not in seen else ""
        seen.add(domain)
        ax.scatter(Z[i, 0], Z[i, 1], c=color, s=60, zorder=3, label=label)
        ax.annotate(name[:14], (Z[i, 0], Z[i, 1]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", alpha=0.7)
    ax.set_title(f"{method} of TDI Fingerprint — Topology Neighbourhood Structure", fontsize=12)
    ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")
    ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "fig3_umap_fingerprint.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3: {out}")
    return out


def fig4_domain_violin(data: Dict, out_dir: Path) -> Path:
    """Box plots of TDI_VR_H0 grouped by domain."""
    from collections import defaultdict
    domain_tdi = defaultdict(list)
    for tdi, domain in zip(data["tdi_vr_h0"], data["domains"]):
        if tdi is not None:
            domain_tdi[domain].append(tdi)
    domains_sorted = sorted(domain_tdi.keys(), key=lambda d: np.median(domain_tdi[d]))
    values = [domain_tdi[d] for d in domains_sorted]
    colors = [DOMAIN_COLORS.get(d, "#95a5a6") for d in domains_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(domains_sorted) * 0.9), 6))
    bp = ax.boxplot(values, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(domains_sorted) + 1))
    ax.set_xticklabels(domains_sorted, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("TDI (VR, H₀)")
    ax.set_title("TDI Distribution by Domain — Vietoris-Rips H₀", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig4_domain_violin.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4: {out}")
    return out


def fig5_signal_ratio(data: Dict, out_dir: Path) -> Path:
    """Bar chart: TDI_random_label / TDI_trained (higher = stronger topology signal)."""
    pairs = [(n, r) for n, r in zip(data["names"], data["signal_ratio"])
             if r is not None]
    pairs.sort(key=lambda x: x[1], reverse=True)
    names, ratios = zip(*pairs) if pairs else ([], [])

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 6))
    colors_bar = [DOMAIN_COLORS.get(
        data["domains"][data["names"].index(n)], "#95a5a6") for n in names]
    ax.bar(range(len(names)), ratios, color=colors_bar, alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="ratio = 1 (no signal)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("TDI Signal Ratio  (rand_label / trained)")
    ax.set_title("Topology Signal Strength per Dataset\n"
                 "Higher = random labels deform topology more than true labels", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = out_dir / "fig5_signal_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5: {out}")
    return out


def fig6_accuracy_vs_tdi(data: Dict, out_dir: Path) -> Path:
    """Scatter: accuracy vs TDI_VR_H0, coloured by domain."""
    fig, ax = plt.subplots(figsize=(9, 6))
    seen = set()
    for acc, tdi, name, domain in zip(data["accuracy"], data["tdi_vr_h0"],
                                       data["names"], data["domains"]):
        if acc is None or tdi is None:
            continue
        color = DOMAIN_COLORS.get(domain, "#95a5a6")
        label = domain if domain not in seen else ""
        seen.add(domain)
        ax.scatter(tdi, acc, c=color, s=55, zorder=3, label=label, alpha=0.85)
        ax.annotate(name[:12], (tdi, acc), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", alpha=0.65)

    # Trend line
    valid = [(t, a) for t, a in zip(data["tdi_vr_h0"], data["accuracy"])
             if t is not None and a is not None]
    if len(valid) > 3:
        ts, acs = zip(*valid)
        z = np.polyfit(ts, acs, 1)
        xr = np.linspace(min(ts), max(ts), 100)
        ax.plot(xr, np.polyval(z, xr), "--", color="#2c3e50", linewidth=1.5,
                alpha=0.6, label="trend")
        r = np.corrcoef(ts, acs)[0, 1]
        ax.text(0.97, 0.04, f"Pearson r = {r:.3f}", transform=ax.transAxes,
                ha="right", fontsize=10, color="#2c3e50")

    ax.set_xlabel("TDI  (Vietoris-Rips, H₀)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("TDI vs Accuracy — Does Topological Complexity Predict Performance?", fontsize=11)
    ax.legend(fontsize=7, ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "fig6_accuracy_vs_tdi.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig6: {out}")
    return out


def fig7_full_tdi_matrix(data: Dict, out_dir: Path) -> Path:
    """Full numeric TDI atlas matrix as annotated heatmap (paper-ready)."""
    from sklearn.impute import SimpleImputer
    col_keys = ["tdi_vr_h0", "tdi_vr_h1", "tdi_alpha_h0", "tdi_alpha_h1",
                "tdi_rand", "signal_ratio", "purity_gain", "entropy_h0"]
    col_labels = ["VR-H₀", "VR-H₁", "α-H₀", "α-H₁",
                  "TDI_rand", "Sig.Ratio", "Purity↑", "Ent-H₀"]

    mat = np.column_stack([[v if v is not None else np.nan for v in data[k]]
                           for k in col_keys])
    # Normalise each column to [0,1] for visual comparison
    mat_norm = (mat - np.nanmin(mat, axis=0)) / (np.nanmax(mat, axis=0) - np.nanmin(mat, axis=0) + 1e-9)

    N = len(data["names"])
    fig_h = max(8, N * 0.25)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    im = ax.imshow(mat_norm, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=25, ha="right")
    ax.set_yticks(range(N))
    ax.set_yticklabels(data["names"], fontsize=7)
    # Annotate raw values
    for i in range(N):
        for j in range(len(col_keys)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=5, color="white" if mat_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="Normalised value", fraction=0.015, pad=0.01)
    ax.set_title("TDI Atlas — Full Topology Fingerprint Matrix", fontsize=12)
    plt.tight_layout()
    out = out_dir / "fig7_full_tdi_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig7: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description="TDI Atlas Visualiser")
    p.add_argument("--atlas", default="results/tdi_atlas.csv")
    p.add_argument("--out", default="results/figures/atlas")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading atlas from {args.atlas} ...")
    records = load_atlas(args.atlas)
    print(f"  {len(records)} datasets")

    data = build_matrix(records)

    print("\nGenerating figures:")
    fig1_tdi_heatmap(data, out_dir)
    fig2_pca_scatter(data, out_dir)
    fig3_umap_scatter(data, out_dir)
    fig4_domain_violin(data, out_dir)
    fig5_signal_ratio(data, out_dir)
    fig6_accuracy_vs_tdi(data, out_dir)
    fig7_full_tdi_matrix(data, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
