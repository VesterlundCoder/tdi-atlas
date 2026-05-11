"""
TDI Atlas — Topological Deformation Index mapping across datasets and filtrations.

Sweeps 50+ OpenML/sklearn/CMF datasets through multiple persistent homology
filtrations and model architectures. Outputs a structured atlas CSV and JSON
suitable for downstream PCA/UMAP visualisation and cross-domain comparison.

Usage:
    python tdi_atlas.py                            # full run (~2h)
    python tdi_atlas.py --fast                     # 10 datasets, quick smoke test
    python tdi_atlas.py --datasets wine iris       # specific datasets only
    python tdi_atlas.py --out results/atlas.csv

Dependencies:
    pip install scikit-learn openml ripser persim torch numpy scipy umap-learn
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import openml
    _OPENML_OK = True
except ImportError:
    _OPENML_OK = False
    warnings.warn("openml not installed. Install with: pip install openml")

try:
    from ripser import ripser as _ripser
    _RIPSER_OK = True
except ImportError:
    _RIPSER_OK = False

try:
    from persim import wasserstein as _wasserstein
    _PERSIM_OK = True
except ImportError:
    _PERSIM_OK = False

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dataset registry ──────────────────────────────────────────────────────────

SKLEARN_DATASETS = [
    "wine", "iris", "breast_cancer", "digits"
]

# OpenML dataset IDs covering diverse domains.
# Format: (openml_id, short_name, domain)
OPENML_REGISTRY = [
    # ── Physics / signal ────────────────────────────────────────────────────
    (61,    "ionosphere",           "physics"),
    (40,    "sonar",                "physics"),
    (1120,  "magic",                "physics"),
    (60,    "waveform_5000",        "physics"),
    (300,   "isolet",               "speech"),
    (1489,  "phoneme",              "speech"),
    (40685, "shuttle",              "aerospace"),
    (1216,  "climate_model",        "engineering"),
    (41138, "APSFailure",           "engineering"),
    # ── Medicine / biology ───────────────────────────────────────────────────
    (37,    "diabetes",             "medicine"),
    (1510,  "wdbc",                 "medicine"),
    (1464,  "blood_transfusion",    "medicine"),
    (15,    "breast_w",             "medicine"),
    (38,    "sick",                 "medicine"),
    (55,    "hepatitis",            "medicine"),
    (57,    "hypothyroid",          "medicine"),
    (1466,  "cardiotocography",     "medicine"),
    (1100,  "arrhythmia",           "medicine"),
    (40981, "cardio2",              "medicine"),
    (1217,  "yeast",                "biology"),
    (1484,  "miceprotein",          "biology"),
    (40706, "cancer_gene_expr",     "biology"),
    (1504,  "steel_plates_fault",   "engineering"),
    # ── Finance / social ─────────────────────────────────────────────────────
    (1462,  "banknote",             "finance"),
    (1461,  "bank_marketing",       "finance"),
    (31,    "credit_g",             "finance"),
    (29,    "credit_approval",      "finance"),
    (1590,  "adult",                "social"),
    (40701, "churn",                "finance"),
    (40900, "Titanic",              "social"),
    (23,    "contraceptive",        "social"),
    (4154,  "Buzzinsocialmedia",    "social"),
    (1075,  "credit_card_default",  "finance"),
    # ── Software metrics ─────────────────────────────────────────────────────
    (1063,  "kc1",                  "software"),
    (1068,  "pc1",                  "software"),
    (1049,  "pc4",                  "software"),
    (1050,  "pc3",                  "software"),
    (1056,  "mc1",                  "software"),
    (1046,  "mozilla4",             "software"),
    (1048,  "kc2",                  "software"),
    (1067,  "kc3",                  "software"),
    (1071,  "jm1",                  "software"),
    # ── Vision / digits ──────────────────────────────────────────────────────
    (554,   "mnist_784",            "vision"),
    (40996, "fashion_mnist",        "vision"),
    (6,     "letter",               "vision"),
    (28,    "optdigits",            "vision"),
    (32,    "pendigits",            "vision"),
    (1501,  "semeion",              "digits"),
    (14,    "mfeat_fourier",        "vision"),
    (16,    "mfeat_karhunen",       "vision"),
    (18,    "mfeat_morphological",  "vision"),
    (22,    "mfeat_zernike",        "vision"),
    (12,    "mfeat_factors",        "vision"),
    (36,    "segment",              "vision"),
    (40979, "movement_libras",      "vision"),
    (40980, "pendigits2",           "vision"),
    (41147, "fabert",               "vision"),
    # ── NLP / text features ──────────────────────────────────────────────────
    (44,    "spambase",             "nlp_features"),
    (4135,  "Amazon_reviews",       "nlp_features"),
    (458,   "analcatdata_authorship","nlp_features"),
    (1468,  "cnae",                 "nlp_features"),
    (4534,  "PhishingWebsites",     "cybersecurity"),
    # ── Ecology / remote sensing ─────────────────────────────────────────────
    (1043,  "sylva_agnostic",       "ecology"),
    (40983, "wilt",                 "ecology"),
    (40975, "satellite",            "remote_sensing"),
    (69,    "satimage",             "remote_sensing"),
    (40675, "gas_drift",            "chemistry"),
    (1494,  "qsar_biodeg",          "chemistry"),
    (1161,  "seismic_bumps",        "engineering"),
    # ── Energy / games / misc ────────────────────────────────────────────────
    (1169,  "electricity",          "energy"),
    (40668, "connect_4",            "games"),
    (3,     "kr_vs_kp",             "games"),
    (50,    "tic_tac_toe",          "games"),
    (24,    "mushroom",             "biology"),
    (11,    "balance_scale",        "psychology"),
    (53,    "heart_statlog",        "medicine"),
    (10,    "lymph",                "medicine"),
    (2,     "anneal",               "engineering"),
    (21,    "car",                  "engineering"),
    (48,    "australian",           "finance"),
    (70,    "zoo",                  "biology"),
    (41,    "glass",                "materials"),
    (42,    "soybean",              "agriculture"),
    (46,    "splice",               "genomics"),
    (188,   "eucalyptus",           "ecology"),
    (54,    "vehicle",              "robotics"),
    (469,   "analcatdata_dmft",     "dentistry"),
    (1038,  "gina_agnostic",        "synthetic"),
    (1478,  "har",                  "robotics"),
    (1497,  "wall_robot",           "robotics"),
    (1473,  "ilpd",                 "medicine"),
    (1476,  "nomao",                "nlp_features"),
    (1487,  "ozone",                "environment"),
    (1475,  "first_order",          "synthetic"),
    (1479,  "hill_valley",          "synthetic"),
    (40496, "Thai_Student",         "education"),
    (1242,  "eye_movements",        "neuroscience"),
    (40691, "wine_quality_red",     "food"),
    (40693, "wine_quality_white",   "food"),
    (1480,  "volkert",              "synthetic"),
    (1491,  "100_plants_texture",   "ecology"),
    (1492,  "100_plants_margin",    "ecology"),
    (1241,  "synthetic_control",    "synthetic"),
    (1563,  "gesture_phase",        "robotics"),
    (1568,  "eeg_eye_state",        "neuroscience"),
    (40710, "hapt",                 "robotics"),
    (40711, "vowel_context",        "speech"),
    (40984, "australian2",          "finance"),
    (41142, "christine",            "synthetic"),
    (41143, "jasmine",              "synthetic"),
    (41144, "madeline",             "synthetic"),
    (41145, "philippine",           "synthetic"),
    (41146, "sylvine",              "synthetic"),
]

SYNTHETIC_DATASETS = [
    "swiss_roll", "two_moons", "two_circles",
    "blobs_2d", "blobs_10d", "aniso"
]

FAST_DATASETS = [
    "wine", "iris", "breast_cancer",
    ("openml", 61, "ionosphere", "physics"),
    ("openml", 1462, "banknote", "finance"),
    ("openml", 37, "diabetes", "medicine"),
    ("openml", 54, "vehicle", "robotics"),
    ("openml", 44, "spambase", "nlp_features"),
    ("synthetic", "two_moons"),
    ("synthetic", "swiss_roll"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    X: np.ndarray
    y: np.ndarray
    name: str
    domain: str
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    n_classes: int = field(init=False)

    def __post_init__(self):
        self.n_samples = len(self.X)
        self.n_features = self.X.shape[1]
        self.n_classes = len(np.unique(self.y))


def _cap_and_scale(X: np.ndarray, y: np.ndarray,
                   max_per_class: int = 1000,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified cap + standardise."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = min(len(idx), max_per_class)
        keep.extend(rng.choice(idx, n, replace=False).tolist())
    keep = np.array(keep)
    rng.shuffle(keep)
    X2 = StandardScaler().fit_transform(X[keep]).astype(np.float32)
    return X2, y[keep].astype(np.int64)


def load_sklearn(name: str) -> DataBundle:
    from sklearn import datasets as skd
    loaders = {
        "wine": skd.load_wine,
        "iris": skd.load_iris,
        "breast_cancer": skd.load_breast_cancer,
        "digits": skd.load_digits,
    }
    d = loaders[name]()
    X, y = _cap_and_scale(d.data.astype(np.float32), d.target.astype(np.int64))
    return DataBundle(X=X, y=y, name=name, domain="biology")


def load_openml(dataset_id: int, name: str, domain: str,
                max_per_class: int = 1000) -> Optional[DataBundle]:
    if not _OPENML_OK:
        return None
    try:
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_df, y_ser, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        X = X_df.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(y_ser.values.ravel()).astype(np.int64)
        if X.shape[0] < 30 or X.shape[1] < 2 or len(np.unique(y)) < 2:
            return None
        X, y = _cap_and_scale(X, y, max_per_class=max_per_class)
        return DataBundle(X=X, y=y, name=name, domain=domain)
    except Exception as e:
        warnings.warn(f"OpenML {dataset_id} ({name}) failed: {e}")
        return None


def load_synthetic(name: str) -> DataBundle:
    from sklearn import datasets as skd
    n = 500
    if name == "swiss_roll":
        X, _ = skd.make_swiss_roll(n_samples=n, noise=0.1, random_state=0)
        y = (X[:, 1] > np.median(X[:, 1])).astype(np.int64)
    elif name == "two_moons":
        X, y = skd.make_moons(n_samples=n, noise=0.15, random_state=0)
    elif name == "two_circles":
        X, y = skd.make_circles(n_samples=n, noise=0.05, factor=0.5, random_state=0)
    elif name == "blobs_2d":
        X, y = skd.make_blobs(n_samples=n, n_features=2, centers=4, random_state=0)
    elif name == "blobs_10d":
        X, y = skd.make_blobs(n_samples=n, n_features=10, centers=5, random_state=0)
    elif name == "aniso":
        X, y = skd.make_blobs(n_samples=n, centers=3, random_state=0)
        rng = np.random.RandomState(0)
        X = X @ rng.randn(2, 2)
    else:
        raise ValueError(f"Unknown synthetic dataset: {name}")
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return DataBundle(X=X, y=y.astype(np.int64), name=name, domain="synthetic")


def load_cmf_hunt(shards_dir: str, max_per_class: int = 800) -> Optional[DataBundle]:
    import glob
    path = Path(shards_dir)
    if not path.exists():
        return None
    features, labels = [], []
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
                def _stats(d):
                    vals = [float(v) for v in d.values() if v is not None] if d else []
                    if not vals:
                        return [0.0] * 5
                    return [np.mean(vals), np.std(vals), np.min(vals), np.max(vals), len(vals)]
                l_s = _stats(params.get("L_off", {}))
                u_s = _stats(params.get("U_off", {}))
                try:
                    lim = abs(float(str(rec.get("source_limit", "0"))[:30]))
                except Exception:
                    lim = 0.0
                fv = np.array(d_flat + [score] + l_s + u_s + [lim], dtype=np.float32)
                features.append(fv)
                labels.append(str(rec.get("source_tier", "?")))
    if not features:
        return None
    max_len = max(len(f) for f in features)
    X_all = np.stack([np.pad(f, (0, max_len - len(f))) for f in features])
    le = LabelEncoder()
    y_all = le.fit_transform(labels).astype(np.int64)
    X_all, y_all = _cap_and_scale(X_all, y_all, max_per_class=max_per_class)
    X_all = StandardScaler().fit_transform(X_all).astype(np.float32)
    return DataBundle(X=X_all, y=y_all, name="cmf_hunt", domain="mathematics")


# ── Persistent homology ───────────────────────────────────────────────────────

def compute_ph_vr(X: np.ndarray, max_dim: int = 1, max_pts: int = 400) -> Dict:
    """Vietoris-Rips persistent homology."""
    if len(X) > max_pts:
        idx = np.random.default_rng(0).choice(len(X), max_pts, replace=False)
        X = X[idx]
    if _RIPSER_OK:
        result = _ripser(X, maxdim=max_dim)
        dgms = result["dgms"]
        dgm0 = dgms[0][np.isfinite(dgms[0][:, 1])]
        dgm1 = dgms[1][np.isfinite(dgms[1][:, 1])] if max_dim >= 1 else np.zeros((0, 2))
        return {"dgm_0": dgm0, "dgm_1": dgm1, "method": "ripser_vr"}
    else:
        return _scipy_h0(X)


def compute_ph_alpha(X: np.ndarray, max_pts: int = 300) -> Dict:
    """Alpha complex PH via gudhi (falls back to VR if unavailable)."""
    if len(X) > max_pts:
        idx = np.random.default_rng(1).choice(len(X), max_pts, replace=False)
        X = X[idx]
    try:
        import gudhi
        alpha = gudhi.AlphaComplex(points=X[:, :min(X.shape[1], 3)].tolist())
        st = alpha.create_simplex_tree()
        st.compute_persistence()
        pairs = st.persistence_pairs()
        dgm0, dgm1 = [], []
        for birth_simplices, death_simplices in pairs:
            if not birth_simplices or not death_simplices:
                continue
            b = st.filtration(birth_simplices)
            d = st.filtration(death_simplices)
            if np.isfinite(b) and np.isfinite(d):
                dim = len(birth_simplices) - 1
                if dim == 0:
                    dgm0.append([b, d])
                elif dim == 1:
                    dgm1.append([b, d])
        return {
            "dgm_0": np.array(dgm0) if dgm0 else np.zeros((0, 2)),
            "dgm_1": np.array(dgm1) if dgm1 else np.zeros((0, 2)),
            "method": "gudhi_alpha",
        }
    except ImportError:
        return compute_ph_vr(X, max_pts=max_pts)


def _scipy_h0(X: np.ndarray) -> Dict:
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(X, method="single", metric="euclidean")
    lifetimes = Z[:, 2]
    dgm0 = np.column_stack([np.zeros(len(lifetimes)), lifetimes])
    return {"dgm_0": dgm0, "dgm_1": np.zeros((0, 2)), "method": "scipy_h0"}


def persistence_entropy(dgm: np.ndarray) -> float:
    if len(dgm) == 0:
        return 0.0
    lt = dgm[:, 1] - dgm[:, 0]
    lt = lt[lt > 1e-12]
    if len(lt) == 0:
        return 0.0
    lt = lt / lt.sum()
    return float(-np.sum(lt * np.log(lt + 1e-12)))


def diagram_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
    if _PERSIM_OK:
        try:
            return float(_wasserstein(dgm1, dgm2))
        except Exception:
            pass
    def _lt_vec(d, k=20):
        if len(d) == 0:
            return np.zeros(k)
        lt = np.sort(d[:, 1] - d[:, 0])[::-1]
        return np.pad(lt[:k], (0, max(0, k - len(lt))))
    return float(np.linalg.norm(_lt_vec(dgm1) - _lt_vec(dgm2)))


FILTRATIONS = {
    "vr":    lambda X: compute_ph_vr(X, max_dim=1),
    "alpha": lambda X: compute_ph_alpha(X),
}


# ── MLP model ─────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self.hidden_dims = hidden
        self._activations: Dict[str, np.ndarray] = {}
        self._hooks = []

    def forward(self, x):
        return self.net(x)

    def register_hooks(self):
        self._activations = {}
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.ReLU):
                def _hook(mod, inp, out, idx=i):
                    self._activations[f"layer_{idx}"] = out.detach().cpu().numpy()
                self._hooks.append(layer.register_forward_hook(_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_layer_reps(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        self.register_hooks()
        self.eval()
        with torch.no_grad():
            _ = self.forward(torch.tensor(X))
        reps = dict(self._activations)
        self.remove_hooks()
        reps["input"] = X
        return reps


def train_mlp(X_tr, y_tr, X_val, y_val,
              hidden=(64, 32), epochs=100, seed=0,
              batch_size=256) -> Tuple[MLP, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_classes = int(y_tr.max()) + 1
    model = MLP(X_tr.shape[1], list(hidden), n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(X_tr)
    Xtr_t = torch.tensor(X_tr)
    ytr_t = torch.tensor(y_tr)
    use_batch = N > batch_size
    for epoch in range(epochs):
        model.train()
        if use_batch:
            perm = rng.permutation(N)
            for start in range(0, N, batch_size):
                bidx = perm[start:start + batch_size]
                loss = F.cross_entropy(model(Xtr_t[bidx]), ytr_t[bidx])
                opt.zero_grad(); loss.backward(); opt.step()
        else:
            loss = F.cross_entropy(model(Xtr_t), ytr_t)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = accuracy_score(y_val, model(torch.tensor(X_val)).argmax(1).numpy())
    return model, float(acc)


# ── TDI computation ───────────────────────────────────────────────────────────

def compute_tdi_for_reps(layer_reps: Dict[str, np.ndarray],
                          filtration_fn, dim: int = 0) -> Tuple[float, List[float]]:
    dgm_key = f"dgm_{dim}"
    names = sorted(layer_reps.keys())
    diagrams = []
    for name in names:
        ph = filtration_fn(layer_reps[name])
        diagrams.append(ph[dgm_key])
    dists = []
    for i in range(len(diagrams) - 1):
        dists.append(diagram_distance(diagrams[i], diagrams[i + 1]))
    tdi = float(sum(dists))
    return tdi, dists


def knn_purity(X: np.ndarray, y: np.ndarray, k: int = 5, max_pts: int = 400) -> float:
    N = len(X)
    if N <= k:
        return 0.0
    if N > max_pts:
        idx = np.random.default_rng(3).choice(N, max_pts, replace=False)
        X, y = X[idx], y[idx]
        N = max_pts
    D = cdist(X, X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argsort(D, axis=1)[:, :k]
    return float(np.mean([np.mean(y[nn_idx[i]] == y[i]) for i in range(N)]))


# ── Per-dataset sweep ─────────────────────────────────────────────────────────

def sweep_dataset(bundle: DataBundle,
                  epochs: int = 100,
                  seed: int = 0,
                  verbose: bool = True) -> Dict:
    """Full TDI sweep for one dataset across all filtrations × homology dims."""
    t0 = time.time()
    X_tr, X_te, y_tr, y_te = train_test_split(
        bundle.X, bundle.y, test_size=0.25, stratify=bundle.y, random_state=seed
    )

    # Train MLP
    model, acc = train_mlp(X_tr, y_tr, X_te, y_te, epochs=epochs, seed=seed)

    # Extract layer representations on full dataset
    layer_reps = model.get_layer_reps(bundle.X)

    # Input topology baseline
    input_ph_vr = compute_ph_vr(bundle.X)
    input_entropy_h0 = persistence_entropy(input_ph_vr["dgm_0"])
    input_entropy_h1 = persistence_entropy(input_ph_vr["dgm_1"])
    input_purity = knn_purity(bundle.X, bundle.y)

    # TDI across filtrations × dimensions
    tdi_matrix: Dict[str, Dict[str, float]] = {}
    for filt_name, filt_fn in FILTRATIONS.items():
        tdi_matrix[filt_name] = {}
        for dim in [0, 1]:
            try:
                tdi_val, _ = compute_tdi_for_reps(layer_reps, filt_fn, dim=dim)
                tdi_matrix[filt_name][f"h{dim}"] = tdi_val
            except Exception as e:
                tdi_matrix[filt_name][f"h{dim}"] = None

    # Random-label control TDI (VR H0)
    y_rand = np.random.default_rng(99).permutation(bundle.y)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        bundle.X, y_rand, test_size=0.25, random_state=seed
    )
    model_rand, _ = train_mlp(X_tr_r, y_tr_r, X_te_r, y_te_r, epochs=epochs, seed=seed)
    reps_rand = model_rand.get_layer_reps(bundle.X)
    tdi_rand, _ = compute_tdi_for_reps(reps_rand, FILTRATIONS["vr"], dim=0)

    # Penultimate-layer purity
    layer_names = sorted(layer_reps.keys())
    penult_key = layer_names[-2] if len(layer_names) >= 2 else layer_names[-1]
    final_purity = knn_purity(layer_reps[penult_key], bundle.y)

    elapsed = time.time() - t0

    result = {
        "dataset":          bundle.name,
        "domain":           bundle.domain,
        "n_samples":        bundle.n_samples,
        "n_features":       bundle.n_features,
        "n_classes":        bundle.n_classes,
        "accuracy":         acc,
        "tdi_matrix":       tdi_matrix,
        "tdi_vr_h0":        tdi_matrix.get("vr", {}).get("h0"),
        "tdi_vr_h1":        tdi_matrix.get("vr", {}).get("h1"),
        "tdi_alpha_h0":     tdi_matrix.get("alpha", {}).get("h0"),
        "tdi_alpha_h1":     tdi_matrix.get("alpha", {}).get("h1"),
        "tdi_random_label": tdi_rand,
        "signal_ratio":     (tdi_rand / tdi_matrix["vr"]["h0"])
                            if tdi_matrix["vr"].get("h0") else None,
        "input_entropy_h0": input_entropy_h0,
        "input_entropy_h1": input_entropy_h1,
        "input_knn_purity": input_purity,
        "final_knn_purity": final_purity,
        "purity_gain":      final_purity - input_purity,
        "elapsed_s":        elapsed,
    }

    if verbose:
        print(
            f"  {bundle.name:<28} "
            f"N={bundle.n_samples:<5} D={bundle.n_features:<4} "
            f"acc={acc:.3f}  "
            f"TDI_VR={result['tdi_vr_h0']:.2f}  "
            f"TDI_rand={tdi_rand:.2f}  "
            f"ratio={result['signal_ratio']:.2f}  "
            f"({elapsed:.1f}s)"
        )
    return result


# ── Dataset iterator ──────────────────────────────────────────────────────────

def iter_datasets(args) -> List[DataBundle]:
    bundles = []

    # sklearn
    for name in SKLEARN_DATASETS:
        if args.datasets and name not in args.datasets:
            continue
        try:
            bundles.append(load_sklearn(name))
        except Exception as e:
            warnings.warn(f"sklearn {name}: {e}")

    # Synthetic
    for name in SYNTHETIC_DATASETS:
        if args.datasets and name not in args.datasets:
            continue
        try:
            bundles.append(load_synthetic(name))
        except Exception as e:
            warnings.warn(f"synthetic {name}: {e}")

    # CMF
    if not args.datasets or "cmf" in args.datasets:
        cmf_path = args.cmf_shards
        if cmf_path and Path(cmf_path).exists():
            b = load_cmf_hunt(cmf_path)
            if b:
                bundles.append(b)

    # OpenML
    if _OPENML_OK:
        registry = OPENML_REGISTRY if not args.fast else OPENML_REGISTRY[:8]
        for (oid, name, domain) in registry:
            if args.datasets and name not in args.datasets:
                continue
            b = load_openml(oid, name, domain)
            if b:
                bundles.append(b)

    return bundles


# ── Main ──────────────────────────────────────────────────────────────────────

def iter_openml_catalog(n_wanted: int, exclude_names: set,
                         seed: int = 42) -> List[Tuple[int, str, str]]:
    """
    Dynamically query OpenML for classification datasets not already in
    OPENML_REGISTRY, filtered to viable size/feature ranges.
    Returns up to n_wanted (id, name, domain) tuples.
    """
    if not _OPENML_OK:
        return []
    try:
        import pandas as pd
        print(f"  Querying OpenML catalog for up to {n_wanted} additional datasets...")
        df = openml.datasets.list_datasets(output_format="dataframe")
        # Filter to classification-compatible datasets
        df = df[
            (df["NumberOfInstances"].between(150, 100_000)) &
            (df["NumberOfFeatures"].between(3, 500)) &
            (df["NumberOfClasses"].between(2, 50)) &
            (df["NumberOfMissingValues"] < df["NumberOfInstances"] * 0.3)
        ].copy()
        # Exclude already-registered names
        df["name_lower"] = df["name"].str.lower().str.replace(r"[^a-z0-9]", "_", regex=True)
        already = {n.lower().replace("-","_").replace(" ","_") for n in exclude_names}
        df = df[~df["name_lower"].isin(already)]
        # Deduplicate by name, prefer newer versions
        df = df.sort_values("version", ascending=False).drop_duplicates(subset="name")
        # Sample deterministically
        df = df.sample(frac=1, random_state=seed).head(n_wanted)
        result = []
        for _, row in df.iterrows():
            safe_name = str(row["name"]).replace("-","_").replace(" ","_")[:32]
            result.append((int(row["did"]), safe_name, "openml_catalog"))
        print(f"  Catalog returned {len(result)} additional dataset candidates.")
        return result
    except Exception as e:
        warnings.warn(f"OpenML catalog query failed: {e}")
        return []


def load_existing_results(out_path: Path) -> set:
    """Return set of dataset names already written to the output CSV."""
    if not out_path.exists():
        return set()
    import csv
    done = set()
    try:
        with open(out_path) as f:
            for row in csv.DictReader(f):
                name = row.get("dataset")
                if name:
                    done.add(name)
    except Exception:
        pass
    return done


def write_result_row(out_path: Path, result: Dict, flat_keys: List[str],
                     write_header: bool = False):
    """Append a single result row to the CSV (streaming write)."""
    import csv
    mode = "w" if write_header else "a"
    with open(out_path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(result)


def parse_args():
    p = argparse.ArgumentParser(description="TDI Atlas — cross-domain topology mapping")
    p.add_argument("--fast", action="store_true",
                   help="Quick mode: 10 datasets, 50 epochs")
    p.add_argument("--n-datasets", type=int, default=None,
                   help="Target total number of datasets (fills up with OpenML catalog)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Specific dataset names to run")
    p.add_argument("--out", default="results/tdi_atlas.csv")
    p.add_argument("--resume", action="store_true",
                   help="Skip datasets already present in the output CSV")
    p.add_argument("--cmf-shards", type=str,
                   default="/Users/davidsvensson/Desktop/rd-lumi-z3/cmf_loop_project/hunt_shards",
                   help="Path to CMF hunt_shards/ directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.epochs = min(args.epochs, 50)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")

    flat_keys = [
        "dataset", "domain", "n_samples", "n_features", "n_classes",
        "accuracy", "tdi_vr_h0", "tdi_vr_h1", "tdi_alpha_h0", "tdi_alpha_h1",
        "tdi_random_label", "signal_ratio",
        "input_entropy_h0", "input_entropy_h1",
        "input_knn_purity", "final_knn_purity", "purity_gain",
    ]

    # Resume: find already-completed datasets
    done_names: set = set()
    if args.resume and out_path.exists():
        done_names = load_existing_results(out_path)
        print(f"  Resume mode: {len(done_names)} datasets already in {out_path}")

    verbose = not args.quiet
    print("=" * 70)
    print("TDI ATLAS — Topological Deformation Index Cross-Domain Mapping")
    print(f"  epochs={args.epochs}  fast={args.fast}  n_datasets={args.n_datasets}")
    print(f"  resume={args.resume}  out={args.out}")
    print("=" * 70)

    # Build dataset list (static registry + optional catalog expansion)
    bundles = iter_datasets(args)

    # Optional: pad up to --n-datasets using OpenML catalog
    if args.n_datasets and _OPENML_OK and not args.fast:
        static_names = {b.name for b in bundles}
        n_needed = args.n_datasets - len(bundles)
        if n_needed > 0:
            extra = iter_openml_catalog(
                n_wanted=n_needed + 20,  # ask for extra; some will fail to load
                exclude_names=static_names,
                seed=args.seed,
            )
            for (oid, name, domain) in extra:
                if len(bundles) >= args.n_datasets:
                    break
                b = load_openml(oid, name, domain)
                if b and b.name not in static_names:
                    bundles.append(b)
                    static_names.add(b.name)

    # Filter out already-done datasets (resume mode)
    if done_names:
        before = len(bundles)
        bundles = [b for b in bundles if b.name not in done_names]
        print(f"  Skipping {before - len(bundles)} already-done datasets.")

    total = len(bundles) + len(done_names)
    todo = len(bundles)
    print(f"\nTotal target: {total}  |  To process: {todo}  |  Already done: {len(done_names)}\n")

    if verbose:
        header = (f"  {'#':<5} {'Dataset':<28} {'N':<6} {'D':<5} "
                  f"{'acc':<7} {'TDI_VR':<9} {'TDI_rand':<10} {'ratio':<7} {'time'}")
        print(header)
        print("-" * len(header))

    # Initialise CSV file (write header only if starting fresh)
    needs_header = not out_path.exists() or not args.resume
    if needs_header:
        write_result_row(out_path, {}, flat_keys, write_header=True)

    all_results: List[Dict] = []
    n_done = len(done_names)
    t_start = time.time()

    for bundle in bundles:
        n_done += 1
        try:
            # Inject progress counter into verbose output
            if verbose:
                elapsed_total = time.time() - t_start
                eta = (elapsed_total / max(n_done - len(done_names), 1)) * max(todo - (n_done - len(done_names)), 0)
                print(f"  [{n_done}/{total}]  ", end="", flush=True)
            r = sweep_dataset(bundle, epochs=args.epochs, seed=args.seed, verbose=verbose)
            all_results.append(r)
            # Stream to CSV immediately (safe against crashes)
            write_result_row(out_path, r, flat_keys, write_header=False)
            # Append to JSON incrementally
            with open(json_path, "a") as jf:
                jf.write(json.dumps(r,
                    default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")
        except Exception as e:
            warnings.warn(f"Failed on {bundle.name}: {e}")

    elapsed_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE  {len(all_results)} new datasets processed  "
          f"(total in file: {len(all_results)+len(done_names)})")
    print(f"  Atlas CSV  → {out_path}")
    print(f"  Atlas JSONL → {json_path}")
    print(f"  Wall time: {elapsed_total/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
