from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .window_features import model_columns


def _spatial_sample(frame, maximum, window_size, stride, seed):
    """Prefer non-overlapping windows when estimating cluster separation."""
    spacing = max(1, math.ceil(window_size / stride))
    separated = frame.index[
        frame.grid_row.mod(spacing).eq(0) & frame.grid_col.mod(spacing).eq(0)
    ].to_numpy()
    candidates = separated if len(separated) >= 3 else frame.index.to_numpy()
    if len(candidates) <= maximum:
        return candidates
    return np.random.default_rng(seed).choice(candidates, maximum, replace=False)


def _elbow(k_values, inertias):
    """Find maximum distance from the normalized inertia end-point line."""
    if len(k_values) < 3:
        return int(k_values[np.argmin(inertias)])
    x = np.asarray(k_values, float)
    y = np.asarray(inertias, float)
    x = (x - x.min()) / max(np.ptp(x), 1e-12)
    y = (y - y.min()) / max(np.ptp(y), 1e-12)
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    points = np.column_stack([x - x[0], y - y[0]])
    distances = np.abs(line[0] * points[:, 1] - line[1] * points[:, 0])
    distances /= max(np.linalg.norm(line), 1e-12)
    return int(k_values[int(np.argmax(distances))])


def _plot_selection(scores: pd.DataFrame, selected_k: int, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scores.k, scores.inertia, "o-", color="#0072B2")
    axes[0].axvline(selected_k, color="#D55E00", linestyle="--")
    axes[0].set(title="Elbow analysis", xlabel="Number of clusters", ylabel="Inertia")
    axes[1].plot(scores.k, scores.silhouette, "o-", color="#009E73")
    axes[1].axvline(selected_k, color="#D55E00", linestyle="--")
    axes[1].set(title="Silhouette analysis", xlabel="Number of clusters", ylabel="Score")
    fig.suptitle(f"Automatically selected k = {selected_k}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fit_clusters(frame, cfg, output_dir: Path):
    """Select k with elbow + silhouette, then fit all non-empty windows."""
    features = model_columns(frame)
    preprocess = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    x = preprocess.fit_transform(frame[features])
    # Retain at least two axes and approximately 90% of non-redundant variance.
    full = PCA(random_state=cfg.random_seed).fit(x)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = min(x.shape[1], max(2, int(np.searchsorted(cumulative, .90) + 1)))
    xp = PCA(n_components=n_components, random_state=cfg.random_seed).fit_transform(x)
    sample_idx = _spatial_sample(
        frame, cfg.silhouette_sample, cfg.window_size_px, cfg.stride_px,
        cfg.random_seed,
    )
    max_k = min(cfg.k_max, len(sample_idx) - 1, len(frame) - 1)
    if max_k < cfg.k_min:
        raise ValueError("Not enough non-empty windows to evaluate cluster number")

    rows = []
    models = {}
    minimum_size = max(2, math.ceil(cfg.minimum_cluster_fraction * len(frame)))
    for k in range(cfg.k_min, max_k + 1):
        model = MiniBatchKMeans(
            n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init=10
        ).fit(xp)
        labels = model.labels_
        sampled_labels = labels[sample_idx]
        silhouette = (
            silhouette_score(xp[sample_idx], sampled_labels)
            if 1 < len(np.unique(sampled_labels)) < len(sampled_labels) else -1.0
        )
        smallest = int(np.bincount(labels, minlength=k).min())
        rows.append({
            "k": k, "inertia": float(model.inertia_),
            "silhouette": float(silhouette), "smallest_cluster": smallest,
            "valid_size": smallest >= minimum_size,
        })
        models[k] = model
    scores = pd.DataFrame(rows)
    valid = scores.loc[scores.valid_size]
    if valid.empty:
        valid = scores
    elbow_k = _elbow(valid.k.to_numpy(), valid.inertia.to_numpy())
    best_row = valid.loc[valid.silhouette.idxmax()]
    elbow_silhouette = float(valid.loc[valid.k.eq(elbow_k), "silhouette"].iloc[0])
    best_silhouette = float(best_row.silhouette)
    close_enough = (
        elbow_silhouette >= .95 * best_silhouette
        if best_silhouette > 0 else elbow_silhouette >= best_silhouette - .02
    )
    selected_k = elbow_k if close_enough else int(best_row.k)
    selected = models[selected_k]

    repeat_ari = []
    for repeat in range(1, max(1, cfg.stability_repeats)):
        labels = MiniBatchKMeans(
            n_clusters=selected_k, random_state=cfg.random_seed + repeat,
            batch_size=2048, n_init=10,
        ).fit_predict(xp)
        repeat_ari.append(adjusted_rand_score(selected.labels_, labels))
    stability = float(np.mean(repeat_ari)) if repeat_ari else 1.0

    result = frame.copy()
    result["cluster"] = selected.labels_.astype(int)
    # Make label identities and colours deterministic across equivalent runs.
    centroid_order = np.argsort(selected.cluster_centers_[:, 0])
    relabel = np.empty(selected_k, dtype=int)
    relabel[centroid_order] = np.arange(selected_k)
    result["cluster"] = relabel[result.cluster.to_numpy()]
    _plot_selection(scores, selected_k, output_dir / "cluster_selection.png")
    return result, {
        "selected_k": selected_k,
        "elbow_k": elbow_k,
        "best_silhouette_k": int(best_row.k),
        "best_silhouette": best_silhouette,
        "selected_silhouette": float(
            scores.loc[scores.k.eq(selected_k), "silhouette"].iloc[0]
        ),
        "stability_ari": stability,
        "stable": stability >= .80,
        "pca_components": n_components,
        "pca_variance_explained": float(cumulative[n_components - 1]),
        "windows": len(frame),
        "spatially_separated_windows_for_silhouette": len(sample_idx),
    }
