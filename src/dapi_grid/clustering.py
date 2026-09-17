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

from .window_features import MORPHOLOGY


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


def _elbow(k_values, inertias, minimum_strength=0.03):
    """Return a supported interior knee, or None for a near-linear curve."""
    if len(k_values) < 3:
        return None
    x = np.asarray(k_values, float)
    y = np.asarray(inertias, float)
    x = (x - x.min()) / max(np.ptp(x), 1e-12)
    y = (y - y.min()) / max(np.ptp(y), 1e-12)
    line = np.array([x[-1] - x[0], y[-1] - y[0]])
    points = np.column_stack([x - x[0], y - y[0]])
    distances = np.abs(line[0] * points[:, 1] - line[1] * points[:, 0])
    distances /= max(np.linalg.norm(line), 1e-12)
    # End points are definitions of the chord, not possible elbows.
    interior = distances[1:-1]
    if not len(interior):
        return None
    index = int(np.argmax(interior)) + 1
    if not np.isfinite(distances[index]) or distances[index] < minimum_strength:
        return None
    return int(k_values[index])


def _plot_selection(scores: pd.DataFrame, selected_k: int | None, path: Path, elbow_k=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scores.k, scores.inertia, "o-", color="#0072B2")
    if selected_k is not None:
        axes[0].axvline(selected_k, color="#D55E00", linestyle="--")
    if elbow_k is not None:
        axes[0].scatter(
            [elbow_k], [scores.loc[scores.k.eq(elbow_k), "inertia"].iloc[0]],
            color="#CC79A7", s=70, zorder=3, label=f"Detected elbow: {elbow_k}",
        )
        axes[0].legend()
    axes[0].set(title="Elbow analysis", xlabel="Number of clusters", ylabel="Inertia")
    axes[1].plot(scores.k, scores.silhouette, "o-", color="#009E73")
    if selected_k is not None:
        axes[1].axvline(selected_k, color="#D55E00", linestyle="--")
    axes[1].set(title="Silhouette analysis", xlabel="Number of clusters", ylabel="Score")
    fig.suptitle(
        f"Selected k = {selected_k}" if selected_k is not None
        else "Choose k after reviewing both diagnostics"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_cluster_count(scores, valid, models, cfg, elbow_k, best_row, graph_path):
    """Choose k by prompt, automatic diagnostics, or an explicit fixed value."""
    _plot_selection(scores, None, graph_path, elbow_k=elbow_k)
    best_silhouette = float(best_row.silhouette)
    mode = getattr(cfg, "cluster_selection_mode", "automatic")
    if cfg.fixed_k is not None:
        mode = "fixed"

    if mode == "fixed":
        if cfg.fixed_k not in models:
            raise ValueError("fixed_k is invalid for the available observations")
        return int(cfg.fixed_k), "fixed_by_user"

    if mode == "prompt":
        print("\nCluster diagnostics:")
        print(scores[["k", "inertia", "silhouette", "smallest_cluster", "valid_size"]].to_string(index=False))
        print(f"\nDiagnostic graph saved to: {graph_path.resolve()}")
        print(f"Detected elbow: {elbow_k if elbow_k is not None else 'none'}")
        print(f"Best silhouette: k={int(best_row.k)}")
        allowed = sorted(models)
        while True:
            try:
                answer = input(f"Choose the number of clusters {allowed}: ").strip()
            except EOFError as error:
                raise RuntimeError(
                    "Interactive cluster selection needs a terminal. Use "
                    "cluster_selection_mode: automatic or fixed in the YAML."
                ) from error
            try:
                selected = int(answer)
            except ValueError:
                print("Enter one integer from the displayed list.")
                continue
            if selected in models:
                return selected, "prompted_by_user"
            print(f"Choose one of: {allowed}")

    if elbow_k is None:
        return int(best_row.k), "silhouette_no_reliable_elbow"
    elbow_silhouette = float(valid.loc[valid.k.eq(elbow_k), "silhouette"].iloc[0])
    close_enough = (
        elbow_silhouette >= .95 * best_silhouette
        if best_silhouette > 0 else elbow_silhouette >= best_silhouette - .02
    )
    return (
        (elbow_k, "elbow_with_supported_silhouette")
        if close_enough else (int(best_row.k), "best_silhouette")
    )


def fit_nuclear_clusters(frame, cfg, output_dir: Path):
    """Cluster individual nuclear morphologies and predict every retained nucleus."""
    features = [feature for feature in MORPHOLOGY if feature in frame]
    if len(features) != len(MORPHOLOGY):
        missing = sorted(set(MORPHOLOGY) - set(features))
        raise ValueError(f"Nuclear morphology columns are missing: {missing}")
    rng = np.random.default_rng(cfg.random_seed)
    training_idx = (
        rng.choice(len(frame), cfg.maximum_training_nuclei, replace=False)
        if len(frame) > cfg.maximum_training_nuclei else np.arange(len(frame))
    )
    training = frame.iloc[training_idx]
    preprocess = make_pipeline(SimpleImputer(strategy="median"), RobustScaler())
    x = preprocess.fit_transform(training[features])
    # Retain at least two axes and approximately 90% of non-redundant variance.
    full = PCA(random_state=cfg.random_seed).fit(x)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n_components = min(x.shape[1], max(2, int(np.searchsorted(cumulative, .90) + 1)))
    pca = PCA(n_components=n_components, random_state=cfg.random_seed).fit(x)
    xp = pca.transform(x)
    sample_idx = (
        rng.choice(len(xp), cfg.silhouette_sample, replace=False)
        if len(xp) > cfg.silhouette_sample else np.arange(len(xp))
    )
    max_k = min(cfg.k_max, len(sample_idx) - 1, len(training) - 1)
    if max_k < cfg.k_min:
        raise ValueError("Not enough tissue nuclei to evaluate cluster number")

    rows = []
    models = {}
    minimum_size = max(2, math.ceil(cfg.minimum_cluster_fraction * len(training)))
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
    best_silhouette = float(best_row.silhouette)
    graph_path = output_dir / "cluster_selection.png"
    selected_k, selection_method = choose_cluster_count(
        scores, valid, models, cfg, elbow_k, best_row, graph_path
    )
    selected = models[selected_k]

    repeat_ari = []
    for repeat in range(1, max(1, cfg.stability_repeats)):
        labels = MiniBatchKMeans(
            n_clusters=selected_k, random_state=cfg.random_seed + repeat,
            batch_size=2048, n_init=10,
        ).fit_predict(xp)
        repeat_ari.append(adjusted_rand_score(selected.labels_, labels))
    stability = float(np.mean(repeat_ari)) if repeat_ari else 1.0

    # Make label identities and colours deterministic across equivalent runs.
    centroid_order = np.argsort(selected.cluster_centers_[:, 0])
    relabel = np.empty(selected_k, dtype=int)
    relabel[centroid_order] = np.arange(selected_k)
    result = frame.copy()
    all_x = preprocess.transform(frame[features])
    all_xp = pca.transform(all_x)
    result["cluster"] = relabel[selected.predict(all_xp)]
    _plot_selection(
        scores, selected_k, graph_path, elbow_k=elbow_k
    )
    return result, {
        "selected_k": selected_k,
        "selection_method": selection_method,
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
        "nuclei": len(frame),
        "training_nuclei": len(training),
        "nuclei_for_silhouette": len(sample_idx),
        "features": features,
    }


# Backwards-compatible import name for callers of the earlier simplified build.
fit_clusters = fit_nuclear_clusters
