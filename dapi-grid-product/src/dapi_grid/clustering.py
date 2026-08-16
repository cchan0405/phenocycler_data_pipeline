from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .grid_features import model_feature_columns


def fit_grid_clusters(grid_df: pd.DataFrame, cfg, output_dir: Path):
    features = model_feature_columns(
        grid_df,
        density_mode=cfg.density_mode,
        include_dapi_intensity=cfg.include_dapi_intensity,
    )
    if len(grid_df) < 3:
        raise ValueError("At least three QC-passed grid squares are required.")
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    x = preprocess.fit_transform(grid_df[features])
    # Give correlated feature families comparable total influence. Density is a
    # useful context variable, but should not overwhelm morphology.
    families = {
        "density": [i for i, f in enumerate(features) if f == "log_nuclear_density"],
        "phenotype": [i for i, f in enumerate(features) if f.startswith("phenotype_")],
        "architecture": [i for i, f in enumerate(features) if f.startswith("architecture_")],
    }
    claimed = {i for indices in families.values() for i in indices}
    families["morphology"] = [i for i in range(len(features)) if i not in claimed]
    family_weights = {}
    for name, indices in families.items():
        if not indices:
            continue
        weight = 1.0 / np.sqrt(len(indices))
        if name == "density":
            weight *= cfg.density_weight if cfg.density_mode == "controlled" else 1.0
        x[:, indices] *= weight
        family_weights[name] = float(weight)
    n_pca = min(15, x.shape[1], max(2, x.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed)
    xp = pca.fit_transform(x)

    max_k = min(cfg.k_max, len(grid_df) - 1)
    candidates = [cfg.fixed_k] if cfg.fixed_k is not None else list(range(cfg.k_min, max_k + 1))
    scores: list[dict] = []
    best_model = None
    best_score = -np.inf
    rng = np.random.default_rng(cfg.random_seed)
    if len(xp) > cfg.silhouette_sample:
        score_idx = rng.choice(len(xp), cfg.silhouette_sample, replace=False)
    else:
        score_idx = np.arange(len(xp))

    for k in candidates:
        if k is None or k < 2 or k >= len(grid_df):
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.random_seed,
            batch_size=cfg.minibatch_size,
            n_init=10,
        )
        labels = model.fit_predict(xp)
        silhouette = silhouette_score(xp[score_idx], labels[score_idx])
        repeat_scores = []
        for repeat in range(1, max(1, cfg.stability_repeats)):
            repeated = MiniBatchKMeans(
                n_clusters=k,
                random_state=cfg.random_seed + repeat,
                batch_size=cfg.minibatch_size,
                n_init=5,
            ).fit_predict(xp)
            repeat_scores.append(adjusted_rand_score(labels, repeated))
        stability = float(np.mean(repeat_scores)) if repeat_scores else 1.0
        selection_score = (
            float(silhouette)
            + cfg.stability_weight * stability
            + cfg.complexity_weight * np.log2(k)
        )
        scores.append(
            {
                "k": int(k),
                "silhouette": float(silhouette),
                "stability_ari": stability,
                "selection_score": selection_score,
            }
        )
        if selection_score > best_score:
            best_score, best_model = selection_score, model

    if best_model is None:
        raise ValueError("No valid cluster count. Lower k_min or provide more valid grids.")
    result = grid_df.copy()
    result["cluster"] = best_model.predict(xp).astype(int)
    result["distance_to_centroid"] = np.min(best_model.transform(xp), axis=1)
    artifact = {
        "features": features,
        "preprocess": preprocess,
        "pca": pca,
        "cluster_model": best_model,
        "family_weights": family_weights,
    }
    joblib.dump(artifact, output_dir / "grid_cluster_model.joblib")
    (output_dir / "cluster_selection.json").write_text(
        json.dumps(
            {
                "selected_k": int(best_model.n_clusters),
                "selected_score": float(best_score),
                "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                "candidates": scores,
                "features": features,
                "feature_family_weights": family_weights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
