from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .grid_features import model_feature_columns


def fit_grid_clusters(grid_df: pd.DataFrame, cfg, output_dir: Path):
    features = model_feature_columns(grid_df)
    if len(grid_df) < 3:
        raise ValueError("At least three QC-passed grid squares are required.")
    preprocess = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
        ]
    )
    x = preprocess.fit_transform(grid_df[features])
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
        score = silhouette_score(xp[score_idx], labels[score_idx])
        scores.append({"k": int(k), "silhouette": float(score)})
        if score > best_score:
            best_score, best_model = score, model

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
    }
    joblib.dump(artifact, output_dir / "grid_cluster_model.joblib")
    (output_dir / "cluster_selection.json").write_text(
        json.dumps(
            {
                "selected_k": int(best_model.n_clusters),
                "selected_silhouette": float(best_score),
                "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                "candidates": scores,
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result

