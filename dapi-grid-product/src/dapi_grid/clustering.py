from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, SplineTransformer

from .grid_features import model_feature_columns


def _residualize_density(x, density, training_mask, features, cfg, output_dir):
    if not cfg.residualize:
        return x, [], []
    if np.nanstd(density[training_mask]) < 1e-12:
        pd.DataFrame(
            [{"feature": "ALL", "density_r2_training": 0.0,
              "note": "Density was constant; residualisation was skipped."}]
        ).to_csv(output_dir / "density_residualization_report.csv", index=False)
        return x, [], []
    corrected = x.copy()
    models, report = [], []
    d_train = density[training_mask].reshape(-1, 1)
    d_all = density.reshape(-1, 1)
    for index, feature in enumerate(features):
        y_train = x[training_mask, index]
        if (
            feature == "log_nuclear_density"
            or feature.endswith("_available")
            or np.nanstd(y_train) < 1e-12
        ):
            models.append(None)
            continue
        model = Pipeline(
            [
                (
                    "spline",
                    SplineTransformer(
                        n_knots=cfg.spline_knots, degree=3, include_bias=False
                    ),
                ),
                ("ridge", Ridge(alpha=cfg.ridge_alpha)),
            ]
        )
        model.fit(d_train, y_train)
        predicted = model.predict(d_all)
        corrected[:, index] = x[:, index] - predicted
        report.append(
            {
                "feature": feature,
                "density_r2_training": float(
                    r2_score(y_train, predicted[training_mask])
                ),
                "observed_std_training": float(np.std(y_train)),
                "residual_std_training": float(
                    np.std(corrected[training_mask, index])
                ),
            }
        )
        models.append(model)
    report_df = pd.DataFrame(report)
    if not report_df.empty:
        report_df = report_df.sort_values("density_r2_training", ascending=False)
    report_df.to_csv(output_dir / "density_residualization_report.csv", index=False)
    return corrected, models, report


def _feature_families(features):
    families = {
        "density": [i for i, f in enumerate(features) if f == "log_nuclear_density"],
        "phenotype": [i for i, f in enumerate(features) if f.startswith("phenotype_")],
        "architecture": [i for i, f in enumerate(features) if f.startswith("architecture_")],
    }
    claimed = {i for indices in families.values() for i in indices}
    families["morphology"] = [i for i in range(len(features)) if i not in claimed]
    return families


def fit_grid_clusters(
    grid_df,
    cfg,
    output_dir: Path,
    *,
    training_cfg=None,
    density_cfg=None,
    confidence_cfg=None,
):
    """Fit clusters on reliable windows and predict every eligible window."""
    features = model_feature_columns(
        grid_df,
        density_mode=cfg.density_mode,
        include_dapi_intensity=cfg.include_dapi_intensity,
    )
    if len(grid_df) < 3:
        raise ValueError("At least three prediction windows are required.")

    if training_cfg is None:
        training_mask = np.ones(len(grid_df), dtype=bool)
        min_nuclei_train, min_tissue_train = 1, 0.0
    else:
        tissue = grid_df.get(
            "analysis_tissue_fraction", grid_df["tissue_fraction"]
        ).to_numpy()
        training_mask = (
            (grid_df["n_nuclei"].to_numpy() >= training_cfg.min_nuclei_train)
            & (tissue >= training_cfg.min_tissue_fraction_train)
        )
        min_nuclei_train = training_cfg.min_nuclei_train
        min_tissue_train = training_cfg.min_tissue_fraction_train
    if training_mask.sum() < max(3, cfg.fixed_k or cfg.k_min):
        raise ValueError(
            "Too few reliable training windows. Lower training.min_nuclei_train "
            "or training.min_tissue_fraction_train."
        )

    feature_imputer = SimpleImputer(strategy="median")
    feature_imputer.fit(grid_df.loc[training_mask, features])
    x = feature_imputer.transform(grid_df[features])
    density = grid_df["log_nuclear_density"].to_numpy(dtype=float)
    if density_cfg is not None:
        x, density_models, density_report = _residualize_density(
            x, density, training_mask, features, density_cfg, output_dir
        )
    else:
        density_models, density_report = [], []

    scaler = RobustScaler().fit(x[training_mask])
    x = scaler.transform(x)
    families = _feature_families(features)
    family_weights = {}
    for name, indices in families.items():
        if not indices:
            continue
        weight = 1.0 / np.sqrt(len(indices))
        if name == "density":
            weight *= cfg.density_weight if cfg.density_mode == "controlled" else 1.0
        x[:, indices] *= weight
        family_weights[name] = float(weight)

    x_train = x[training_mask]
    n_pca = min(15, x_train.shape[1], max(2, x_train.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=cfg.random_seed).fit(x_train)
    xp = pca.transform(x)
    xp_train = xp[training_mask]

    max_k = min(cfg.k_max, len(xp_train) - 1)
    candidates = (
        [cfg.fixed_k]
        if cfg.fixed_k is not None
        else list(range(cfg.k_min, max_k + 1))
    )
    scores, best_model, best_score = [], None, -np.inf
    rng = np.random.default_rng(cfg.random_seed)
    score_idx = (
        rng.choice(len(xp_train), cfg.silhouette_sample, replace=False)
        if len(xp_train) > cfg.silhouette_sample
        else np.arange(len(xp_train))
    )
    for k in candidates:
        if k is None or k < 2 or k >= len(xp_train):
            continue
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=cfg.random_seed,
            batch_size=cfg.minibatch_size,
            n_init=10,
        )
        labels = model.fit_predict(xp_train)
        sampled = labels[score_idx]
        silhouette = (
            silhouette_score(xp_train[score_idx], sampled)
            if 1 < len(np.unique(sampled)) < len(sampled)
            else -1.0
        )
        repeat_scores = []
        for repeat in range(1, max(1, cfg.stability_repeats)):
            repeated = MiniBatchKMeans(
                n_clusters=k,
                random_state=cfg.random_seed + repeat,
                batch_size=cfg.minibatch_size,
                n_init=5,
            ).fit_predict(xp_train)
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
                "selection_score": float(selection_score),
            }
        )
        if selection_score > best_score:
            best_score, best_model = selection_score, model
    if best_model is None:
        raise ValueError("No valid cluster count for the reliable training windows.")

    labels_all = best_model.predict(xp).astype(int)
    distances = best_model.transform(xp)
    ordered = np.sort(distances, axis=1)
    nearest, second = ordered[:, 0], ordered[:, 1]
    cluster_confidence = np.clip(
        1.0 - nearest / np.maximum(second, 1e-12), 0.0, 1.0
    )
    sampling_confidence = np.clip(
        grid_df["n_nuclei"].to_numpy(float) / max(1, min_nuclei_train), 0.0, 1.0
    )
    overall_confidence = np.sqrt(cluster_confidence * sampling_confidence)
    threshold = confidence_cfg.low_confidence_threshold if confidence_cfg else 0.40

    result = grid_df.copy()
    result["cluster"] = labels_all
    result["distance_to_centroid"] = nearest
    result["sampling_confidence"] = sampling_confidence
    result["cluster_confidence"] = cluster_confidence
    result["overall_confidence"] = overall_confidence
    result["is_training_window"] = training_mask
    result["prediction_status"] = np.where(
        overall_confidence < threshold,
        "low_confidence",
        np.where(training_mask, "training_quality", "predicted"),
    )

    result.loc[training_mask].to_csv(output_dir / "training_windows.csv", index=False)
    result.loc[~training_mask].to_csv(output_dir / "predicted_windows.csv", index=False)
    (
        result.groupby("cluster")
        .agg(
            windows=("cluster", "size"),
            training_windows=("is_training_window", "sum"),
            median_nuclei=("n_nuclei", "median"),
            median_confidence=("overall_confidence", "median"),
            low_confidence_windows=(
                "prediction_status", lambda s: int((s == "low_confidence").sum())
            ),
        )
        .reset_index()
        .to_csv(output_dir / "cluster_summary.csv", index=False)
    )
    (
        result.sort_values(["cluster", "distance_to_centroid"])
        .groupby("cluster", as_index=False)
        .head(12)
        .to_csv(output_dir / "representative_windows.csv", index=False)
    )
    profile_rows = []
    for feature in features:
        overall_median = float(result[feature].median())
        overall_iqr = float(result[feature].quantile(0.75) - result[feature].quantile(0.25))
        for cluster, group in result.groupby("cluster"):
            cluster_median = float(group[feature].median())
            profile_rows.append(
                {
                    "cluster": int(cluster),
                    "feature": feature,
                    "cluster_median": cluster_median,
                    "overall_median": overall_median,
                    "difference_in_overall_iqr": (
                        (cluster_median - overall_median) / overall_iqr
                        if overall_iqr > 0
                        else 0.0
                    ),
                }
            )
    pd.DataFrame(profile_rows).to_csv(
        output_dir / "cluster_feature_profiles.csv", index=False
    )

    artifact = {
        "features": features,
        "feature_imputer": feature_imputer,
        "density_models": density_models,
        "density_correction": density_cfg,
        "scaler": scaler,
        "pca": pca,
        "cluster_model": best_model,
        "family_weights": family_weights,
        "training_min_nuclei": min_nuclei_train,
        "training_min_tissue_fraction": min_tissue_train,
    }
    joblib.dump(artifact, output_dir / "grid_cluster_model.joblib")
    (output_dir / "cluster_selection.json").write_text(
        json.dumps(
            {
                "selected_k": int(best_model.n_clusters),
                "selected_score": float(best_score),
                "training_windows": int(training_mask.sum()),
                "prediction_windows": int((~training_mask).sum()),
                "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                "candidates": scores,
                "features": features,
                "feature_family_weights": family_weights,
                "density_residualization_enabled": bool(
                    density_cfg is not None and density_cfg.residualize
                ),
                "density_residualization_features": len(density_report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result
