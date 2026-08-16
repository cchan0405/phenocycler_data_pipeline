from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


BASE_FEATURES = [
    "area",
    "perimeter",
    "circularity",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "aspect_ratio",
    "mean_dapi",
]


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def _architecture(group: pd.DataFrame, effective_tissue_area: float) -> dict[str, float]:
    coords = group[["x", "y"]].to_numpy(dtype=float)
    if len(coords) >= 2:
        distances = cKDTree(coords).query(coords, k=2)[0][:, 1]
        expected_spacing = np.sqrt(effective_tissue_area / len(coords))
        normalized = distances / expected_spacing if expected_spacing > 0 else distances
        nn_median = float(np.median(normalized))
        nn_iqr = float(np.quantile(normalized, 0.75) - np.quantile(normalized, 0.25))
        nn_cv = float(np.std(distances) / np.mean(distances)) if np.mean(distances) else 0.0
    else:
        nn_median = nn_iqr = nn_cv = np.nan
    if "orientation" in group:
        angles = group["orientation"].dropna().to_numpy(dtype=float)
        alignment = (
            float(np.hypot(np.mean(np.cos(2 * angles)), np.mean(np.sin(2 * angles))))
            if len(angles)
            else np.nan
        )
    else:
        alignment = np.nan
    return {
        "architecture_normalized_nn_median": nn_median,
        "architecture_normalized_nn_iqr": nn_iqr,
        "architecture_nn_cv": nn_cv,
        "architecture_orientation_alignment": alignment,
    }


def aggregate_grid_features(
    nuclei: pd.DataFrame,
    *,
    grid_size: int,
    min_nuclei: int,
    tissue_fraction: dict[tuple[int, int], float],
    min_tissue_fraction: float,
) -> pd.DataFrame:
    df = nuclei.copy()
    df["grid_row"] = (df["y"] // grid_size).astype(int)
    df["grid_col"] = (df["x"] // grid_size).astype(int)
    rows: list[dict] = []
    for (gr, gc), group in df.groupby(["grid_row", "grid_col"], sort=True):
        tf = float(tissue_fraction.get((int(gr), int(gc)), 0.0))
        if len(group) < min_nuclei or tf < min_tissue_fraction:
            continue
        effective_tissue_area = max(float(grid_size * grid_size) * tf, 1.0)
        row = {
            "grid_row": int(gr),
            "grid_col": int(gc),
            "x0": int(gc * grid_size),
            "y0": int(gr * grid_size),
            "n_nuclei": int(len(group)),
            "nuclear_density_px2": float(len(group) / effective_tissue_area),
            "log_nuclear_density": float(np.log1p(len(group) / effective_tissue_area)),
            "tissue_fraction": tf,
        }
        row.update(_architecture(group, effective_tissue_area))
        for feature in BASE_FEATURES:
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{feature}_median"] = float(values.median())
            row[f"{feature}_iqr"] = _iqr(values)
            row[f"{feature}_q10"] = float(values.quantile(0.1))
            row[f"{feature}_q90"] = float(values.quantile(0.9))
        if "nuclear_phenotype" in group:
            proportions = group["nuclear_phenotype"].value_counts(normalize=True)
            n_phenotypes = int(df["nuclear_phenotype"].max()) + 1
            for phenotype in range(n_phenotypes):
                row[f"phenotype_{phenotype}_proportion"] = float(proportions.get(phenotype, 0.0))
            p = proportions.to_numpy(dtype=float)
            row["phenotype_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def model_feature_columns(
    grid_df: pd.DataFrame,
    *,
    density_mode: str = "controlled",
    include_dapi_intensity: bool = False,
) -> list[str]:
    excluded = {
        "grid_row",
        "grid_col",
        "x0",
        "y0",
        "cluster",
        "tissue_fraction",
        "n_nuclei",
        "nuclear_density_px2",
        "distance_to_centroid",
    }
    if density_mode == "exclude":
        excluded.add("log_nuclear_density")
    if not include_dapi_intensity:
        excluded.update(c for c in grid_df.columns if c.startswith("mean_dapi_"))
    return [c for c in grid_df.columns if c not in excluded]
