from __future__ import annotations

import numpy as np
import pandas as pd


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
        row = {
            "grid_row": int(gr),
            "grid_col": int(gc),
            "x0": int(gc * grid_size),
            "y0": int(gr * grid_size),
            "n_nuclei": int(len(group)),
            "nuclear_density_px2": float(len(group) / (grid_size * grid_size)),
            "tissue_fraction": tf,
        }
        for feature in BASE_FEATURES:
            values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_std"] = float(values.std(ddof=0))
            row[f"{feature}_median"] = float(values.median())
            row[f"{feature}_q90"] = float(values.quantile(0.9))
        rows.append(row)
    return pd.DataFrame(rows)


def model_feature_columns(grid_df: pd.DataFrame) -> list[str]:
    excluded = {
        "grid_row",
        "grid_col",
        "x0",
        "y0",
        "cluster",
        "tissue_fraction",
    }
    return [c for c in grid_df.columns if c not in excluded]

