from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .grid_features import BASE_FEATURES


def _safe_summary(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"median": np.nan, "iqr": np.nan, "q10": np.nan, "q90": np.nan}
    return {
        "median": float(clean.median()),
        "iqr": float(clean.quantile(0.75) - clean.quantile(0.25)),
        "q10": float(clean.quantile(0.10)),
        "q90": float(clean.quantile(0.90)),
    }


def _architecture(group: pd.DataFrame, effective_area: float) -> dict[str, float]:
    coords = group[["x", "y"]].to_numpy(dtype=float)
    if len(coords) >= 2:
        distances = cKDTree(coords).query(coords, k=2)[0][:, 1]
        expected = np.sqrt(effective_area / len(coords))
        normalized = distances / expected if expected > 0 else distances
        nn_median = float(np.median(normalized))
        nn_iqr = float(np.quantile(normalized, 0.75) - np.quantile(normalized, 0.25))
        nn_cv = float(np.std(distances) / np.mean(distances)) if np.mean(distances) else 0.0
    else:
        nn_median = nn_iqr = nn_cv = np.nan
    angles = group.get("orientation", pd.Series(dtype=float)).dropna().to_numpy(float)
    alignment = (
        float(np.hypot(np.mean(np.cos(2 * angles)), np.mean(np.sin(2 * angles))))
        if len(angles) >= 2
        else np.nan
    )
    return {
        "architecture_normalized_nn_median": nn_median,
        "architecture_normalized_nn_iqr": nn_iqr,
        "architecture_nn_cv": nn_cv,
        "architecture_orientation_alignment": alignment,
        "architecture_available": float(len(coords) >= 2),
    }


def aggregate_overlapping_windows(
    nuclei: pd.DataFrame,
    *,
    level0_shape: tuple[int, int],
    analysis_window_px: int,
    display_stride_px: int,
    display_tissue_fraction: dict[tuple[int, int], float],
    analysis_tissue_fraction: dict[tuple[int, int], float],
    min_nuclei_predict: int,
    min_tissue_fraction_predict: float,
    phenotype_prior: float = 0.5,
) -> pd.DataFrame:
    """Summarise overlapping neighbourhoods centred on display tiles."""
    coords = nuclei[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(coords)
    radius = np.sqrt(2.0) * analysis_window_px / 2.0
    half = analysis_window_px / 2.0
    n_phenotypes = (
        int(nuclei["nuclear_phenotype"].max()) + 1
        if "nuclear_phenotype" in nuclei
        else 0
    )
    rows: list[dict] = []
    for (gr, gc), display_tf in display_tissue_fraction.items():
        if display_tf < min_tissue_fraction_predict:
            continue
        cy = min(level0_shape[0] - 1, (gr + 0.5) * display_stride_px)
        cx = min(level0_shape[1] - 1, (gc + 0.5) * display_stride_px)
        candidate = tree.query_ball_point([cx, cy], radius)
        if not candidate:
            continue
        group = nuclei.iloc[candidate]
        inside = (
            (group["x"] >= cx - half)
            & (group["x"] < cx + half)
            & (group["y"] >= cy - half)
            & (group["y"] < cy + half)
        )
        group = group.loc[inside]
        if len(group) < min_nuclei_predict:
            continue
        analysis_tf = float(analysis_tissue_fraction.get((gr, gc), display_tf))
        effective_area = max(analysis_window_px**2 * analysis_tf, 1.0)
        row: dict[str, float | int] = {
            "grid_row": int(gr),
            "grid_col": int(gc),
            "x0": int(gc * display_stride_px),
            "y0": int(gr * display_stride_px),
            "analysis_center_x": float(cx),
            "analysis_center_y": float(cy),
            "n_nuclei": int(len(group)),
            "nuclear_density_px2": float(len(group) / effective_area),
            "log_nuclear_density": float(np.log1p(len(group) / effective_area)),
            "tissue_fraction": float(display_tf),
            "analysis_tissue_fraction": analysis_tf,
        }
        row.update(_architecture(group, effective_area))
        for feature in BASE_FEATURES:
            for statistic, value in _safe_summary(group[feature]).items():
                row[f"{feature}_{statistic}"] = value
        if n_phenotypes:
            counts = group["nuclear_phenotype"].value_counts()
            denominator = len(group) + phenotype_prior * n_phenotypes
            proportions = []
            for phenotype in range(n_phenotypes):
                p = float((counts.get(phenotype, 0) + phenotype_prior) / denominator)
                row[f"phenotype_{phenotype}_proportion"] = p
                proportions.append(p)
            p = np.asarray(proportions)
            row["phenotype_entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        rows.append(row)
    return pd.DataFrame(rows)
