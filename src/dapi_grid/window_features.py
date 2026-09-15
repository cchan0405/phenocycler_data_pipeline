from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


MORPHOLOGY = [
    "area", "perimeter", "circularity", "eccentricity", "solidity",
    "major_axis_length", "minor_axis_length", "aspect_ratio",
]


def aggregate_windows(nuclei, *, image_shape, window_size, stride):
    """Summarise nuclear morphology in overlapping, non-empty windows."""
    tree = cKDTree(nuclei[["x", "y"]].to_numpy(float))
    half = window_size / 2
    radius = np.sqrt(2) * half
    rows = []
    n_rows = int(np.ceil(image_shape[0] / stride))
    n_cols = int(np.ceil(image_shape[1] / stride))
    for gr in range(n_rows):
        for gc in range(n_cols):
            cy = min(image_shape[0] - 1, (gr + 0.5) * stride)
            cx = min(image_shape[1] - 1, (gc + 0.5) * stride)
            idx = tree.query_ball_point([cx, cy], radius)
            if not idx:
                continue
            group = nuclei.iloc[idx]
            group = group.loc[
                (group.x >= cx - half) & (group.x < cx + half)
                & (group.y >= cy - half) & (group.y < cy + half)
            ]
            if group.empty:
                continue
            row = {
                "grid_row": gr, "grid_col": gc, "x0": gc * stride,
                "y0": gr * stride, "n_nuclei": len(group),
            }
            for feature in MORPHOLOGY:
                values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{feature}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{feature}_iqr"] = (
                    float(values.quantile(.75) - values.quantile(.25))
                    if len(values) else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def model_columns(frame):
    excluded = {
        "grid_row", "grid_col", "x0", "y0", "n_nuclei", "cluster",
        "local_cluster", "sample", "group",
    }
    return [column for column in frame.columns if column not in excluded]
