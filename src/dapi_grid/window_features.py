from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .image_ops import mask_fraction


MORPHOLOGY = [
    "area", "perimeter", "circularity", "eccentricity", "solidity",
    "major_axis_length", "minor_axis_length", "aspect_ratio",
]


def aggregate_windows(
    nuclei, *, image_shape, window_size, stride, tissue_mask=None,
    minimum_tissue_fraction=0.0, mask_rescue_nuclei=1,
    minimum_nuclei=1,
):
    """Summarise morphology and cell-phenotype composition in tissue windows."""
    tree = cKDTree(nuclei[["x", "y"]].to_numpy(float))
    cluster_ids = sorted(nuclei["cluster"].dropna().astype(int).unique()) if "cluster" in nuclei else []
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
            if len(group) < minimum_nuclei:
                continue
            box = (
                max(0, int(cy - half)), max(0, int(cx - half)),
                min(image_shape[0], int(np.ceil(cy + half))),
                min(image_shape[1], int(np.ceil(cx + half))),
            )
            tissue_fraction = (
                mask_fraction(tissue_mask, box, image_shape)
                if tissue_mask is not None else 1.0
            )
            # The mask removes background false detections. A sufficiently dense
            # group rescues genuine tissue missed by the coarse preview mask.
            if (tissue_fraction < minimum_tissue_fraction
                    and len(group) < mask_rescue_nuclei):
                continue
            row = {
                "grid_row": gr, "grid_col": gc, "x0": gc * stride,
                "y0": gr * stride, "n_nuclei": len(group),
                "tissue_fraction": tissue_fraction,
            }
            for feature in MORPHOLOGY:
                values = group[feature].replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{feature}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{feature}_iqr"] = (
                    float(values.quantile(.75) - values.quantile(.25))
                    if len(values) else np.nan
                )
            if cluster_ids:
                counts = group["cluster"].astype(int).value_counts()
                for cluster in cluster_ids:
                    row[f"phenotype_{cluster}_fraction"] = float(
                        counts.get(cluster, 0) / len(group)
                    )
                row["dominant_phenotype"] = int(counts.idxmax())
            rows.append(row)
    return pd.DataFrame(rows)


def model_columns(frame):
    excluded = {
        "grid_row", "grid_col", "x0", "y0", "n_nuclei", "cluster",
        "local_cluster", "sample", "group", "tissue_fraction",
    }
    return [column for column in frame.columns if column not in excluded]
