from __future__ import annotations

import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.measure import regionprops_table


SHAPE_PROPERTIES = [
    "label",
    "area",
    "perimeter",
    "centroid",
    "eccentricity",
    "solidity",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "mean_intensity",
    "max_intensity",
]


def segment_and_measure(
    image: np.ndarray,
    model,
    *,
    prob_thresh: float,
    nms_thresh: float,
    n_tiles: tuple[int, int],
) -> pd.DataFrame:
    norm = normalize(image.astype(np.float32), 1, 99.8, axis=(0, 1))
    labels, _ = model.predict_instances(
        norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=n_tiles
    )
    props = pd.DataFrame(
        regionprops_table(labels, intensity_image=norm, properties=SHAPE_PROPERTIES)
    )
    if props.empty:
        return props
    props = props.rename(
        columns={
            "centroid-0": "local_y",
            "centroid-1": "local_x",
            "mean_intensity": "mean_dapi",
            "max_intensity": "max_dapi",
        }
    )
    props["circularity"] = np.where(
        props["perimeter"] > 0,
        4 * np.pi * props["area"] / np.square(props["perimeter"]),
        np.nan,
    )
    props["aspect_ratio"] = np.where(
        props["minor_axis_length"] > 0,
        props["major_axis_length"] / props["minor_axis_length"],
        np.nan,
    )
    return props


def keep_owned_qc_nuclei(df: pd.DataFrame, chunk, qc) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["y"] = out["local_y"] + chunk.read_y0
    out["x"] = out["local_x"] + chunk.read_x0
    owned = (
        (out["y"] >= chunk.core_y0)
        & (out["y"] < chunk.core_y1)
        & (out["x"] >= chunk.core_x0)
        & (out["x"] < chunk.core_x1)
    )
    valid = (
        (out["area"] >= qc.min_area_px)
        & (out["area"] <= qc.max_area_px)
        & (out["solidity"] >= qc.min_solidity)
        & (out["max_dapi"] < qc.max_dapi_normalized)
    )
    out = out.loc[owned & valid].copy()
    out["chunk_id"] = chunk.chunk_id
    return out.drop(columns=["local_y", "local_x"], errors="ignore")

