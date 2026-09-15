from __future__ import annotations

import numpy as np
import pandas as pd


SHAPE_PROPERTIES = [
    "label", "area", "perimeter", "centroid", "eccentricity", "solidity",
    "major_axis_length", "minor_axis_length", "orientation",
]


def segment_and_measure(image, model, *, prob_thresh, nms_thresh, n_tiles):
    """Return nucleus measurements and StarDist's actual boundary polygons."""
    from csbdeep.utils import normalize
    from skimage.measure import regionprops_table

    norm = normalize(image.astype(np.float32), 1, 99.8, axis=(0, 1))
    labels, details = model.predict_instances(
        norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=n_tiles
    )
    props = pd.DataFrame(regionprops_table(labels, properties=SHAPE_PROPERTIES))
    props = props.rename(columns={"centroid-0": "local_y", "centroid-1": "local_x"})
    if props.empty:
        return props, np.empty((0, 2, 0), dtype=np.float32)
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
    polygons = np.asarray(details.get("coord", []), dtype=np.float32)
    if len(polygons) != len(props):
        raise RuntimeError("StarDist polygon and label counts do not match")
    return props, polygons


def keep_owned_nuclei(df: pd.DataFrame, polygons: np.ndarray, chunk):
    """Keep detections whose centres belong to the non-overlapping chunk core."""
    if df.empty:
        return df, polygons
    out = df.copy()
    out["y"] = out["local_y"] + chunk.read_y0
    out["x"] = out["local_x"] + chunk.read_x0
    owned = (
        (out["y"] >= chunk.core_y0) & (out["y"] < chunk.core_y1)
        & (out["x"] >= chunk.core_x0) & (out["x"] < chunk.core_x1)
    ).to_numpy()
    kept_polygons = polygons[owned].copy()
    if kept_polygons.size:
        kept_polygons[:, 0, :] += chunk.read_y0
        kept_polygons[:, 1, :] += chunk.read_x0
    out = out.loc[owned].copy()
    out["chunk_id"] = chunk.chunk_id
    return out.drop(columns=["local_y", "local_x"], errors="ignore"), kept_polygons
