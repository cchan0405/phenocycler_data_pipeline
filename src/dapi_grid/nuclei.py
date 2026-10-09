from __future__ import annotations

import numpy as np
import pandas as pd
import re

from .image_ops import points_in_mask


SHAPE_PROPERTIES = [
    "label", "area", "perimeter", "centroid", "eccentricity", "solidity",
    "major_axis_length", "minor_axis_length", "orientation",
]


def zernike_indices(degree):
    """Return Mahotas' non-redundant (radial degree, angular order) sequence."""
    return [
        (n, m)
        for n in range(degree + 1)
        for m in range(n + 1)
        if (n - m) % 2 == 0
    ]


def measure_zernike(labels, degree=8):
    """Measure rotation-invariant Zernike magnitudes for every labelled nucleus."""
    import mahotas
    from skimage.measure import regionprops

    indices = zernike_indices(degree)
    columns = [f"zernike_n{n}_m{m}" for n, m in indices]
    rows = []
    for region in regionprops(labels):
        mask = region.image.astype(np.uint8, copy=False)
        coordinates = np.argwhere(mask)
        centre = coordinates.mean(axis=0)
        radius = max(
            1.0,
            float(np.sqrt(np.square(coordinates - centre).sum(axis=1)).max()) + .5,
        )
        values = mahotas.features.zernike_moments(
            mask, radius=radius, degree=degree, cm=tuple(centre)
        )
        if len(values) != len(columns):
            raise RuntimeError(
                f"Mahotas returned {len(values)} Zernike moments; "
                f"expected {len(columns)} for degree {degree}"
            )
        rows.append([region.label, *values])
    return pd.DataFrame(rows, columns=["label", *columns])


def marker_slug(name):
    """Create a stable, CSV-safe marker identifier."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_").lower()


def segment_and_measure(
    image, model, *, prob_thresh, nms_thresh, n_tiles,
    marker_images=None, cell_expansion_px=8, zernike_degree=8,
):
    """Return nucleus measurements and StarDist's actual boundary polygons."""
    from csbdeep.utils import normalize
    from skimage.measure import regionprops_table
    from skimage.segmentation import expand_labels

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
    if zernike_degree is not None:
        zernike = measure_zernike(labels, degree=zernike_degree)
        props = props.merge(zernike, on="label", how="left", validate="one_to_one")
    marker_images = marker_images or []
    expanded = None
    for marker, marker_image in marker_images:
        marker_image = np.asarray(marker_image)
        if marker_image.shape != labels.shape:
            raise ValueError(
                f"Marker {marker.name} shape {marker_image.shape} does not match "
                f"DAPI shape {labels.shape}"
            )
        slug = marker_slug(marker.name)
        compartments = (
            ("nucleus", "cell") if marker.compartment == "both"
            else (marker.compartment,)
        )
        for compartment in compartments:
            if compartment == "cell":
                if expanded is None:
                    expanded = expand_labels(labels, distance=cell_expansion_px)
                measurement_labels = expanded
            else:
                measurement_labels = labels
            intensity = pd.DataFrame(regionprops_table(
                measurement_labels, intensity_image=marker_image,
                properties=("label", "intensity_mean"),
            )).rename(columns={
                "intensity_mean": f"marker__{slug}__{compartment}_mean"
            })
            props = props.merge(intensity, on="label", how="left", validate="one_to_one")
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


def keep_tissue_nuclei(df, polygons, tissue_mask, level0_shape):
    """Discard detections whose centroids lie outside the expanded tissue mask."""
    if df.empty:
        return df, polygons
    keep = points_in_mask(
        df["y"].to_numpy(), df["x"].to_numpy(), tissue_mask, level0_shape
    )
    return df.loc[keep].copy(), polygons[np.asarray(keep, dtype=bool)].copy()
