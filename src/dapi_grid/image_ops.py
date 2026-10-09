from __future__ import annotations

import numpy as np


def make_tissue_mask(
    image, *, sigma=1.0, min_object_area=200, closing_radius=2,
    expansion_radius=0,
):
    """Create a conservative low-resolution DAPI tissue-support mask."""
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects

    image = np.asarray(image, dtype=np.float32)
    valid = image[np.isfinite(image)]
    if valid.size < 10:
        return np.zeros(image.shape, dtype=bool)
    low, high = np.percentile(valid, [1, 99.8])
    normalized = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
    smooth = gaussian(normalized, sigma=sigma, preserve_range=True)
    positive = smooth[smooth > 0]
    if positive.size < 10:
        return np.zeros(image.shape, dtype=bool)
    mask = smooth > threshold_otsu(positive)
    mask = remove_small_objects(mask, min_size=min_object_area)
    mask = binary_closing(mask, footprint=disk(closing_radius))
    if expansion_radius:
        mask = binary_dilation(mask, footprint=disk(expansion_radius))
    return mask


def points_in_mask(y, x, mask, level0_shape):
    """Return whether level-0 point coordinates fall inside a preview mask."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    my = np.clip((y * mask.shape[0] / level0_shape[0]).astype(int), 0, mask.shape[0] - 1)
    mx = np.clip((x * mask.shape[1] / level0_shape[1]).astype(int), 0, mask.shape[1] - 1)
    return mask[my, mx]


def mask_fraction(mask, box, level0_shape):
    """Map a level-0 (y0, x0, y1, x1) box onto the preview mask."""
    y0, x0, y1, x1 = box
    scale_y = mask.shape[0] / level0_shape[0]
    scale_x = mask.shape[1] / level0_shape[1]
    my0, mx0 = max(0, int(y0 * scale_y)), max(0, int(x0 * scale_x))
    my1 = min(mask.shape[0], max(my0 + 1, int(np.ceil(y1 * scale_y))))
    mx1 = min(mask.shape[1], max(mx0 + 1, int(np.ceil(x1 * scale_x))))
    return float(mask[my0:my1, mx0:mx1].mean())


def mapped_region(array, box, level0_shape):
    """Map a level-0 box onto a lower-resolution array and return that region."""
    y0, x0, y1, x1 = box
    scale_y = array.shape[0] / level0_shape[0]
    scale_x = array.shape[1] / level0_shape[1]
    ay0, ax0 = max(0, int(y0 * scale_y)), max(0, int(x0 * scale_x))
    ay1 = min(array.shape[0], max(ay0 + 1, int(np.ceil(y1 * scale_y))))
    ax1 = min(array.shape[1], max(ax0 + 1, int(np.ceil(x1 * scale_x))))
    return np.asarray(array[ay0:ay1, ax0:ax1])


def chunk_mask_statistics(chunk, tissue_mask, preview, level0_shape, threshold):
    """Return interpretable mask and preview-signal statistics for one chunk."""
    core_box = (chunk.core_y0, chunk.core_x0, chunk.core_y1, chunk.core_x1)
    read_box = (chunk.read_y0, chunk.read_x0, chunk.read_y1, chunk.read_x1)
    signal = mapped_region(preview, read_box, level0_shape).astype(np.float64)
    finite = signal[np.isfinite(signal)]
    read_fraction = mask_fraction(tissue_mask, read_box, level0_shape)
    return {
        "chunk_id": chunk.chunk_id,
        "core_y0": chunk.core_y0, "core_x0": chunk.core_x0,
        "core_y1": chunk.core_y1, "core_x1": chunk.core_x1,
        "read_y0": chunk.read_y0, "read_x0": chunk.read_x0,
        "read_y1": chunk.read_y1, "read_x1": chunk.read_x1,
        "core_tissue_fraction": mask_fraction(tissue_mask, core_box, level0_shape),
        "read_tissue_fraction": read_fraction,
        "minimum_chunk_tissue_fraction": threshold,
        "selected_for_segmentation": read_fraction >= threshold,
        "tissue_fraction_margin": read_fraction - threshold,
        "selection_reason": (
            "meets_tissue_fraction_threshold"
            if read_fraction >= threshold else "below_tissue_fraction_threshold"
        ),
        "preview_mean": float(finite.mean()) if finite.size else np.nan,
        "preview_median": float(np.median(finite)) if finite.size else np.nan,
        "preview_std": float(finite.std()) if finite.size else np.nan,
        "preview_min": float(finite.min()) if finite.size else np.nan,
        "preview_max": float(finite.max()) if finite.size else np.nan,
    }
