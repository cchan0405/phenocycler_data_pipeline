from __future__ import annotations

import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects


def robust_norm(img: np.ndarray, p_low: float = 1, p_high: float = 99.8) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def tissue_mask(
    dapi_low: np.ndarray,
    *,
    gaussian_sigma: float,
    min_object_area: int,
    closing_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    norm = robust_norm(dapi_low)
    smooth = gaussian(norm, sigma=gaussian_sigma, preserve_range=True)
    valid = smooth[smooth > 0]
    if valid.size < 10:
        return norm, np.zeros_like(norm, dtype=bool)
    threshold = threshold_otsu(valid)
    mask = smooth > threshold
    mask = remove_small_objects(mask, min_size=min_object_area)
    mask = binary_closing(mask, footprint=disk(closing_radius))
    return norm, mask


def lowres_fraction_for_level0_box(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    level0_shape: tuple[int, int],
) -> float:
    y0, x0, y1, x1 = box
    sy = mask.shape[0] / level0_shape[0]
    sx = mask.shape[1] / level0_shape[1]
    ly0, lx0 = max(0, int(y0 * sy)), max(0, int(x0 * sx))
    ly1 = min(mask.shape[0], max(ly0 + 1, int(np.ceil(y1 * sy))))
    lx1 = min(mask.shape[1], max(lx0 + 1, int(np.ceil(x1 * sx))))
    return float(mask[ly0:ly1, lx0:lx1].mean())

