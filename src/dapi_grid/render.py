from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

from .image_ops import robust_norm


COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#6A3D9A",
    "#B15928",
    "#1B9E77",
]


def render_overlay(
    dapi_preview: np.ndarray,
    grid_df,
    *,
    level0_shape: tuple[int, int],
    grid_size: int,
    alpha: float,
    draw_grid: bool,
    output_path: Path,
    confidence_weighted: bool = False,
    minimum_confidence: float = 0.0,
) -> None:
    preview = robust_norm(dapi_preview)
    gh = int(np.ceil(level0_shape[0] / grid_size))
    gw = int(np.ceil(level0_shape[1] / grid_size))
    grid = np.full((gh, gw), np.nan, dtype=float)
    for row in grid_df.itertuples():
        confidence = float(getattr(row, "overall_confidence", 1.0))
        if confidence >= minimum_confidence:
            grid[int(row.grid_row), int(row.grid_col)] = int(row.cluster)
    n_clusters = int(grid_df["cluster"].max()) + 1
    cmap = ListedColormap(COLORS[:n_clusters])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5), n_clusters)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(
        preview,
        cmap="gray",
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    rgba = cmap(norm(grid))
    rgba[..., 3] = np.where(np.isnan(grid), 0.0, alpha)
    if confidence_weighted and "overall_confidence" in grid_df:
        confidence_grid = np.zeros((gh, gw), dtype=float)
        for row in grid_df.itertuples():
            confidence_grid[int(row.grid_row), int(row.grid_col)] = float(
                row.overall_confidence
            )
        rgba[..., 3] *= confidence_grid
    ax.imshow(
        rgba,
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    if draw_grid:
        ax.set_xticks(np.arange(0, level0_shape[1] + grid_size, grid_size), minor=True)
        ax.set_yticks(np.arange(0, level0_shape[0] + grid_size, grid_size), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.15, alpha=0.3)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        scalar_map, ax=ax, ticks=np.arange(n_clusters), fraction=0.03
    )
    cbar.set_label("Unsupervised nuclear-shape grid cluster")
    ax.set_title("Whole-tissue DAPI nuclear-shape grid")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray((preview * 255).astype(np.uint8)).save(
        output_path.with_name("dapi_preview.png")
    )


def render_confidence_overlay(
    dapi_preview,
    grid_df,
    *,
    level0_shape,
    grid_size,
    output_path: Path,
) -> None:
    preview = robust_norm(dapi_preview)
    gh = int(np.ceil(level0_shape[0] / grid_size))
    gw = int(np.ceil(level0_shape[1] / grid_size))
    values = np.full((gh, gw), np.nan, dtype=float)
    for row in grid_df.itertuples():
        values[int(row.grid_row), int(row.grid_col)] = float(row.overall_confidence)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(
        preview,
        cmap="gray",
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    overlay = ax.imshow(
        values,
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.75,
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    fig.colorbar(overlay, ax=ax, fraction=0.03, label="Overall prediction confidence")
    ax.set_title("Whole-tissue prediction confidence")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_status_overlay(
    dapi_preview,
    grid_df,
    *,
    level0_shape,
    grid_size,
    output_path: Path,
) -> None:
    preview = robust_norm(dapi_preview)
    gh = int(np.ceil(level0_shape[0] / grid_size))
    gw = int(np.ceil(level0_shape[1] / grid_size))
    status_codes = {"low_confidence": 0, "predicted": 1, "training_quality": 2}
    values = np.full((gh, gw), np.nan, dtype=float)
    for row in grid_df.itertuples():
        values[int(row.grid_row), int(row.grid_col)] = status_codes[row.prediction_status]
    cmap = ListedColormap(["#D55E00", "#F0E442", "#009E73"])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-0.5, 3.5), 3)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(
        preview,
        cmap="gray",
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    overlay = ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        alpha=0.65,
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    cbar = fig.colorbar(overlay, ax=ax, ticks=[0, 1, 2], fraction=0.03)
    cbar.ax.set_yticklabels(["low confidence", "predicted", "training quality"])
    ax.set_title("Training-quality versus predicted windows")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
