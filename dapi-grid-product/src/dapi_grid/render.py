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
) -> None:
    preview = robust_norm(dapi_preview)
    gh = int(np.ceil(level0_shape[0] / grid_size))
    gw = int(np.ceil(level0_shape[1] / grid_size))
    grid = np.full((gh, gw), np.nan, dtype=float)
    for row in grid_df.itertuples():
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
    overlay = ax.imshow(
        grid,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        extent=[0, level0_shape[1], level0_shape[0], 0],
        interpolation="nearest",
    )
    if draw_grid:
        ax.set_xticks(np.arange(0, level0_shape[1] + grid_size, grid_size), minor=True)
        ax.set_yticks(np.arange(0, level0_shape[0] + grid_size, grid_size), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.15, alpha=0.3)
    cbar = fig.colorbar(overlay, ax=ax, ticks=np.arange(n_clusters), fraction=0.03)
    cbar.set_label("Unsupervised nuclear-shape grid cluster")
    ax.set_title("Whole-tissue DAPI nuclear-shape grid")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray((preview * 255).astype(np.uint8)).save(
        output_path.with_name("dapi_preview.png")
    )

