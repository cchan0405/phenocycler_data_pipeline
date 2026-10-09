from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from PIL import Image, ImageDraw


COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#6A3D9A", "#B15928", "#1B9E77",
]


def _normalize(image, lo, hi):
    image = np.asarray(image, dtype=np.float32)
    return (np.clip((image - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)


def render_cluster_overlay(preview, grid_df, *, image_shape, stride, alpha, path):
    valid = np.asarray(preview, dtype=np.float32)
    lo, hi = np.percentile(valid[np.isfinite(valid)], [1, 99.8])
    background = _normalize(preview, lo, hi)
    rows = int(np.ceil(image_shape[0] / stride))
    cols = int(np.ceil(image_shape[1] / stride))
    grid = np.full((rows, cols), np.nan)
    for item in grid_df.itertuples():
        grid[int(item.grid_row), int(item.grid_col)] = int(item.cluster)
    n_clusters = int(grid_df.cluster.max()) + 1
    cmap = ListedColormap(COLORS[:n_clusters])
    cmap.set_bad((0, 0, 0, 0))
    norm = BoundaryNorm(np.arange(-.5, n_clusters + .5), n_clusters)
    rgba = cmap(norm(grid))
    rgba[..., 3] = np.where(np.isnan(grid), 0, alpha)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(background, cmap="gray", extent=[0, image_shape[1], image_shape[0], 0])
    ax.imshow(rgba, extent=[0, image_shape[1], image_shape[0], 0], interpolation="nearest")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                       ticks=np.arange(n_clusters), fraction=.03)
    bar.set_label("Dominant individual-nucleus phenotype")
    ax.set_title("Spatial summary of dominant nuclear phenotype")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_segmentation_ometiff(
    reader, segment_dir: Path, output_path: Path, *, preview,
    chunk_size: int, outline_width: int = 2, tile_size: int = 512,
    nuclei=None, fill_alpha: float = 0.35,
):
    """Write StarDist masks coloured by each nucleus's own morphology cluster."""
    import tifffile

    level0_shape = reader.level_shape(0)
    n_chunk_cols = int(np.ceil(level0_shape[1] / chunk_size))
    values = np.asarray(preview, dtype=np.float32)
    valid = values[np.isfinite(values)]
    lo, hi = np.percentile(valid, [1, 99.8])
    cluster_lookup = {}
    if nuclei is not None and "cluster" in nuclei:
        cluster_lookup = {
            int(chunk_id): group["cluster"].astype(int).to_numpy()
            for chunk_id, group in nuclei.groupby("chunk_id", sort=False)
        }
    palette = [
        tuple(int(round(channel * 255)) for channel in to_rgb(colour))
        for colour in COLORS
    ]
    fill_opacity = int(round(255 * np.clip(fill_alpha, 0, 1)))

    @lru_cache(maxsize=32)
    def load_polygons(chunk_id):
        path = segment_dir / f"segments_{chunk_id:06d}.npz"
        if not path.exists():
            return np.empty((0, 2, 0), dtype=np.float32)
        with np.load(path) as saved:
            return saved["polygons"]

    def polygons_for_box(y0, x0, y1, x1):
        row0, row1 = int(y0 // chunk_size), int((max(y0, y1 - 1)) // chunk_size)
        col0, col1 = int(x0 // chunk_size), int((max(x0, x1 - 1)) // chunk_size)
        for chunk_row in range(row0, row1 + 1):
            for chunk_col in range(col0, col1 + 1):
                chunk_id = chunk_row * n_chunk_cols + chunk_col
                polygons = load_polygons(chunk_id)
                clusters = cluster_lookup.get(chunk_id)
                if clusters is not None and len(clusters) != len(polygons):
                    raise RuntimeError(
                        f"Chunk {chunk_id}: {len(polygons)} polygons but "
                        f"{len(clusters)} nucleus cluster labels"
                    )
                for index, polygon in enumerate(polygons):
                    py, px = polygon
                    if px.max() >= x0 and px.min() < x1 and py.max() >= y0 and py.min() < y1:
                        yield polygon, (int(clusters[index]) if clusters is not None else None)

    def tiles_for_level(level):
        height, width = reader.level_shape(level)
        scale_y = level0_shape[0] / height
        scale_x = level0_shape[1] / width
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                actual_h, actual_w = min(tile_size, height - y), min(tile_size, width - x)
                dapi = reader.read_region(y=y, x=x, height=actual_h, width=actual_w,
                                          level=level)
                gray = _normalize(dapi, lo, hi)
                rgb = np.repeat(gray[..., None], 3, axis=2)
                canvas = Image.fromarray(rgb).convert("RGBA")
                colour_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(colour_layer, "RGBA")
                box0 = (y * scale_y, x * scale_x,
                        (y + actual_h) * scale_y, (x + actual_w) * scale_x)
                for polygon, cluster in polygons_for_box(*box0):
                    py, px = polygon
                    points = [((float(xx) / scale_x) - x, (float(yy) / scale_y) - y)
                              for yy, xx in zip(py, px)]
                    if points:
                        if cluster is None:
                            draw.line(
                                points + [points[0]], fill=(0, 255, 255, 255),
                                width=max(1, outline_width), joint="curve",
                            )
                        else:
                            colour = palette[cluster % len(palette)]
                            draw.polygon(points, fill=colour + (fill_opacity,))
                            draw.line(
                                points + [points[0]], fill=colour + (255,),
                                width=max(1, outline_width), joint="curve",
                            )
                canvas = Image.alpha_composite(canvas, colour_layer).convert("RGB")
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                tile[:actual_h, :actual_w] = np.asarray(canvas)
                yield tile

    levels = list(range(reader.level_count))
    with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
        for position, level in enumerate(levels):
            height, width = reader.level_shape(level)
            tif.write(
                data=tiles_for_level(level), shape=(height, width, 3), dtype=np.uint8,
                tile=(tile_size, tile_size), photometric="rgb", compression="deflate",
                subifds=len(levels) - 1 if position == 0 else None,
                subfiletype=0 if position == 0 else 1,
                metadata={"axes": "YXS"} if position == 0 else None,
            )
