from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from .clustering import fit_nuclear_clusters
from .config import Config
from .io import SlideReader
from .image_ops import chunk_mask_statistics, make_tissue_mask
from .nuclei import keep_owned_nuclei, keep_tissue_nuclei, segment_and_measure
from .render import render_cluster_overlay, render_segmentation_ometiff
from .tiling import iter_chunks
from .window_features import aggregate_windows


LOG = logging.getLogger("dapi_grid")


def _write_csv(df, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_polygons(polygons, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, polygons=polygons.astype(np.float32, copy=False))
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_pipeline(cfg: Config, *, force=False, rerender_segmentation=False):
    output = cfg.output_dir
    previous_schema = None
    previous_marker_signature = None
    previous_summary = output / "run_summary.json"
    if previous_summary.exists():
        try:
            previous_data = json.loads(
                previous_summary.read_text(encoding="utf-8")
            )
            previous_schema = previous_data.get("feature_schema_version")
            previous_cfg = previous_data.get("configuration", {})
            previous_marker_signature = (
                previous_cfg.get("marker_channels", []),
                previous_cfg.get("cell_expansion_px", 8),
                previous_cfg.get("zernike_degree", 8),
            )
        except (OSError, ValueError):
            pass
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    reader = SlideReader(cfg.input_qptiff, cfg.channel)
    marker_readers = [
        (marker, SlideReader(cfg.input_qptiff, marker.channel))
        for marker in cfg.marker_channels
    ]
    marker_signature = (
        [asdict(marker) for marker in cfg.marker_channels], cfg.cell_expansion_px,
        cfg.zernike_degree,
    )
    reprocess_chunks = (
        force or previous_schema != "4.0"
        or previous_marker_signature != marker_signature
    )
    if reprocess_chunks and not force and previous_summary.exists():
        LOG.info("Cell-feature configuration changed; rebuilding segmentation checkpoints")
    image_shape = reader.level_shape(0)
    preview = reader.read_level(cfg.preview_level)
    tissue_mask = make_tissue_mask(
        preview, sigma=cfg.tissue_mask_gaussian_sigma,
        min_object_area=cfg.tissue_mask_min_object_area,
        closing_radius=cfg.tissue_mask_closing_radius,
        expansion_radius=cfg.tissue_mask_expansion_radius,
    )
    from PIL import Image
    Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(output / "tissue_mask.png")

    model = None
    processed = reused = skipped = 0
    chunk_rows = []
    for chunk in iter_chunks(image_shape, cfg.chunk_size_px, cfg.halo_px):
        csv_path = chunks / f"nuclei_{chunk.chunk_id:06d}.csv"
        segment_path = chunks / f"segments_{chunk.chunk_id:06d}.npz"
        statistics = chunk_mask_statistics(
            chunk, tissue_mask, preview, image_shape,
            cfg.minimum_chunk_tissue_fraction,
        )
        if csv_path.exists() and segment_path.exists() and not reprocess_chunks:
            try:
                statistics["retained_nuclei"] = len(pd.read_csv(csv_path))
            except pd.errors.EmptyDataError:
                statistics["retained_nuclei"] = 0
            statistics["processing_status"] = "reused"
            chunk_rows.append(statistics)
            reused += 1
            continue
        if not statistics["selected_for_segmentation"]:
            _write_csv(pd.DataFrame({"chunk_id": pd.Series(dtype=int)}), csv_path)
            _write_polygons(np.empty((0, 2, 0), dtype=np.float32), segment_path)
            statistics["retained_nuclei"] = 0
            statistics["processing_status"] = "skipped_background"
            chunk_rows.append(statistics)
            skipped += 1
            continue
        if model is None:
            from stardist.models import StarDist2D
            model = StarDist2D.from_pretrained(cfg.stardist.model)
        LOG.info("Segmenting level-0 chunk %d", chunk.chunk_id)
        image = reader.read_region(
            y=chunk.read_y0, x=chunk.read_x0,
            height=chunk.read_y1 - chunk.read_y0,
            width=chunk.read_x1 - chunk.read_x0, level=0,
        )
        marker_images = [
            (marker, marker_reader.read_region(
                y=chunk.read_y0, x=chunk.read_x0,
                height=chunk.read_y1 - chunk.read_y0,
                width=chunk.read_x1 - chunk.read_x0, level=0,
            ))
            for marker, marker_reader in marker_readers
        ]
        measured, polygons = segment_and_measure(
            image, model, prob_thresh=cfg.stardist.prob_thresh,
            nms_thresh=cfg.stardist.nms_thresh, n_tiles=cfg.stardist.n_tiles,
            marker_images=marker_images, cell_expansion_px=cfg.cell_expansion_px,
            zernike_degree=cfg.zernike_degree,
        )
        measured, polygons = keep_owned_nuclei(measured, polygons, chunk)
        measured, polygons = keep_tissue_nuclei(
            measured, polygons, tissue_mask, image_shape
        )
        _write_csv(measured, csv_path)
        _write_polygons(polygons, segment_path)
        statistics["retained_nuclei"] = len(measured)
        statistics["processing_status"] = "processed"
        chunk_rows.append(statistics)
        processed += 1
    _write_csv(pd.DataFrame(chunk_rows), output / "chunk_statistics.csv")
    LOG.info(
        "Chunks processed: %d; skipped as background: %d; reused: %d",
        processed, skipped, reused,
    )

    frames = []
    for path in sorted(chunks.glob("nuclei_*.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if frame.empty or not {"x", "y"}.issubset(frame.columns):
            continue
        segment_path = chunks / path.name.replace("nuclei_", "segments_").replace(".csv", ".npz")
        with np.load(segment_path) as saved:
            polygons = saved["polygons"]
        filtered, filtered_polygons = keep_tissue_nuclei(
            frame, polygons, tissue_mask, image_shape
        )
        if len(filtered) != len(frame):
            _write_csv(filtered, path)
            _write_polygons(filtered_polygons, segment_path)
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        raise RuntimeError("No nuclei were detected")
    nuclei = pd.concat(frames, ignore_index=True)
    nuclei, selection = fit_nuclear_clusters(nuclei, cfg, output)
    _write_csv(nuclei, output / "nuclei.csv")

    windows = aggregate_windows(
        nuclei, image_shape=image_shape, window_size=cfg.window_size_px,
        stride=cfg.stride_px, tissue_mask=tissue_mask,
        minimum_tissue_fraction=cfg.minimum_tissue_fraction,
        mask_rescue_nuclei=cfg.mask_rescue_nuclei,
        minimum_nuclei=cfg.minimum_nuclei_per_cluster_window,
    )
    _write_csv(windows, output / "window_phenotype_composition.csv")
    dominant = windows.rename(columns={"dominant_phenotype": "cluster"})
    render_cluster_overlay(
        preview, dominant, image_shape=image_shape, stride=cfg.stride_px,
        alpha=cfg.overlay_alpha,
        path=output / "spatial_dominant_phenotype_overlay.png",
    )
    segmentation_overlay = output / "nuclear_segmentation_overlay.ome.tif"
    if (force or rerender_segmentation or processed or not segmentation_overlay.exists()
            or previous_schema != "4.0"):
        LOG.info("Writing level-0 pyramidal nuclear-segmentation overlay")
        render_segmentation_ometiff(
            reader, chunks, segmentation_overlay, preview=preview,
            chunk_size=cfg.chunk_size_px,
            outline_width=cfg.segmentation_outline_width,
            nuclei=nuclei,
            fill_alpha=cfg.segmentation_fill_alpha,
        )
    else:
        LOG.info("Reusing existing level-0 nuclear-segmentation overlay")

    summary = {
        "feature_schema_version": "4.0",
        "input_qptiff": str(cfg.input_qptiff),
        "image_shape_level0": list(image_shape),
        "nuclei": int(len(nuclei)),
        "chunks_skipped_as_background": int(skipped),
        "cluster_selection": selection,
        "configuration": asdict(cfg),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
    )
    LOG.info("Complete: %s", output)
    return output
