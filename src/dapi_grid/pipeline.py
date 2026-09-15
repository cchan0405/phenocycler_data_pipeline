from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stardist.models import StarDist2D

from .clustering import fit_clusters
from .config import Config
from .io import SlideReader
from .nuclei import keep_owned_nuclei, segment_and_measure
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


def run_pipeline(cfg: Config, *, force=False):
    output = cfg.output_dir
    chunks = output / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    reader = SlideReader(cfg.input_qptiff, cfg.channel)
    image_shape = reader.level_shape(0)
    preview = reader.read_level(cfg.preview_level)

    model = StarDist2D.from_pretrained(cfg.stardist.model)
    processed = reused = 0
    for chunk in iter_chunks(image_shape, cfg.chunk_size_px, cfg.halo_px):
        csv_path = chunks / f"nuclei_{chunk.chunk_id:06d}.csv"
        segment_path = chunks / f"segments_{chunk.chunk_id:06d}.npz"
        if csv_path.exists() and segment_path.exists() and not force:
            reused += 1
            continue
        LOG.info("Segmenting level-0 chunk %d", chunk.chunk_id)
        image = reader.read_region(
            y=chunk.read_y0, x=chunk.read_x0,
            height=chunk.read_y1 - chunk.read_y0,
            width=chunk.read_x1 - chunk.read_x0, level=0,
        )
        measured, polygons = segment_and_measure(
            image, model, prob_thresh=cfg.stardist.prob_thresh,
            nms_thresh=cfg.stardist.nms_thresh, n_tiles=cfg.stardist.n_tiles,
        )
        measured, polygons = keep_owned_nuclei(measured, polygons, chunk)
        _write_csv(measured, csv_path)
        _write_polygons(polygons, segment_path)
        processed += 1
    LOG.info("Chunks processed: %d; reused: %d", processed, reused)

    frames = []
    for path in sorted(chunks.glob("nuclei_*.csv")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No nuclei were detected")
    nuclei = pd.concat(frames, ignore_index=True)
    _write_csv(nuclei, output / "nuclei.csv")

    windows = aggregate_windows(
        nuclei, image_shape=image_shape, window_size=cfg.window_size_px,
        stride=cfg.stride_px,
    )
    clustered, selection = fit_clusters(windows, cfg, output)
    _write_csv(clustered, output / "grid_clusters.csv")
    render_cluster_overlay(
        preview, clustered, image_shape=image_shape, stride=cfg.stride_px,
        alpha=cfg.overlay_alpha, path=output / "whole_tissue_cluster_overlay.png",
    )
    LOG.info("Writing level-0 pyramidal nuclear-segmentation overlay")
    render_segmentation_ometiff(
        reader, chunks, output / "nuclear_segmentation_overlay.ome.tif",
        preview=preview, chunk_size=cfg.chunk_size_px,
        outline_width=cfg.segmentation_outline_width,
    )

    summary = {
        "feature_schema_version": "1.0",
        "input_qptiff": str(cfg.input_qptiff),
        "image_shape_level0": list(image_shape),
        "nuclei": int(len(nuclei)),
        "cluster_selection": selection,
        "configuration": asdict(cfg),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"
    )
    LOG.info("Complete: %s", output)
    return output
