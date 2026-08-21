from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stardist.models import StarDist2D

from .clustering import fit_grid_clusters
from .config import Config
from .image_ops import lowres_fraction_for_level0_box, tissue_mask
from .io import SlideReader
from .nuclei import keep_owned_nuclei, segment_and_measure
from .phenotypes import assign_nuclear_phenotypes
from .render import render_confidence_overlay, render_overlay, render_status_overlay
from .tiling import grid_shape, iter_chunks
from .window_features import aggregate_overlapping_windows

LOG = logging.getLogger("dapi_grid")


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _chunk_path(output: Path, chunk_id: int) -> Path:
    return output / "chunks" / f"nuclei_{chunk_id:06d}.csv"


def run_pipeline(cfg: Config, *, force: bool = False) -> Path:
    out = cfg.output_dir
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    reader = SlideReader(cfg.input_qptiff, cfg.channel)

    LOG.info("Reading low-resolution DAPI at level %d", cfg.detection_level)
    low = reader.read_level(cfg.detection_level)
    low_norm, mask = tissue_mask(
        low,
        gaussian_sigma=cfg.tissue_mask.gaussian_sigma,
        min_object_area=cfg.tissue_mask.min_object_area,
        closing_radius=cfg.tissue_mask.closing_radius,
    )
    level0_shape = reader.level_shape(0)
    detection_shape = reader.level_shape(cfg.detection_level)
    if tuple(low.shape) != tuple(detection_shape):
        raise RuntimeError(
            f"Reader returned low-resolution shape {low.shape}, but metadata reports "
            f"{detection_shape} at level {cfg.detection_level}."
        )
    np.savez_compressed(out / "tissue_mask.npz", mask=mask, dapi=low_norm)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "input_qptiff": str(cfg.input_qptiff),
                "channel": cfg.channel,
                "detection_level": cfg.detection_level,
                "low_shape": list(low.shape),
                "level0_shape": list(level0_shape),
                "grid_size_px": cfg.resolved_display_stride_px,
                "display_stride_px": cfg.resolved_display_stride_px,
                "analysis_window_px": cfg.resolved_analysis_window_px,
                "pixel_size_um": cfg.pixel_size_um,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model = StarDist2D.from_pretrained(cfg.stardist.model)
    processed = skipped = 0
    for chunk in iter_chunks(level0_shape, cfg.chunk_size_px, cfg.halo_px):
        path = _chunk_path(out, chunk.chunk_id)
        if path.exists() and not force:
            skipped += 1
            continue
        tissue_fraction = lowres_fraction_for_level0_box(
            mask,
            (chunk.core_y0, chunk.core_x0, chunk.core_y1, chunk.core_x1),
            level0_shape,
        )
        if tissue_fraction < cfg.tissue_mask.min_positive_fraction:
            _write_csv_atomic(pd.DataFrame(), path)
            skipped += 1
            continue
        LOG.info("Segmenting chunk %d", chunk.chunk_id)
        image = reader.read_region(
            y=chunk.read_y0,
            x=chunk.read_x0,
            height=chunk.read_y1 - chunk.read_y0,
            width=chunk.read_x1 - chunk.read_x0,
            level=0,
        )
        measured = segment_and_measure(
            image,
            model,
            prob_thresh=cfg.stardist.prob_thresh,
            nms_thresh=cfg.stardist.nms_thresh,
            n_tiles=cfg.stardist.n_tiles,
        )
        kept = keep_owned_nuclei(measured, chunk)
        _write_csv_atomic(kept, path)
        processed += 1
    LOG.info("Chunks newly processed: %d; reused/skipped: %d", processed, skipped)

    frames = []
    for path in sorted((out / "chunks").glob("nuclei_*.csv")):
        if path.stat().st_size > 1:
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    frames.append(frame)
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        raise RuntimeError("No StarDist-detected nuclei were found.")
    nuclei = pd.concat(frames, ignore_index=True)
    nuclei, phenotype_artifact = assign_nuclear_phenotypes(nuclei, cfg.phenotypes)
    if phenotype_artifact is not None:
        import joblib

        joblib.dump(phenotype_artifact, out / "nuclear_phenotype_model.joblib")
    _write_csv_atomic(nuclei, out / "nuclei.csv")

    display_stride = cfg.resolved_display_stride_px
    analysis_window = cfg.resolved_analysis_window_px
    n_rows, n_cols = grid_shape(level0_shape, display_stride)
    display_fractions = {}
    analysis_fractions = {}
    half = analysis_window / 2.0
    for gr in range(n_rows):
        for gc in range(n_cols):
            display_box = (
                gr * display_stride,
                gc * display_stride,
                min(level0_shape[0], (gr + 1) * display_stride),
                min(level0_shape[1], (gc + 1) * display_stride),
            )
            display_fractions[(gr, gc)] = lowres_fraction_for_level0_box(
                mask, display_box, level0_shape
            )
            cy = min(level0_shape[0] - 1, (gr + 0.5) * display_stride)
            cx = min(level0_shape[1] - 1, (gc + 0.5) * display_stride)
            analysis_box = (
                max(0, int(cy - half)),
                max(0, int(cx - half)),
                min(level0_shape[0], int(np.ceil(cy + half))),
                min(level0_shape[1], int(np.ceil(cx + half))),
            )
            analysis_fractions[(gr, gc)] = lowres_fraction_for_level0_box(
                mask, analysis_box, level0_shape
            )
    grid_df = aggregate_overlapping_windows(
        nuclei,
        level0_shape=level0_shape,
        analysis_window_px=analysis_window,
        display_stride_px=display_stride,
        display_tissue_fraction=display_fractions,
        analysis_tissue_fraction=analysis_fractions,
        min_nuclei_predict=cfg.coverage.min_nuclei_predict,
        min_tissue_fraction_predict=cfg.coverage.min_tissue_fraction_predict,
    )
    if grid_df.empty:
        raise RuntimeError("No windows passed the prediction-coverage thresholds.")
    _write_csv_atomic(grid_df, out / "grid_features.csv")
    _write_csv_atomic(grid_df, out / "window_features.csv")
    clustered = fit_grid_clusters(
        grid_df,
        cfg.clustering,
        out,
        training_cfg=cfg.training,
        density_cfg=cfg.density_correction,
        confidence_cfg=cfg.confidence,
    )
    _write_csv_atomic(clustered, out / "grid_clusters.csv")
    render_overlay(
        low,
        clustered,
        level0_shape=level0_shape,
        grid_size=display_stride,
        alpha=cfg.render.alpha,
        draw_grid=cfg.render.draw_grid,
        output_path=out / "whole_tissue_cluster_overlay.png",
        confidence_weighted=cfg.confidence.confidence_weighted_alpha,
        minimum_confidence=cfg.confidence.minimum_display_confidence,
    )
    render_overlay(
        low,
        clustered,
        level0_shape=level0_shape,
        grid_size=display_stride,
        alpha=cfg.render.alpha,
        draw_grid=cfg.render.draw_grid,
        output_path=out / "whole_tissue_unweighted_overlay.png",
        confidence_weighted=False,
        minimum_confidence=0.0,
    )
    render_confidence_overlay(
        low,
        clustered,
        level0_shape=level0_shape,
        grid_size=display_stride,
        output_path=out / "whole_tissue_confidence_overlay.png",
    )
    render_status_overlay(
        low,
        clustered,
        level0_shape=level0_shape,
        grid_size=display_stride,
        output_path=out / "whole_tissue_training_vs_predicted.png",
    )
    LOG.info("Complete: %s", out)
    return out
