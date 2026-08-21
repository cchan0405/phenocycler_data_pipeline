from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TissueMaskConfig:
    min_positive_fraction: float = 0.03
    gaussian_sigma: float = 1.0
    min_object_area: int = 200
    closing_radius: int = 2


@dataclass
class StarDistConfig:
    model: str = "2D_versatile_fluo"
    prob_thresh: float = 0.50
    nms_thresh: float = 0.40
    n_tiles: tuple[int, int] = (2, 2)


@dataclass
class GridQCConfig:
    min_nuclei: int = 10
    min_tissue_fraction: float = 0.20


@dataclass
class SpatialConfig:
    """Separate the measurement neighbourhood from the rendered tile size."""

    analysis_window_px: int | None = None
    display_stride_px: int | None = None


@dataclass
class CoverageConfig:
    min_nuclei_predict: int = 1
    min_tissue_fraction_predict: float = 0.005


@dataclass
class TrainingConfig:
    min_nuclei_train: int = 8
    min_tissue_fraction_train: float = 0.10


@dataclass
class DensityCorrectionConfig:
    residualize: bool = True
    spline_knots: int = 4
    ridge_alpha: float = 1.0


@dataclass
class ConfidenceConfig:
    low_confidence_threshold: float = 0.40
    minimum_display_confidence: float = 0.05
    confidence_weighted_alpha: bool = True


@dataclass
class PhenotypeConfig:
    enabled: bool = True
    n_phenotypes: int = 5
    fit_sample: int = 200_000
    minibatch_size: int = 4096
    random_seed: int = 42


@dataclass
class ClusteringConfig:
    k_min: int = 3
    k_max: int = 10
    fixed_k: int | None = None
    random_seed: int = 42
    silhouette_sample: int = 10_000
    minibatch_size: int = 2048
    stability_repeats: int = 5
    stability_weight: float = 0.15
    complexity_weight: float = 0.015
    density_mode: str = "controlled"
    density_weight: float = 0.20
    include_dapi_intensity: bool = False


@dataclass
class RenderConfig:
    preview_level: int = 5
    alpha: float = 0.55
    draw_grid: bool = False


@dataclass
class Config:
    input_qptiff: Path
    output_dir: Path
    channel: str = "DAPI"
    detection_level: int = 5
    chunk_size_px: int = 4096
    halo_px: int = 128
    grid_size_px: int = 512
    grid_size_um: float | None = None
    pixel_size_um: float | None = None
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)
    stardist: StarDistConfig = field(default_factory=StarDistConfig)
    grid_qc: GridQCConfig = field(default_factory=GridQCConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    density_correction: DensityCorrectionConfig = field(
        default_factory=DensityCorrectionConfig
    )
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    phenotypes: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @property
    def resolved_grid_size_px(self) -> int:
        if self.grid_size_um is not None and self.pixel_size_um is not None:
            return max(1, round(self.grid_size_um / self.pixel_size_um))
        return self.grid_size_px

    @property
    def resolved_analysis_window_px(self) -> int:
        return self.spatial.analysis_window_px or self.resolved_grid_size_px

    @property
    def resolved_display_stride_px(self) -> int:
        return self.spatial.display_stride_px or self.resolved_grid_size_px


def _nested(cls: type, data: dict[str, Any], key: str):
    return cls(**data.get(key, {}))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["input_qptiff"] = Path(data["input_qptiff"]).expanduser()
    data["output_dir"] = Path(data["output_dir"]).expanduser()
    data["tissue_mask"] = _nested(TissueMaskConfig, data, "tissue_mask")
    sd = dict(data.get("stardist", {}))
    if "n_tiles" in sd:
        sd["n_tiles"] = tuple(sd["n_tiles"])
    data["stardist"] = StarDistConfig(**sd)
    # Older configuration files may still contain this section. It is ignored:
    # all StarDist detections are now retained after chunk-ownership filtering.
    data.pop("nucleus_qc", None)
    has_coverage = "coverage" in data
    has_training = "training" in data
    data["grid_qc"] = _nested(GridQCConfig, data, "grid_qc")
    data["spatial"] = _nested(SpatialConfig, data, "spatial")
    data["coverage"] = (
        _nested(CoverageConfig, data, "coverage")
        if has_coverage
        else CoverageConfig(
            min_nuclei_predict=data["grid_qc"].min_nuclei,
            min_tissue_fraction_predict=data["grid_qc"].min_tissue_fraction,
        )
    )
    data["training"] = (
        _nested(TrainingConfig, data, "training")
        if has_training
        else TrainingConfig(
            min_nuclei_train=data["grid_qc"].min_nuclei,
            min_tissue_fraction_train=data["grid_qc"].min_tissue_fraction,
        )
    )
    data["density_correction"] = _nested(
        DensityCorrectionConfig, data, "density_correction"
    )
    data["confidence"] = _nested(ConfidenceConfig, data, "confidence")
    data["phenotypes"] = _nested(PhenotypeConfig, data, "phenotypes")
    data["clustering"] = _nested(ClusteringConfig, data, "clustering")
    data["render"] = _nested(RenderConfig, data, "render")
    cfg = Config(**data)
    if cfg.halo_px * 2 >= cfg.chunk_size_px:
        raise ValueError("halo_px must be smaller than half chunk_size_px")
    if cfg.clustering.fixed_k is None and cfg.clustering.k_min > cfg.clustering.k_max:
        raise ValueError("clustering.k_min must be <= clustering.k_max")
    if cfg.clustering.density_mode not in {"exclude", "controlled", "full"}:
        raise ValueError("clustering.density_mode must be exclude, controlled or full")
    if cfg.resolved_analysis_window_px < cfg.resolved_display_stride_px:
        raise ValueError("analysis_window_px must be >= display_stride_px")
    if cfg.training.min_nuclei_train < cfg.coverage.min_nuclei_predict:
        raise ValueError("min_nuclei_train must be >= min_nuclei_predict")
    return cfg
