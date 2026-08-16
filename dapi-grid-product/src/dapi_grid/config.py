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
class NucleusQCConfig:
    min_area_px: float = 80
    max_area_px: float = 3000
    min_solidity: float = 0.35
    max_dapi_normalized: float = 0.98


@dataclass
class GridQCConfig:
    min_nuclei: int = 10
    min_tissue_fraction: float = 0.20


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
    nucleus_qc: NucleusQCConfig = field(default_factory=NucleusQCConfig)
    grid_qc: GridQCConfig = field(default_factory=GridQCConfig)
    phenotypes: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    render: RenderConfig = field(default_factory=RenderConfig)

    @property
    def resolved_grid_size_px(self) -> int:
        if self.grid_size_um is not None and self.pixel_size_um is not None:
            return max(1, round(self.grid_size_um / self.pixel_size_um))
        return self.grid_size_px


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
    data["nucleus_qc"] = _nested(NucleusQCConfig, data, "nucleus_qc")
    data["grid_qc"] = _nested(GridQCConfig, data, "grid_qc")
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
    return cfg
