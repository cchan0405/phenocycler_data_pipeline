from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml


@dataclass
class StarDistConfig:
    model: str = "2D_versatile_fluo"
    prob_thresh: float = 0.50
    nms_thresh: float = 0.40
    n_tiles: tuple[int, int] = (2, 2)


@dataclass
class Config:
    input_qptiff: Path
    output_dir: Path
    channel: str | int = "DAPI"
    preview_level: int = 5
    chunk_size_px: int = 4096
    halo_px: int = 128
    window_size_px: int = 512
    stride_px: int = 256
    k_min: int = 2
    k_max: int = 10
    silhouette_sample: int = 10_000
    minimum_cluster_fraction: float = 0.01
    stability_repeats: int = 5
    random_seed: int = 42
    overlay_alpha: float = 0.55
    segmentation_outline_width: int = 2
    pixel_size_um: float | None = None
    stardist: StarDistConfig = field(default_factory=StarDistConfig)


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("input_qptiff", "output_dir"):
        value = Path(data[key]).expanduser()
        data[key] = (path.parent / value).resolve() if not value.is_absolute() else value
    sd = dict(data.pop("stardist", {}))
    if "n_tiles" in sd:
        sd["n_tiles"] = tuple(sd["n_tiles"])
    data["stardist"] = StarDistConfig(**sd)
    cfg = Config(**data)
    if cfg.halo_px * 2 >= cfg.chunk_size_px:
        raise ValueError("halo_px must be smaller than half chunk_size_px")
    if cfg.window_size_px < cfg.stride_px:
        raise ValueError("window_size_px must be >= stride_px")
    if not 2 <= cfg.k_min <= cfg.k_max:
        raise ValueError("Require 2 <= k_min <= k_max")
    if not 0 <= cfg.minimum_cluster_fraction < 0.5:
        raise ValueError("minimum_cluster_fraction must be between 0 and 0.5")
    return cfg


@dataclass
class ComparisonSample:
    name: str
    results_dir: Path


@dataclass
class ComparisonConfig:
    control: ComparisonSample
    experimentals: list[ComparisonSample]
    output_dir: Path
    k_min: int = 2
    k_max: int = 10
    max_training_windows_per_image: int = 5000
    silhouette_sample: int = 10_000
    minimum_cluster_fraction: float = 0.01
    stability_repeats: int = 5
    random_seed: int = 42
    overlay_alpha: float = 0.55


def _sample(data: dict, base: Path) -> ComparisonSample:
    results = Path(data["results_dir"]).expanduser()
    return ComparisonSample(
        name=str(data["name"]),
        results_dir=(base / results).resolve() if not results.is_absolute() else results,
    )


def load_comparison_config(path: str | Path) -> ComparisonConfig:
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    output = Path(data["output_dir"]).expanduser()
    cfg = ComparisonConfig(
        control=_sample(data["control"], path.parent),
        experimentals=[_sample(item, path.parent) for item in data["experimentals"]],
        output_dir=(path.parent / output).resolve() if not output.is_absolute() else output,
        **{key: value for key, value in data.items()
           if key not in {"control", "experimentals", "output_dir"}},
    )
    names = [cfg.control.name] + [item.name for item in cfg.experimentals]
    if not cfg.experimentals:
        raise ValueError("At least one experimental result is required")
    if len(names) != len(set(names)):
        raise ValueError("Every comparison sample name must be unique")
    if any(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None for name in names):
        raise ValueError("Sample names may contain only letters, numbers, ., _ and -")
    if not 2 <= cfg.k_min <= cfg.k_max:
        raise ValueError("Require 2 <= k_min <= k_max")
    return cfg
