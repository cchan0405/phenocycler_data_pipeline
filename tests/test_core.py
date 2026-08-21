import numpy as np
import pandas as pd
from types import SimpleNamespace

from dapi_grid.grid_features import aggregate_grid_features, model_feature_columns
from dapi_grid.image_ops import lowres_fraction_for_level0_box
from dapi_grid.io import SlideReader
from dapi_grid.tiling import iter_chunks
from dapi_grid.window_features import aggregate_overlapping_windows
from dapi_grid.clustering import fit_grid_clusters
from dapi_grid.nuclei import keep_owned_nuclei


def test_chunks_cover_core_once():
    chunks = list(iter_chunks((100, 130), chunk_size=64, halo=8))
    coverage = np.zeros((100, 130), dtype=np.uint8)
    for c in chunks:
        coverage[c.core_y0 : c.core_y1, c.core_x0 : c.core_x1] += 1
        assert c.read_y0 <= c.core_y0 <= c.core_y1 <= c.read_y1
    assert np.all(coverage == 1)


def test_lowres_fraction_mapping():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :5] = True
    assert lowres_fraction_for_level0_box(mask, (0, 0, 50, 50), (100, 100)) == 1
    assert lowres_fraction_for_level0_box(mask, (50, 50, 100, 100), (100, 100)) == 0


def test_grid_aggregation():
    n = 12
    data = {
        "x": np.arange(n) + 10,
        "y": np.arange(n) + 10,
    }
    for name in [
        "area",
        "perimeter",
        "circularity",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "aspect_ratio",
        "mean_dapi",
    ]:
        data[name] = np.linspace(1, 2, n)
    result = aggregate_grid_features(
        pd.DataFrame(data),
        grid_size=100,
        min_nuclei=10,
        tissue_fraction={(0, 0): 1.0},
        min_tissue_fraction=0.2,
    )
    assert len(result) == 1
    assert result.iloc[0]["n_nuclei"] == 12
    assert "area_iqr" in result
    assert "architecture_normalized_nn_median" in result


def test_grid_composition_and_density_are_separated():
    n = 20
    data = {"x": np.arange(n), "y": np.arange(n), "orientation": np.zeros(n)}
    for name in [
        "area", "perimeter", "circularity", "eccentricity", "solidity",
        "major_axis_length", "minor_axis_length", "aspect_ratio", "mean_dapi",
    ]:
        data[name] = np.linspace(1, 2, n)
    data["nuclear_phenotype"] = np.repeat([0, 1], 10)
    result = aggregate_grid_features(
        pd.DataFrame(data), grid_size=100, min_nuclei=10,
        tissue_fraction={(0, 0): 0.5}, min_tissue_fraction=0.2,
    )
    row = result.iloc[0]
    assert row["phenotype_0_proportion"] == 0.5
    assert row["phenotype_1_proportion"] == 0.5
    assert np.isclose(row["nuclear_density_px2"], 20 / 5000)
    features = model_feature_columns(result)
    assert "n_nuclei" not in features
    assert "nuclear_density_px2" not in features
    assert "log_nuclear_density" in features
    assert not any(name.startswith("mean_dapi_") for name in features)


def test_reader_uses_xy_width_height_order():
    class FakeSlide:
        def __init__(self):
            self.args = None

        def read_region(self, channel, **kwargs):
            self.args = (channel, kwargs)
            return np.zeros((30, 40), dtype=np.uint8)

    reader = SlideReader.__new__(SlideReader)
    reader.channel = "DAPI"
    reader._layer = "DAPI"
    reader._slide = FakeSlide()
    result = reader.read_region(y=11, x=22, height=30, width=40, level=0)
    assert result.shape == (30, 40)
    assert reader._slide.args == (
        "DAPI",
        {"pos": (22, 11), "shape": (40, 30), "level": 0},
    )


def test_all_owned_stardist_detections_are_retained_without_qc():
    chunk = SimpleNamespace(
        chunk_id=7,
        read_y0=90,
        read_x0=190,
        core_y0=100,
        core_x0=200,
        core_y1=300,
        core_x1=400,
    )
    detections = pd.DataFrame(
        {
            "local_y": [20, 30, 40, 5],
            "local_x": [20, 30, 40, 5],
            "area": [1, 100_000, 10, 100],
            "solidity": [0.01, 0.99, 0.10, 0.99],
            "max_dapi": [1.0, 0.5, 0.999, 0.5],
        }
    )
    retained = keep_owned_nuclei(detections, chunk)
    assert len(retained) == 3
    assert retained["chunk_id"].eq(7).all()
    assert set(retained["area"]) == {1, 10, 100_000}


def _synthetic_nuclei(n=120):
    rng = np.random.default_rng(4)
    data = {
        "x": rng.uniform(0, 400, n),
        "y": rng.uniform(0, 400, n),
        "orientation": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "nuclear_phenotype": np.arange(n) % 3,
    }
    for name in [
        "area", "perimeter", "circularity", "eccentricity", "solidity",
        "major_axis_length", "minor_axis_length", "aspect_ratio", "mean_dapi",
    ]:
        data[name] = rng.normal(2 + (np.arange(n) % 3), 0.2, n)
    return pd.DataFrame(data)


def test_overlapping_window_uses_larger_context_than_stride():
    nuclei = _synthetic_nuclei()
    fractions = {(r, c): 1.0 for r in range(4) for c in range(4)}
    windows = aggregate_overlapping_windows(
        nuclei,
        level0_shape=(400, 400),
        analysis_window_px=200,
        display_stride_px=100,
        display_tissue_fraction=fractions,
        analysis_tissue_fraction=fractions,
        min_nuclei_predict=1,
        min_tissue_fraction_predict=0,
    )
    assert len(windows) == 16
    assert windows["n_nuclei"].median() > 1
    assert "phenotype_0_proportion" in windows
    assert "architecture_available" in windows


def test_reliable_windows_fit_and_sparse_windows_only_predict(tmp_path):
    nuclei = _synthetic_nuclei(300)
    fractions = {(r, c): 1.0 for r in range(8) for c in range(8)}
    windows = aggregate_overlapping_windows(
        nuclei,
        level0_shape=(400, 400),
        analysis_window_px=150,
        display_stride_px=50,
        display_tissue_fraction=fractions,
        analysis_tissue_fraction=fractions,
        min_nuclei_predict=1,
        min_tissue_fraction_predict=0,
    )
    windows.loc[0, "n_nuclei"] = 1
    clustering = SimpleNamespace(
        density_mode="exclude", include_dapi_intensity=False, density_weight=0,
        random_seed=42, k_min=2, k_max=3, fixed_k=2,
        silhouette_sample=1000, minibatch_size=64, stability_repeats=2,
        stability_weight=0.15, complexity_weight=0.015,
    )
    training = SimpleNamespace(min_nuclei_train=8, min_tissue_fraction_train=0.1)
    density = SimpleNamespace(residualize=True, spline_knots=4, ridge_alpha=1.0)
    confidence = SimpleNamespace(low_confidence_threshold=0.4)
    result = fit_grid_clusters(
        windows,
        clustering,
        tmp_path,
        training_cfg=training,
        density_cfg=density,
        confidence_cfg=confidence,
    )
    assert not bool(result.loc[0, "is_training_window"])
    assert result.loc[0, "sampling_confidence"] == 1 / 8
    assert (tmp_path / "density_residualization_report.csv").exists()
    assert (tmp_path / "cluster_feature_profiles.csv").exists()
