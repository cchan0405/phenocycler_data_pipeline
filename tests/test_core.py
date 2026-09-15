from types import SimpleNamespace

import numpy as np
import pandas as pd

from dapi_grid.clustering import _elbow, _spatial_sample
from dapi_grid.comparison import _fit_joint, _spatial_metrics, _summary_table
from dapi_grid.nuclei import keep_owned_nuclei
from dapi_grid.render import render_segmentation_ometiff
from dapi_grid.tiling import iter_chunks
from dapi_grid.window_features import aggregate_windows


def test_chunks_cover_level0_once():
    chunks = list(iter_chunks((100, 130), 64, 8))
    coverage = np.zeros((100, 130), dtype=np.uint8)
    for chunk in chunks:
        coverage[chunk.core_y0:chunk.core_y1, chunk.core_x0:chunk.core_x1] += 1
    assert np.all(coverage == 1)


def test_ownership_filters_measurements_and_real_polygons_together():
    chunk = SimpleNamespace(chunk_id=0, read_y0=90, read_x0=190,
                            core_y0=100, core_x0=200, core_y1=300, core_x1=400)
    frame = pd.DataFrame({"local_y": [20, 5], "local_x": [20, 5]})
    polygons = np.array([[[19, 20, 21], [19, 21, 20]],
                         [[4, 5, 6], [4, 6, 5]]], dtype=np.float32)
    retained, retained_polygons = keep_owned_nuclei(frame, polygons, chunk)
    assert len(retained) == len(retained_polygons) == 1
    assert np.allclose(retained_polygons[0, 0], [109, 110, 111])
    assert np.allclose(retained_polygons[0, 1], [209, 211, 210])


def _nuclei(n=120):
    rng = np.random.default_rng(4)
    data = {"x": rng.uniform(0, 400, n), "y": rng.uniform(0, 400, n)}
    for feature in ["area", "perimeter", "circularity", "eccentricity", "solidity",
                    "major_axis_length", "minor_axis_length", "aspect_ratio"]:
        data[feature] = rng.normal(3, .5, n)
    return pd.DataFrame(data)


def test_every_nonempty_window_is_retained():
    windows = aggregate_windows(_nuclei(), image_shape=(400, 400),
                                window_size=200, stride=100)
    assert len(windows) == 16
    assert windows.n_nuclei.min() >= 1
    assert "area_median" in windows and "area_iqr" in windows


def test_spatial_silhouette_sample_avoids_overlapping_neighbours():
    frame = pd.DataFrame([(r, c) for r in range(8) for c in range(8)],
                         columns=["grid_row", "grid_col"])
    selected = _spatial_sample(frame, 100, window_size=200, stride=100, seed=42)
    subset = frame.loc[selected]
    assert subset.grid_row.mod(2).eq(0).all()
    assert subset.grid_col.mod(2).eq(0).all()


def test_elbow_finds_curve_bend():
    assert _elbow(np.array([2, 3, 4, 5, 6]),
                  np.array([100, 55, 38, 32, 29])) in {3, 4}


def test_segmentation_output_is_a_level0_pyramid(tmp_path):
    import tifffile

    class FakeReader:
        level_count = 2

        def level_shape(self, level=0):
            return [(65, 70), (33, 35)][level]

        def read_region(self, *, y, x, height, width, level=0):
            return np.full((height, width), 100, dtype=np.uint16)

    polygon = np.array([[[10, 10, 20, 20], [10, 20, 20, 10]]], dtype=np.float32)
    np.savez_compressed(tmp_path / "segments_000000.npz", polygons=polygon)
    output = tmp_path / "segmentation.ome.tif"
    render_segmentation_ometiff(
        FakeReader(), tmp_path, output, preview=np.arange(100).reshape(10, 10),
        chunk_size=4096, tile_size=32,
    )
    with tifffile.TiffFile(output) as slide:
        assert slide.is_ome
        assert slide.series[0].shape == (65, 70, 3)
        assert len(slide.series[0].levels) == 2


def _window_frame(name, group, shift=0):
    rng = np.random.default_rng(10 + shift)
    rows = []
    for row in range(8):
        for col in range(8):
            rows.append({
                "grid_row": row, "grid_col": col, "x0": col * 100,
                "y0": row * 100, "n_nuclei": 20 + shift,
                "area_median": rng.normal(2 + shift, .2),
                "area_iqr": rng.normal(.5, .05),
                "cluster": 0, "sample": name, "group": group,
            })
    return pd.DataFrame(rows)


def test_joint_model_assigns_shared_clusters_and_builds_control_differences(tmp_path):
    control = _window_frame("control", "control", 0)
    experimental = _window_frame("experiment", "experimental", 2)
    records = [("control", control, {}), ("experiment", experimental, {})]
    cfg = SimpleNamespace(
        max_training_windows_per_image=50, random_seed=42, k_min=2, k_max=4,
        silhouette_sample=1000, minimum_cluster_fraction=.01,
        stability_repeats=2,
    )
    signature = {"window_size_px": 200, "stride_px": 100}
    labelled, selection = _fit_joint(
        records, ["area_median", "area_iqr"], signature, cfg, tmp_path
    )
    assert selection["selected_k"] >= 2
    assert all("cluster" in frame for frame in labelled)
    results = _summary_table(labelled, "control")
    assert "proportion_change" in results
    assert results.loc[results["sample"].eq("control"), "proportion_change"].eq(0).all()


def test_spatial_metrics_detect_contiguous_regions():
    frame = pd.DataFrame({
        "grid_row": [0, 0, 1, 1], "grid_col": [0, 1, 0, 1],
        "cluster": [1, 1, 1, 1],
    })
    metrics = _spatial_metrics(frame)
    assert metrics["neighbour_agreement"] == 1
    assert metrics["median_patch_windows"] == 4
