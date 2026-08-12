import numpy as np
import pandas as pd

from dapi_grid.grid_features import aggregate_grid_features
from dapi_grid.image_ops import lowres_fraction_for_level0_box
from dapi_grid.io import SlideReader
from dapi_grid.tiling import iter_chunks


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


def test_reader_uses_xy_width_height_order():
    class FakeSlide:
        def __init__(self):
            self.args = None

        def read_region(self, channel, **kwargs):
            self.args = (channel, kwargs)
            return np.zeros((30, 40), dtype=np.uint8)

    reader = SlideReader.__new__(SlideReader)
    reader.channel = "DAPI"
    reader._slide = FakeSlide()
    result = reader.read_region(y=11, x=22, height=30, width=40, level=0)
    assert result.shape == (30, 40)
    assert reader._slide.args == (
        "DAPI",
        {"pos": (22, 11), "shape": (40, 30), "level": 0},
    )
