from __future__ import annotations

from pathlib import Path

import numpy as np
from mxtifffile import MxTiffFile


def squeeze_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a single 2-D channel, received {arr.shape}")
    return arr


class SlideReader:
    """Small adapter around MxTiffFile so the rest of the pipeline is testable."""

    def __init__(self, path: str | Path, channel: str = "DAPI"):
        self.path = Path(path)
        self.channel = channel
        self._slide = MxTiffFile(str(self.path))

    def read_level(self, level: int) -> np.ndarray:
        return squeeze_channel(self._slide.read_region(self.channel, level=level))

    def read_region(
        self, *, y: int, x: int, height: int, width: int, level: int = 0
    ) -> np.ndarray:
        # MxTiffFile documents pos=(x, y) and shape=(width, height).
        arr = self._slide.read_region(
            self.channel, pos=(x, y), shape=(width, height), level=level
        )
        return squeeze_channel(arr)

    def level_shape(self, level: int = 0) -> tuple[int, int]:
        """Return (height, width) from TIFF metadata without reading image pixels."""
        levels = self._slide.series[0].levels
        if level < 0 or level >= len(levels):
            raise ValueError(f"Pyramid level {level} is unavailable; found {len(levels)} levels")
        shape = levels[level].pages[0].shape
        return int(shape[0]), int(shape[1])
