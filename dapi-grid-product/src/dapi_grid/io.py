from __future__ import annotations

from pathlib import Path

import numpy as np
from mxtifffile import MxTiffFile


def squeeze_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a single 2-D channel, received {arr.shape}"
        )

    return arr


class SlideReader:
    """Adapter around MxTiffFile with support for biomarker and fluorophore names."""

    def __init__(self, path: str | Path, channel: str | int = "DAPI"):
        self.path = Path(path)
        self.channel = channel
        self._slide = MxTiffFile(str(self.path))

        # Resolve a channel name to its actual layer index.
        self._layer = self._resolve_channel(channel)

        print(
            f"Using channel {self._layer} "
            f"for requested channel '{channel}'"
        )

    def _resolve_channel(self, channel: str | int) -> int:
        """Resolve a channel name/index to an MxTiffFile layer index."""

        # If the config already supplies an integer, use it directly.
        if isinstance(channel, int):
            if channel < 0 or channel >= len(self._slide.channel_info):
                raise ValueError(
                    f"Channel index {channel} is unavailable. "
                    f"Found {len(self._slide.channel_info)} channels."
                )
            return channel

        requested = str(channel).strip().lower()

        # First try biomarker names.
        for info in self._slide.channel_info:
            biomarker = info.get("biomarker")

            if biomarker and str(biomarker).strip().lower() == requested:
                return int(info["index"])

        # Then try fluorophore names.
        for info in self._slide.channel_info:
            fluorophore = info.get("fluorophore")

            if fluorophore and str(fluorophore).strip().lower() == requested:
                return int(info["index"])

        available = [
            {
                "index": info.get("index"),
                "biomarker": info.get("biomarker"),
                "fluorophore": info.get("fluorophore"),
            }
            for info in self._slide.channel_info
        ]

        raise ValueError(
            f"Channel '{channel}' was not found.\n"
            f"Available channels:\n{available}"
        )

    def read_level(self, level: int) -> np.ndarray:
        return squeeze_channel(
            self._slide.read_region(
                self._layer,
                level=level,
            )
        )

    def read_region(
        self,
        *,
        y: int,
        x: int,
        height: int,
        width: int,
        level: int = 0,
    ) -> np.ndarray:
        # MxTiffFile documents pos=(x, y) and shape=(width, height).
        arr = self._slide.read_region(
            self._layer,
            pos=(x, y),
            shape=(width, height),
            level=level,
        )

        return squeeze_channel(arr)

    def level_shape(self, level: int = 0) -> tuple[int, int]:
        """Return (height, width) from TIFF metadata without reading pixels."""

        levels = self._slide.series[0].levels

        if level < 0 or level >= len(levels):
            raise ValueError(
                f"Pyramid level {level} is unavailable; "
                f"found {len(levels)} levels"
            )

        shape = levels[level].pages[0].shape

        return int(shape[0]), int(shape[1])