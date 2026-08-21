from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    core_y0: int
    core_x0: int
    core_y1: int
    core_x1: int
    read_y0: int
    read_x0: int
    read_y1: int
    read_x1: int


def iter_chunks(shape: tuple[int, int], chunk_size: int, halo: int):
    height, width = shape
    chunk_id = 0
    for y0 in range(0, height, chunk_size):
        for x0 in range(0, width, chunk_size):
            y1, x1 = min(height, y0 + chunk_size), min(width, x0 + chunk_size)
            yield Chunk(
                chunk_id,
                y0,
                x0,
                y1,
                x1,
                max(0, y0 - halo),
                max(0, x0 - halo),
                min(height, y1 + halo),
                min(width, x1 + halo),
            )
            chunk_id += 1


def grid_shape(image_shape: tuple[int, int], grid_size: int) -> tuple[int, int]:
    return math.ceil(image_shape[0] / grid_size), math.ceil(image_shape[1] / grid_size)

