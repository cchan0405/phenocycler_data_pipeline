import torch

def extract_tiles_limit(image, tile_size = 256, stride = 128, max_tiles = 500):
    H, W = image.shape
    tiles = []
    for x in range(0, W - tile_size + 1, stride):
        for y in range(0, H - tile_size + 1, stride):
            tile = image[y : y+tile_size, x : x+tile_size]
            tiles.append(tile)
            if len(tiles) >= max_tiles:
                return tiles
    return tiles