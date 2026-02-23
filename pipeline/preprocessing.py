import torch

def extract_tiles(image, tile_size = 256, stride = 128):
    H, W = image.shape
    tiles = []
    for x in range(0, W - tile_size + 1, stride):
        for y in range(0, H - tile_size + 1, stride):
            tile = image[y : y+tile_size, x : x+tile_size]
            tiles.append(tile)
    return tiles
    
def background_cleaner(tile, signal_threshold = 0.1, tile_threshold = 0.25):
    if tile.dim() == 3:
        tile = tile[0]
    else:
        pass
    average_signal = (tile > signal_threshold).float().mean()
    return average_signal > tile_threshold

def filtering(tiles):
    filtered_tiles = []
    for tile in tiles:
        tile = torch.from_numpy(tile)
        if background_cleaner(tile):
            filtered_tiles.append(tile)
    return filtered_tiles