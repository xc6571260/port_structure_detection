import numpy as np

def tile_image(img, tile_size=1024):
    h, w = img.shape[:2]
    tiles, positions = [], []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img[y:min(y+tile_size, h), x:min(x+tile_size, w)]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append(tile)
            positions.append((x, y))
    return tiles, positions, h, w
