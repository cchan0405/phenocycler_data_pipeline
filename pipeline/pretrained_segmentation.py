from qptifffile import QPTiffFile
import matplotlib.pyplot as plt
import numpy as np
from pipeline.data_loading import image_loader
from cellpose import models

def cellpose_segmentor(file):
    f = QPTiffFile(file)
    dapi_image = f.read_region('DAPI')
    dapi_image = dapi_image[:2000, :2000]
    model = models.CellposeModel(model_type='nuclei')
    masks, _, _ = model.eval(dapi_image, diameter=None, channels=[0,0])

    opal780_image = image_loader(file)
    opal780_image = opal780_image[:2000, :2000]

    cell_tiles = []
    for cell_id in np.unique(masks, opal780_image):
        if cell_id == 0:  # skip background
            continue
        # get bounding box of this cell
        rows, cols = np.where(masks == cell_id)
        y1, y2 = rows.min(), rows.max()
        x1, x2 = cols.min(), cols.max()
        
        # extract from opal780
        
        tile = opal780_image[y1:y2, x1:x2]
        cell_tiles.append(tile)
        if len(cell_tiles) > 20:
            return cell_tiles
        return cell_tiles