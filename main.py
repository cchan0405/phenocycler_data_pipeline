import matplotlib.pyplot as plt
from pipeline.data_loading import image_loader
from pipeline.preprocessing import extract_tiles, filtering
from pipeline.debug import extract_tiles_limit
from pipeline.dataset import DataCharacteristic, loader_creation
from pipeline.model import Autoencoder, train
from pipeline.pretrained_segmentation import cellpose_segmentor

opal780_cell_tiles = cellpose_segmentor('/Users/chesterchan/Chester_phenocycler/data/neg ctr_Scan1.qptiff')



# tiles = extract_tiles_limit(opal780_image)
# filtered_tiles = filtering(tiles)

processed_tiles, loader = loader_creation(opal780_cell_tiles)

model = Autoencoder()
outputs = train(model, loader, num_epochs=10)