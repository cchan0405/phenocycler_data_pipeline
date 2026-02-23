from qptifffile import QPTiffFile
import matplotlib.pyplot as plt
import numpy as np

def image_loader(file):
    f = QPTiffFile(file)
    opal780_image = f.read_region('Opal 780')
    return opal780_image

