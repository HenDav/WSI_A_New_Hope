import openslide
import PIL
import numpy as np
import utils
from openslide import Image
import os
import glob

transform = utils.get_transform()

data_Pre = utils.PreSavedTiles_MILdataset(transform=transform)

data_WSI = utils.WSI_MILdataset(transform=transform)


for i in range(len(data_Pre)):
    data_Pre[i]
    print()
    data_WSI[i]
    print()

print('Done!')

