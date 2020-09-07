import openslide
import PIL
import numpy as np
import utils
from openslide import Image
import os
import glob

transform = utils.get_transform()
data = utils.WSI_MILdataset(transform=None)

data[5]

print('Done!')

