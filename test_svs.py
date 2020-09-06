import openslide
import PIL
import numpy as np
import utils
from openslide import Image
import os
import glob
"""
file_20 = 'good images/00083f2c-c6aa-4e5f-9af0-945d8c9ff97b'
file_40 = 'good images/00cd9b79-7c3a-4192-b030-b8a7fb398c31'

slide_20 = openslide.open_slide(glob.glob(os.path.join(file_20, '*.svs'))[0])
slide_40 = openslide.open_slide(glob.glob(os.path.join(file_40, '*.svs'))[0])

#thumb_20 = slide_20.get_thumbnail((2000, 2000)).show()
#thumb_40 = slide_40.get_thumbnail((2000, 2000)).show()

slide_20.read_region((11000, 1500), 0, (1000, 1000)).show()
slide_40.read_region((11000, 8000),0 , (1000, 1000)).show()
"""

file = 'tcga-data/739cd23e-8d2b-43bf-9a97-88bddc7eff40'
slide = openslide.open_slide(glob.glob(os.path.join(file, '*.svs'))[0])
thumb = slide.get_thumbnail((2000, 2000)).show()
print('Done!')

