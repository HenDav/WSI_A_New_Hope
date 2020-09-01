import openslide
import PIL
import numpy as np
import utils

file = 'tcga-data/0e40ab7d-657e-4cdb-8f80-5fe3c7dc2779/TCGA-E2-A1BC-11A-03-TSC.df3e3516-968b-43e9-992e-93f73e758963.svs'

#utils.make_slides_xl_file()

#utils.make_segmentations()

ds = utils.WSI_MILdataset(256)
print('Done!')
