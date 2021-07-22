import openslide
from tqdm import tqdm
import numpy as np
from utils import get_optimal_slide_level
import pandas as pd
from glob import glob
import os
import sys

REMOVABLE = True

if sys.platform == 'darwin':
    ndpi_slide_main_dir = r'All Data/ABCTB'
    tif_slide_main_dir = r'All Data/ABCTB_TIF'
    ndpi_slide_data_filename = r'slides_data_ABCTB_conversion.xlsx'
    tif_slide_data_filename = r'slides_data_ABCTB_TIF.xlsx'
    if REMOVABLE:
        ndpi_slide_main_dir = r'/Volumes/McKinley/ABCTB'
        tif_slide_main_dir = r'/Volumes/HD_5TB/Data/ABCTB_TIF'
        slide_data_path = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/'
        ndpi_slide_data_filename = r'slides_data_ABCTB_with_batch_num.xlsx'
        tif_slide_data_filename = r'slides_data_ABCTB_TIF.xlsx'
elif sys.platform == 'linux':
    ndpi_slide_main_dir = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
    tif_slide_main_dir = r'/mnt/gipmed_new/Data/ABCTB_TIF'
    slide_data_path = tif_slide_main_dir
    ndpi_slide_data_filename = r'slides_data_ABCTB.xlsx'
    tif_slide_data_filename = r'slides_data_ABCTB_TIF.xlsx'

tile_size = 1000

tif_slide_filenames = glob(os.path.join(tif_slide_main_dir, '*.tif'))

ndpi_slide_data_DF = pd.read_excel(os.path.join(ndpi_slide_main_dir, ndpi_slide_data_filename))
try:
    tif_slide_data_DF = pd.read_excel(os.path.join(tif_slide_main_dir, tif_slide_data_filename))
except FileNotFoundError:
    tif_slide_data_DF = pd.DataFrame()


for tif_filename in tif_slide_filenames:
    ndpi_filename = '.'.join(os.path.join(ndpi_slide_main_dir, tif_filename.split('/')[-1]).split('.')[:-1]) + '.ndpi'
    ndpi_slide = openslide.open_slide(ndpi_filename)
    tif_slide = openslide.open_slide(tif_filename)

    basic_slide_name = ndpi_filename.split('/')[-1]

    # Get data from original DataFrame which will later be saved in new DataFrame for TIF slides:
    data_for_DF = {}
    data_for_DF['file'] = tif_filename.split('/')[-1]
    data_for_DF['patient barcode'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['patient barcode'].item()
    data_for_DF['id'] = 'ABCTB_TIF'
    data_for_DF['DX'] = True
    data_for_DF['ER status'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['ER status'].item()
    data_for_DF['PR status'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['PR status'].item()
    data_for_DF['Her2 status'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['Her2 status'].item()
    data_for_DF['test fold idx breast'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['test fold idx breast'].item()
    data_for_DF['test fold idx'] = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['test fold idx'].item()
    data_for_DF['Manipulated Objective Power'] = 10
    data_for_DF['Width'] = tif_slide.dimensions[0]
    data_for_DF['Height'] = tif_slide.dimensions[1]

    slide_magnification = ndpi_slide_data_DF[ndpi_slide_data_DF['file'] == basic_slide_name]['Manipulated Objective Power'].item()


    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(ndpi_slide,
                                                                                      magnification=slide_magnification,
                                                                                      desired_mag=10,
                                                                                      tile_size=tile_size)
    ndpi_width, ndpi_height = ndpi_slide.dimensions
    ndpi_grid = [(col, row) for row in range(0, ndpi_height, level_0_tile_size) for col in range(0, ndpi_width, level_0_tile_size)]

    tif_width, tif_height = tif_slide.dimensions

    tif_grid = [(col, row) for row in range(0, tif_height, tile_size) for col in range(0, tif_width, tile_size)]
    diff_sum = 0

    #shift = 64

    for idx in tqdm(range(len(tif_grid))):
        ndpi_location = ndpi_grid[idx]
        tif_location = tif_grid[idx]

        #ndpi_location = (ndpi_location[0] + shift * 4, ndpi_location[1] + shift * 4)
        #tif_location = (tif_location[0] + shift, tif_location[1] + shift)

        ndpi_tile = ndpi_slide.read_region(location=ndpi_location, level=best_slide_level, size=(adjusted_tile_size, adjusted_tile_size)).convert('RGB')
        tif_tile = tif_slide.read_region(location=tif_location, level=0, size=(tile_size, tile_size)).convert('RGB')

        ndpi_tile_array = np.array(ndpi_tile)
        tif_tile_array = np.array(tif_tile)

        diff_sum += np.sum(abs(ndpi_tile_array - tif_tile_array))

    data_for_DF['Diff from ndpi Slide'] = diff_sum
    tif_slide_data_DF = tif_slide_data_DF.append(data_for_DF, ignore_index=True)
    tif_slide_data_DF.to_excel(os.path.join(tif_slide_main_dir, tif_slide_data_filename))
    print('For slide {}, Difference: {}'.format(tif_filename, diff_sum))
tif_slide_data_DF['DX'] = tif_slide_data_DF['DX'].astype('bool')
tif_slide_data_DF['Height'] = tif_slide_data_DF['Height'].astype('int')
tif_slide_data_DF['Width'] = tif_slide_data_DF['Width'].astype('int')
tif_slide_data_DF['Manipulated Objective Power'] = tif_slide_data_DF['Manipulated Objective Power'].astype('int')

tif_slide_data_DF.set_index('file', inplace=True)
tif_slide_data_DF.to_excel(os.path.join(tif_slide_main_dir, tif_slide_data_filename))
print('Done')
