import pyvips
import openslide
import numpy as np
from utils import get_optimal_slide_level
from tqdm import tqdm
import shutil
import subprocess
import os
import pandas as pd
from utils import get_cpu
import sys
import multiprocessing
from functools import partial


if sys.platform == 'darwin':
    original_path = r'All Data/ABCTB'
    slide_data_filename = r'slides_data_ABCTB_conversion.xlsx'
    new_slides_path = r'All Data/ABCTB_TIF'

elif sys.platform == 'linux':
    original_path = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
    slide_data_filename = r'slides_data_ABCTB.xlsx'
    #new_slides_path = r'/home/womer/project/All Data/ABCTB_TIF'
    new_slides_path = r'/mnt/gipmed/All_Data/ABCTB_TIF'


def convert_1_slide(slide_name):
    '''
    if sys.platform == 'darwin':
        original_path = r'All Data/ABCTB'
        slide_data_filename = r'slides_data_ABCTB_conversion.xlsx'
        new_slides_path = r'All Data/ABCTB_TIF'

    elif sys.platform == 'linux':
        original_path = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
        slide_data_filename = r'slides_data_ABCTB.xlsx'
        # new_slides_path = r'/home/womer/project/All Data/ABCTB_TIF'
        new_slides_path = r'/mnt/gipmed/All_Data/ABCTB_TIF'
    '''

    slidename = os.path.join(original_path, slide_name)
    division_slide_size = 512
    slide_tile_size = 512  # 1024
    slide = openslide.open_slide(slidename)
    slide_data_DF = pd.read_excel(os.path.join(original_path, slide_data_filename))
    slide_magnification = slide_data_DF[slide_data_DF['file'] == slide_name]['Manipulated Objective Power'].item()

    slide_basic_name = '.'.join(slide_name.split('.')[:-1])
    if not os.path.isdir(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name)):
        os.mkdir(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name))

    if create_thumbnails:
        tmb = slide.get_thumbnail((1000, 1000))
        tmb.save(os.path.join(new_slides_path, 'TIF_Thumbs', slide_basic_name + '_thumb_ndpi.png'))

    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide,
                                                                                      magnification=slide_magnification,
                                                                                      desired_mag=10,
                                                                                      tile_size=division_slide_size)

    slide_width, slide_height = slide.dimensions
    #slide_width, slide_height = slide.level_dimensions[best_slide_level]

    grid = [(col, row) for row in range(0, slide_height, level_0_tile_size) for col in range(0, slide_width, level_0_tile_size)]
    num_cols = int(np.ceil(slide_width / level_0_tile_size))
    num_tiles = len(grid)

    #print('Dividing slide into tiles')
    #for idx, location in enumerate(tqdm(grid)):
    print('Slidename: {}, Num tiles: {}'.format(slide_basic_name, num_tiles))
    for idx, location in enumerate(tqdm(grid)):
        tile = slide.read_region(location=location, level=best_slide_level, size=(adjusted_tile_size, adjusted_tile_size)).convert('RGB')
        tile_array = np.array(tile)
        linear = tile_array.reshape(adjusted_tile_size * adjusted_tile_size * 3)
        tile_vips = pyvips.Image.new_from_memory(linear.data, adjusted_tile_size, adjusted_tile_size, 3, 'uchar')
        tile_vips.write_to_file(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(idx) + '.vips'))

    #print('Finished slide division')

    tile_list = []
    #print('Starting tile gathering ...')
    for idx, _ in enumerate(tqdm(range(num_tiles))):
        tile_filename = os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(idx) + '.vips')
        tile = pyvips.Image.new_from_file(tile_filename, access='sequential')
        tile_list.append(tile)

    full_image = pyvips.Image.arrayjoin(tile_list, across=num_cols, shim=0)

    #print('Saving vips...')
    if not os.path.isdir(os.path.join(new_slides_path, 'complete_vips_slides')):
        os.mkdir(os.path.join(new_slides_path, 'complete_vips_slides'))

    full_image.write_to_file(os.path.join(new_slides_path, 'complete_vips_slides', slide_basic_name + '.vips'))

    # Delete all tiles:
    shutil.rmtree(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name))


    if True:
        #print('Converting to tilled tif')
        vips_filename = os.path.join(new_slides_path, 'complete_vips_slides', slide_basic_name + '.vips')
        tif_full_command = os.path.join(new_slides_path, slide_basic_name + '.tif' + ':none,tile:512x512')

        try:
            process = subprocess.run(['vips',
                                      'im_vips2tiff',
                                      vips_filename,
                                      tif_full_command],
                                     check=True)
        except:
            print('vips conversion to tilled tif was not succesful')

        os.remove(vips_filename)

    if create_thumbnails:
        tif_filename = os.path.join(new_slides_path, slide_basic_name + '.tif')
        tif_slide = openslide.open_slide(tif_filename)
        tmb_tif = tif_slide.get_thumbnail((1000, 1000))
        tmb_tif.save(os.path.join(new_slides_path, 'TIF_Thumbs', slide_basic_name + '_thumb.png'))

#############################################################################################################
#############################################################################################################
#############################################################################################################


create_thumbnails = True
convert_to_tilled_tif = True
multi = False
num_workers = 4  # get_cpu()

'''
if sys.platform == 'darwin':
    original_path = r'All Data/ABCTB'
    slide_data_filename = r'slides_data_ABCTB_conversion.xlsx'
    new_slides_path = r'All Data/ABCTB_TIF'

elif sys.platform == 'linux':
    original_path = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
    slide_data_filename = r'slides_data_ABCTB.xlsx'
    #new_slides_path = r'/home/womer/project/All Data/ABCTB_TIF'
    new_slides_path = r'/mnt/gipmed/All_Data/ABCTB_TIF'
'''
# Create folders:
if not os.path.isdir(new_slides_path):
    os.mkdir(new_slides_path)
if not os.path.isdir(os.path.join(new_slides_path, 'slide_tiles')):
    os.mkdir(os.path.join(new_slides_path, 'slide_tiles'))
if create_thumbnails and not os.path.isdir(os.path.join(new_slides_path, 'TIF_Thumbs')):
    os.mkdir(os.path.join(new_slides_path, 'TIF_Thumbs'))

slide_data_DF = pd.read_excel(os.path.join(original_path, slide_data_filename))
files = list(slide_data_DF['file'])

if multi:
    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=len(files)) as pbar:
            for i, _ in enumerate(pool.map(partial(convert_1_slide),
                                           files)):
                pbar.update()
else:
    for i, slide_name in enumerate(files):
        convert_1_slide(slide_name)


# Remove folders:
if os.path.isdir(os.path.join(new_slides_path, 'slide_tiles')):
    os.rmdir(os.path.join(new_slides_path, 'slide_tiles'))
if convert_to_tilled_tif:
    if os.path.isdir(os.path.join(new_slides_path, 'complete_vips_slides')):
        os.rmdir(os.path.join(new_slides_path, 'complete_vips_slides'))

print('Done')
