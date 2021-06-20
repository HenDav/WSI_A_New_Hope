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
import resource



if sys.platform == 'darwin':
    original_path = r'All Data/ABCTB'
    slide_data_filename = r'slides_data_ABCTB_conversion.xlsx'
    new_slides_path = r'All Data/ABCTB_TIF'

    resource.setrlimit(resource.RLIMIT_NOFILE, (20480, -1))
    MAX_FILES, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

elif sys.platform == 'linux':
    original_path = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
    slide_data_filename = r'slides_data_ABCTB.xlsx'
    new_slides_path = r'/mnt/gipmed_new/Data/ABCTB_TIF'

    try:
        process = subprocess.run(['ulimit',
                                  '-n',
                                  str(65536)],
                                 check=True)
    except:
        print('Could not define new Max file number')

    #resource.setrlimit(resource.RLIMIT_NOFILE, (65536, -1))
    MAX_FILES, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

print('Max file number is: {}'.format(MAX_FILES))


def connect_tiles(slide_basic_name, num_tiles, num_cols):
    if num_tiles <= MAX_FILES:
        if print_comments:
            print('Concatenate all tiles in 1 phase')
        tile_list = []
        for idx, _ in enumerate(range(num_tiles)):
            tile_filename = os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(idx) + '.vips')
            tile = pyvips.Image.new_from_file(tile_filename, access='sequential')
            tile_list.append(tile)

        full_image = pyvips.Image.arrayjoin(tile_list, across=num_cols, shim=0)

    else:  # If there are more than MAX_FILES than the tile concatenation will be done in parts.
        if print_comments:
            print('Concatenate all tiles in 2 phases')
        tiles_per_file = num_cols * (MAX_FILES // num_cols)
        tiles_per_row = num_cols
        tiles_per_file = tiles_per_row
        num_files = int(np.ceil(num_tiles / tiles_per_file))

        if not os.path.isfile(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name,
                                           'vertical_image_tile_' + str(num_files - 1) + '.vips')):
            tile_list = []
            file_counter = 0
            for idx, _ in enumerate(range(num_tiles)):
                tile_filename = os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(idx) + '.vips')
                tile = pyvips.Image.new_from_file(tile_filename, access='sequential')
                tile_list.append(tile)
                if len(tile_list) == tiles_per_file or idx == num_tiles - 1:  # Save this part of the full image
                    image_part = pyvips.Image.arrayjoin(tile_list, across=num_cols, shim=0)
                    image_part.write_to_file(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, 'vertical_image_tile_' + str(file_counter) + '.vips'))

                    tile_list = []
                    file_counter += 1

        # Now we'll have to concatenate all the vertical image tiles:
        if not os.path.isfile(os.path.join(new_slides_path, 'complete_vips_slides', slide_basic_name + '.vips')):
            vertical_image_parts_list = []
            for idx, _ in enumerate(range(num_files)):
                vertical_image_filename = os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, 'vertical_image_tile_' + str(idx) + '.vips')
                vertical_image_part = pyvips.Image.new_from_file(vertical_image_filename, access='sequential')
                vertical_image_parts_list.append(vertical_image_part)

        full_image = pyvips.Image.arrayjoin(vertical_image_parts_list, across=1, shim=0)
        if print_comments:
            print('Finished tile concatenation')

    if print_comments:
        print('Saving vips...')
    if not os.path.isdir(os.path.join(new_slides_path, 'complete_vips_slides')):
        os.mkdir(os.path.join(new_slides_path, 'complete_vips_slides'))

    full_image.write_to_file(os.path.join(new_slides_path, 'complete_vips_slides', slide_basic_name + '.vips'))

    if print_comments:
        print('Finished Saving !')

    # Delete all tiles:
    if print_comments:
        print('Deleting all slide tiles ...')
    try:
        shutil.rmtree(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name))
        if print_comments:
            print('Finished deletion')
    except OSError:
        try:
            os.rmdir(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name))
            print('Delete Successful')
        except OSError:
            print('Could NOT delete all slide tiles or the folder.')



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
    tif_filename = os.path.join(new_slides_path, slide_basic_name + '.tif')
    if os.path.isfile(tif_filename):  # Check if conversion was already done
        print('TIF Slide Exists')
        return

    if not os.path.isdir(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name)):
        os.mkdir(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name))

    # Create thumbnail for original slide:
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

    print('{} with {} tiles and {} cols. MAX FILES is {}'.format(slide_basic_name, num_tiles, num_cols, MAX_FILES))

    if print_comments:
        print('Dividing slide into tiles')
    #for idx, location in enumerate(tqdm(grid)):
    #print('Slidename: {}, Num tiles: {}'.format(slide_basic_name, num_tiles))
    # Check if the slide is already divided into tiles (if yes , than skip to the next phase).
    if not os.path.isfile(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(num_tiles - 1) + '.vips')):
        for idx, location in enumerate(grid):
            tile = slide.read_region(location=location, level=best_slide_level, size=(adjusted_tile_size, adjusted_tile_size)).convert('RGB')
            tile_array = np.array(tile)
            linear = tile_array.reshape(adjusted_tile_size * adjusted_tile_size * 3)
            tile_vips = pyvips.Image.new_from_memory(linear.data, adjusted_tile_size, adjusted_tile_size, 3, 'uchar')
            tile_vips.write_to_file(os.path.join(new_slides_path, 'slide_tiles', slide_basic_name, str(idx) + '.vips'))

        if print_comments:
            print('Finished slide division')

    if print_comments:
        print('Starting tile gathering ...')
    connect_tiles(slide_basic_name, num_tiles, num_cols)
    '''        
    tile_list = []
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
    '''

    if convert_to_tilled_tif:
        if print_comments:
            print('Converting to tilled tif')
        vips_filename = os.path.join(new_slides_path, 'complete_vips_slides', slide_basic_name + '.vips')
        #tif_full_command = os.path.join(new_slides_path, slide_basic_name + '.tif' + ':none,tile:512x512')
        tif_full_command = os.path.join(new_slides_path,
                                        slide_basic_name + '.tif' + ':none,tile:' + str(slide_tile_size) + 'x' + str(slide_tile_size))

        try:
            if sys.platform == 'darwin':
                process = subprocess.run(['vips',
                                          'im_vips2tiff',
                                          vips_filename,
                                          tif_full_command],
                                         check=True)
            elif sys.platform == 'linux':
                process = subprocess.run(['/usr/bin/vips',
                                          'im_vips2tiff',
                                          vips_filename,
                                          tif_full_command],
                                         check=True)

            print('Conversion successful !)')
        except:
            print('vips conversion to tilled tif was not succesful')

        os.remove(vips_filename)

    # Create thumbnail for converted slide:
    tif_filename = os.path.join(new_slides_path, slide_basic_name + '.tif')
    tif_slide = openslide.open_slide(tif_filename)
    tmb_tif = tif_slide.get_thumbnail((1000, 1000))
    tmb_tif.save(os.path.join(new_slides_path, 'TIF_Thumbs', slide_basic_name + '_thumb.png'))

    print('{} FINISHED'.format(slide_basic_name))

#############################################################################################################
#############################################################################################################
#############################################################################################################

print_comments = False
convert_to_tilled_tif = True
multi = True
num_workers = get_cpu()

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
if not os.path.isdir(os.path.join(new_slides_path, 'TIF_Thumbs')):
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
