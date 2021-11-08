import pandas as pd
import utils
import os
import openslide
import sys
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Extract tiles from excel')
parser.add_argument('-o', dest='original', action='store_true', help='Extract original size  ?')
parser.add_argument('-mag', dest='desired_magnification', type=int, default=10, help='Desired Magnification for tiles')
parser.add_argument('-png', dest='png', action='store_true', help='Save as .png ?')
args = parser.parse_args()

TILE_SIZE = 256

if sys.platform == 'darwin':
    highest_DF = pd.read_excel(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/patches_to_extract_pos.xlsx')
    lowest_DF = pd.read_excel(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/patches_to_extract_neg.xlsx')
elif sys.platform == 'linux':
    highest_DF = pd.read_excel(r'/home/womer/project/Data For Gil/patches_to_extract_pos.xlsx')
    lowest_DF = pd.read_excel(r'/home/womer/project/Data For Gil/patches_to_extract_neg.xlsx')

highest_slide_filenames = list(highest_DF['SlideName'])
highest_tile_indices = list(highest_DF['TileIdx'])
highest_tile_locations = []
for tile_idx in range(len(highest_DF)):
    highest_tile_locations.append((highest_DF['TileLocation1'][tile_idx], highest_DF['TileLocation2'][tile_idx]))

lowest_slide_filenames = list(lowest_DF['SlideName'])
lowest_tile_indices = list(lowest_DF['TileIdx'])
lowest_tile_locations = []
for tile_idx in range(len(lowest_DF)):
    lowest_tile_locations.append([lowest_DF['TileLocation1'][tile_idx], lowest_DF['TileLocation2'][tile_idx]])

# Extracting the tiles:
#dir_dict = utils.get_datasets_dir_dict(Dataset='CARMEL')
# open slides_data.xlsx file:
slide_data_file = r'/home/womer/project/All Data/Ran_Features/Grid_data/slides_data_CARMEL_ALL.xlsx' if sys.platform == 'linux' else r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/slides_data_CARMEL_ALL.xlsx'

slides_meta_data_DF = pd.read_excel(slide_data_file)
slides_meta_data_DF.set_index('file', inplace=True)
batches = {'Highest': [highest_slide_filenames, highest_tile_indices, highest_tile_locations],
           'Lowest': [lowest_slide_filenames, lowest_tile_indices, lowest_tile_locations]}

for key in batches.keys():
    slide_filenames = batches[key][0]
    tile_indices = batches[key][1]
    tile_locations = batches[key][2]

    for file_idx in tqdm(range(len(tile_indices))):
        if sys.platform == 'darwin':
            dir_dict = utils.get_datasets_dir_dict(Dataset='CARMEL')
            image_file = os.path.join(dir_dict['CARMEL'], slide_filenames[file_idx])
        elif sys.platform == 'linux':
            dir_dict = utils.get_datasets_dir_dict(Dataset='CARMEL')
            file_id = slides_meta_data_DF.loc[slide_filenames[file_idx]]['id']
            image_file = os.path.join(dir_dict[file_id], slide_filenames[file_idx])

        tile_location = tile_locations[file_idx]
        tile_index_in_xl = tile_indices[file_idx]


        slide = openslide.open_slide(image_file)
        # Compute level to extract tile from:
        slide_level_0_magnification = slides_meta_data_DF.loc[slide_filenames[file_idx]]['Manipulated Objective Power']

        if args.original:
            magnification_ratio = slide_level_0_magnification // args.desired_magnification
            desired_magnification = slide_level_0_magnification
            tile_size = TILE_SIZE * magnification_ratio
        else:
            desired_magnification = args.desired_magnification
            tile_size = TILE_SIZE

        best_slide_level, adjusted_tile_size, level_0_tile_size = utils.get_optimal_slide_level(slide=slide,
                                                                                                magnification=slide_level_0_magnification,
                                                                                                desired_mag=desired_magnification,
                                                                                                tile_size=tile_size)

        # Get the tiles:
        image_tiles, _, _ = utils._get_tiles(slide=slide,
                                             locations=[tile_location],
                                             tile_size_level_0=level_0_tile_size,
                                             adjusted_tile_sz=adjusted_tile_size,
                                             output_tile_sz=tile_size,
                                             best_slide_level=best_slide_level
                                             )

        # Save tile:
        if not os.path.isdir('Data For Gil'):
            os.mkdir('Data For Gil')

        tile_number = (8 - len(str(tile_index_in_xl))) * '0' + str(tile_index_in_xl)
        #tile_filename = os.path.join('Data For Gil', str(desired_magnification) + '_' + tile_number)
        tile_filename = os.path.join('Data For Gil', tile_number)
        tile_filename_extension = '.png' if args.png else '.jpg'
        image_tiles[0].save(tile_filename + tile_filename_extension)


print('Done')