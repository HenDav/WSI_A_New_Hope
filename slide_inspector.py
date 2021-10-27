import os
import openslide
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import _choose_data
import numpy as np
from shutil import copyfile
import argparse
import pandas as pd
import pickle

parser = argparse.ArgumentParser(description='Slide inspector')
parser.add_argument('--in_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_6/CARMEL6', type=str, help='input dir')
parser.add_argument('--out_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_6/thumbs', type=str, help='output dir')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--grid_only', action='store_true', help='plot grid images only')
parser.add_argument('--grid_path_name', default='', type=str, help='extension of grid_images path') #RanS 3.6.21

args = parser.parse_args()
in_dir = args.in_dir
out_dir = args.out_dir
rewrite_figs = True

if args.grid_path_name != '':
    grid_image_path = os.path.join(in_dir, 'SegData', 'GridImages_' + args.grid_path_name)
else:
    grid_image_paths = glob.glob(os.path.join(in_dir, 'SegData', 'GridImages*'))
    if len(grid_image_paths) > 1:
        print('more than one GridImages Folder! select one')
    else:
        grid_image_path = grid_image_paths[0]

def slide_2_image(slide_file, ind, mag, n_legit_tiles, desired_mag, grid_only):
    #fn = os.path.basename(slide_file)[:-5]
    fn = os.path.splitext(os.path.basename(slide_file))[0] #RanS 9.5.21
    #seg_file = os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.jpg')
    success_flag = True
    if not grid_only:
        grid_file = os.path.join(in_dir, 'Grids_' + str(args.mag), fn + '--tlsz256' + '.data')
        if not os.path.isfile(grid_file):
            print('no grid file')
            return -1
        #img = openslide.open_slide(slide_file)
        #level_1 = img.level_count-5
        #loc_x = int(img.properties['openslide.bounds-x'])
        #loc_y = int(img.properties['openslide.bounds-y'])
        #size_x_low = int(int(img.properties['openslide.bounds-width']) / (2 ** level_1))
        #size_y_low = int(int(img.properties['openslide.bounds-height']) / (2 ** level_1))
        #im_low = (img.read_region(location=(loc_x, loc_y), level=level_1, size=(size_x_low, size_y_low))).convert('RGB')
        #im_low.save(os.path.join(out_dir, str(ind).zfill(4) +'_0_thumb_' + fn + '.jpg'), 'JPEG')

        #get random patches
        fig = plt.figure()
        fig.set_size_inches(32, 18)
        grid_shape = (4, 8)
        n_patches = int(np.prod(grid_shape))
        grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0)
        patch_size = 256

        n_patches = np.minimum(n_patches, n_legit_tiles)

        #plot_seg_map = False
        slide = openslide.open_slide(slide_file)
        with open(grid_file, 'rb') as filehandle:
            grid_list = pickle.load(filehandle)

        if n_patches == -1:
            print('no valid patches found for slide ', fn)
            success_flag = False

        tiles, time_list, _ = _choose_data(grid_list, slide, n_patches, mag, patch_size, False, desired_mag, False, False)

        for ii in range(n_patches):
            grid[ii].imshow(tiles[ii])
            grid[ii].set_yticklabels([])
            grid[ii].set_xticklabels([])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, str(ind).zfill(4) +'_2_patches_' + fn + '.jpg'))
        plt.close()

    #thumb image
    if os.path.isfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.jpg')):
        copyfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.jpg'),
                 os.path.join(out_dir, str(ind).zfill(4) + '_0_thumb_' + fn + '.jpg'))
    elif os.path.isfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.png')): #old format
        copyfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.png'),
                 os.path.join(out_dir, str(ind).zfill(4) + '_0_thumb_' + fn + '.png'))
    else:
        print('no thumb image found for slide ' + fn)
        success_flag = False
    # seg image
    # copyfile(os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.jpg'), os.path.join(out_dir, str(ind).zfill(4) + '_1_SegImage_' + fn + '.jpg'))
    # RanS 10.3.21 - grid image

    # thumb image
    if os.path.isfile(os.path.join(grid_image_path, fn + '_GridImage.jpg')):
        copyfile(os.path.join(grid_image_path, fn + '_GridImage.jpg'),
                 os.path.join(out_dir, str(ind).zfill(4) + '_1_GridImage_' + fn + '.jpg'))
    else:
        print('no grid image found for slide ' + fn)
        success_flag = False

    if success_flag:
        return fn
    else:
        return -1

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

#slide_files_mrxs = np.sort(glob.glob(os.path.join(in_dir, '*.mrxs')))
slide_files_svs = glob.glob(os.path.join(in_dir, '*.svs'))
slide_files_ndpi = glob.glob(os.path.join(in_dir, '*.ndpi'))
slide_files_mrxs = glob.glob(os.path.join(in_dir, '*.mrxs'))
slide_files_tiff = glob.glob(os.path.join(in_dir, '*.tiff'))
slide_files_tif = glob.glob(os.path.join(in_dir, '*.tif'))
slides = np.sort(slide_files_svs + slide_files_ndpi + slide_files_mrxs + slide_files_tiff + slide_files_tif)

#meta_data_file = os.path.join(os.path.dirname(in_dir), 'slides_data.xlsx')
dataset = os.path.basename(in_dir)
slide_meta_data_file = os.path.join(in_dir, 'slides_data_' + dataset + '.xlsx') #RanS 24.3.21
grid_meta_data_file = os.path.join(in_dir, 'Grids_' + str(args.mag), 'Grid_data.xlsx')
slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
#meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
#                             **grid_meta_data_DF.set_index('file').to_dict()})
meta_data_DF = pd.merge(slide_meta_data_DF, grid_meta_data_DF, on="file") #RanS 4.5.21


#meta_data_DF = pd.read_excel(meta_data_file)
all_magnifications = list(meta_data_DF['Manipulated Objective Power'])

ind = 0
fn_list = []

for _, file in enumerate(tqdm(slides)):
    fn_full = os.path.basename(file)
    fn = fn_full[:-5]

    out_path = os.path.join(out_dir, fn + '.jpg')

    if not os.path.isfile(out_path) or rewrite_figs:
        #mag = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn, 'Manipulated Objective Power'].item()

        if args.grid_only:
            mag = 0
            n_legit_tiles = 0
        else:
            try:
                mag = meta_data_DF.loc[meta_data_DF['file'] == fn_full, 'Manipulated Objective Power'].item()
                #n_legit_tiles = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn,
                n_legit_tiles = meta_data_DF.loc[meta_data_DF['file'] == fn_full,
                                                 'Legitimate tiles - 256 compatible @ X' + str(args.mag)].values[0]
            except:
                print('fn:', fn, ' had problem with slides data (multiple identical filenames?)')
                print("meta_data_DF.loc[meta_data_DF['file'] == fn_full, 'Manipulated Objective Power']:") #temp
                print(meta_data_DF.loc[meta_data_DF['file'] == fn_full, 'Manipulated Objective Power']) #temp
                n_legit_tiles = -1

        fn = slide_2_image(file, ind, mag, n_legit_tiles, args.mag, args.grid_only)
        if fn != -1:
            fn_list.append(fn)
    ind += 1

#save successful slides, RanS 9.5.21
fn_list_df = pd.DataFrame(fn_list)
fn_list_df.to_excel(os.path.join(out_dir, 'slide_review_list.xlsx'))
print('finished')