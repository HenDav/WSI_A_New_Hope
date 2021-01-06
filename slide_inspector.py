import os
import openslide
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import ImageGrid
from utils import _choose_data
import numpy as np
from shutil import copyfile
import cv2 as cv
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Slide inspector')
parser.add_argument('--in_dir', default=r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_3\CARMEL3', type=str, help='input dir')
parser.add_argument('--out_dir', default=r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_3\thumbs', type=str, help='output dir')

args = parser.parse_args()
#in_dir = r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_3\CARMEL3'
#out_dir = r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_3\thumbs'
in_dir = args.in_dir
out_dir = args.out_dir
rewrite_figs = True

def slide_2_image(slide_file, ind, mag, n_legit_tiles):
    fn = os.path.basename(slide_file)[:-5]
    seg_file = os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.png')
    segmap_file = os.path.join(in_dir, 'SegData', 'SegMaps', fn + '_SegMap.png')
    grid_file = os.path.join(in_dir, 'Grids', fn + '--tlsz256' + '.data')
    if not os.path.isfile(seg_file) or not os.path.isfile(grid_file):
        return

    img = openslide.open_slide(slide_file)
    level_1 = img.level_count-5
    loc_x = int(img.properties['openslide.bounds-x'])
    loc_y = int(img.properties['openslide.bounds-y'])
    size_x_low = int(int(img.properties['openslide.bounds-width']) / (2 ** level_1))
    size_y_low = int(int(img.properties['openslide.bounds-height']) / (2 ** level_1))
    im_low = (img.read_region(location=(loc_x, loc_y), level=level_1, size=(size_x_low, size_y_low))).convert('RGB')
    im_low.save(os.path.join(out_dir, str(ind).zfill(4) +'_0_thumb_' + fn + '.jpg'), 'JPEG')

    #RanS 28.12.20 - seg image
    copyfile(os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.png'), os.path.join(out_dir, str(ind).zfill(4) + '_1_SegImage_' + fn + '.png'))

    #get random patches
    fig = plt.figure()
    fig.set_size_inches(32, 18)
    #grid_shape = (2, 4)
    #grid_shape = (8, 16)
    grid_shape = (4, 8)
    n_patches = int(np.prod(grid_shape))
    grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0)
    w, h = img.level_dimensions[0]
    #patch_size = 1024
    patch_size = 256

    n_patches = np.minimum(n_patches, n_legit_tiles)

    plot_seg_map = False
    if plot_seg_map:
        tiles, time_list, locs = _choose_data(grid_file, slide_file, n_patches, mag, patch_size)
    else:
        tiles, time_list = _choose_data(grid_file, slide_file, n_patches, mag, patch_size)

    # RanS 31.12.20
    if plot_seg_map:
        segmap = cv.imread(segmap_file)
        fig2 = plt.figure()
        fig2.set_size_inches(32, 18)
        grid2 = ImageGrid(fig2, 111, nrows_ncols=grid_shape, axes_pad=0)
        objective_pwr = 20
        locs_segmap = np.array(locs)/objective_pwr

    for ii in range(n_patches):
        grid[ii].imshow(tiles[ii])
        grid[ii].set_yticklabels([])
        grid[ii].set_xticklabels([])
        if plot_seg_map:
            grid2[ii].imshow(segmap[int(locs_segmap[ii, 0]):int(locs_segmap[ii, 0]+patch_size/objective_pwr), int(locs_segmap[ii, 1]):int(locs_segmap[ii, 1]+patch_size/objective_pwr), :])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, str(ind).zfill(4) +'_2_patches_' + fn + '.jpg'))
    plt.close()


if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

slide_files_mrxs = np.sort(glob.glob(os.path.join(in_dir, '*.mrxs')))

meta_data_file = os.path.join(os.path.dirname(in_dir), 'slides_data.xlsx')
meta_data_DF = pd.read_excel(meta_data_file)
all_magnifications = list(meta_data_DF['Manipulated Objective Power'])

ind = 0
for _, file in enumerate(tqdm(slide_files_mrxs)):
    fn = os.path.basename(file)[:-5]
    out_path = os.path.join(out_dir, fn + '.jpg')


    if not os.path.isfile(out_path) or rewrite_figs:
        #try:
        mag = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn, 'Manipulated Objective Power']
        try:
            n_legit_tiles = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn, 'Legitimate tiles - 256 compatible @ X20'].values[0]
        except:
            print('fn:', fn)
            n_legit_tiles = -1
        #print('n_legit_tiles:', str(n_legit_tiles))
        slide_2_image(file, ind, mag, n_legit_tiles)
        #except:
        #    print('failed on slide ', fn)
    ind += 1

print('finished')