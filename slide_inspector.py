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
parser.add_argument('--in_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_4/CARMEL4', type=str, help='input dir')
parser.add_argument('--out_dir', default=r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_4/thumbs', type=str, help='output dir')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')

args = parser.parse_args()
in_dir = args.in_dir
out_dir = args.out_dir
rewrite_figs = True


def slide_2_image(slide_file, ind, mag, n_legit_tiles, desired_mag):
    fn = os.path.basename(slide_file)[:-5]
    seg_file = os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.jpg')
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
    #im_low.save(os.path.join(out_dir, str(ind).zfill(4) +'_0_thumb_' + fn + '.jpg'), 'JPEG')

    copyfile(os.path.join(in_dir, 'SegData', 'Thumbs', fn + '_thumb.jpg'), os.path.join(out_dir, str(ind).zfill(4) + '_0_thumb_' + fn + '.jpg'))

    #seg image
    #copyfile(os.path.join(in_dir, 'SegData', 'SegImages', fn + '_SegImage.jpg'), os.path.join(out_dir, str(ind).zfill(4) + '_1_SegImage_' + fn + '.jpg'))
    # RanS 10.3.21 - grid image
    copyfile(os.path.join(in_dir, 'SegData', 'GridImages', fn + '_GridImage.jpg'), os.path.join(out_dir, str(ind).zfill(4) + '_1_GridImage_' + fn + '.jpg'))

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

    tiles, time_list = _choose_data(grid_list, slide, n_patches, mag, patch_size, False, desired_mag)

    for ii in range(n_patches):
        grid[ii].imshow(tiles[ii])
        grid[ii].set_yticklabels([])
        grid[ii].set_xticklabels([])

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
        mag = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn, 'Manipulated Objective Power'].item()
        try:
            n_legit_tiles = meta_data_DF.loc[meta_data_DF['patient barcode'] == fn,
                                             'Legitimate tiles - 256 compatible @ X' + str(args.mag)].values[0]
        except:
            print('fn:', fn)
            n_legit_tiles = -1
        slide_2_image(file, ind, mag, n_legit_tiles, args.mag)
    ind += 1

print('finished')