import pandas as pd
import matplotlib.pyplot as plt
import openslide
import numpy as np
import cv2
import os
from utils import get_optimal_slide_level, get_datasets_dir_dict
import glob
import argparse
import re, sys


def get_slide_name(file):
    start = re.search("Inference_Full_Slide_", file)
    end = re.search("_ScoreHeatMap.xlsx", file)
    return file[start.span()[1]:end.span()[0]]


parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-dn', type=str,
                    default=r'/home/rschley/code/WSI_MIL/general_try4/Inference/Full_Slide_Inference/TCGA_LUNG/exp375_epoch1000',
                    help='Heatmap directory name')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA_LUNG', help='DataSet to use')
parser.add_argument('--binary', action='store_true', help='show binary map')
parser.add_argument('--calib', action='store_true', help='show calibrated map')
parser.add_argument('--superimpose', action='store_true', help='superimpose the heatmap on top of the slide')
parser.add_argument('--vmin', default=0, type=float, help='minimum color value in map')
args = parser.parse_args()

if sys.platform == 'win32':
    # override args, manually define everything
    dn = r'C:\ran_data\TCGA_lung\heatmaps\test'
    file_list = [
        r'C:\ran_data\TCGA_lung\heatmaps\300821\is_cancer_BatchOfSlides_Exp_375_Epoch_400_Inference_Full_Slide_TCGA-05-5420-11A-01-TS1.062c76b9-163d-4a4a-963d-ca2d56bddaa7.svs_ScoreHeatMap.xlsx']
    slides_dir = r'C:\ran_data\TCGA_lung\TCGA_LUNG'
    calib = False
    binary = False
    args.vmin = 0.9
else:
    dn = args.dn
    file_list = glob.glob(os.path.join(dn, '*ScoreHeatMap.xlsx'))
    dataset_dir_dict = get_datasets_dir_dict(Dataset=args.dataset)
    slides_dir = dataset_dir_dict[args.dataset]
    calib = args.calib
    binary = args.binary

if not os.path.isdir(os.path.join(dn, 'out')):
    os.mkdir(os.path.join(dn, 'out'))

for file in file_list:
    slide_name = get_slide_name(file)
    slide_file = os.path.join(slides_dir, slide_name)

    if args.superimpose and not os.path.isdir(os.path.join(dn, 'out', slide_name)):
        os.mkdir(os.path.join(dn, 'out', slide_name))

    heatmap_DF = pd.read_excel(file)
    heatmap_DF.drop('Unnamed: 0', inplace=True, axis=1)
    heatmap = heatmap_DF.to_numpy()
    heatmap[heatmap == -1] = np.nan

    if args.binary:
        heatimage = heatmap
        heatimage[heatmap >= args.vmin] = 1
        heatimage[heatmap < args.vmin] = 0
    elif calib:
        heatmap_vec = np.reshape(heatmap, (heatmap.size))
        ind_sorted = np.argsort(heatmap_vec)
        heatmap_vec[ind_sorted] = np.linspace(0, 1, heatmap.size)
        heatmap_calib = np.reshape(heatmap_vec, heatmap.shape)
        heatmap_calib[np.isnan(heatmap)] = np.nan
        heatmap_calib = (heatmap_calib - np.nanmin(heatmap_calib)) / (np.nanmax(heatmap_calib) - np.nanmin(heatmap_calib))
        heatimage = heatmap_calib
    else:
        heatimage = heatmap

    slide = openslide.OpenSlide(slide_file)
    height = slide.dimensions[1]
    width = slide.dimensions[0]

    if args.superimpose:
        mag_dict = {'.svs': 'aperio.AppMag', '.ndpi': 'hamamatsu.SourceLens', '.mrxs': 'openslide.objective-power',
                    'tiff': 'tiff.Software'}  # RanS 25.3.21, dummy for TIFF
        _, data_format = os.path.splitext(os.path.basename(slide_file))
        objective_pwr = int(float(slide.properties[mag_dict[data_format]]))
        if slide_name[-5:] == '.mrxs':
            objective_pwr = 20  # RanS 7.12.21, value in slide is wrong
        magnification = 10
        ds = objective_pwr // magnification
        height_ds = int(height / ds)
        width_ds = int(width / ds)
        patch_size = 2048
        N_patches_y = int(height_ds / patch_size)
        N_patches_x = int(width_ds / patch_size)
        heatmap_resized = cv2.resize(heatimage, dsize=(width_ds, height_ds), interpolation=cv2.INTER_NEAREST)
        best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide, objective_pwr,
                                                                                          magnification, patch_size)
    my_cmap = 'jet'

    # take a 10000 pixel high thumb
    mag_thumb = 5
    height_thumb = np.min((int(height / mag_thumb), 10000))
    #height_thumb = 10000
    width_thumb = int(width / height * height_thumb)
    thumb = slide.get_thumbnail((width_thumb, height_thumb))
    heatmap_thumb = cv2.resize(heatimage, dsize=(width_thumb, height_thumb), interpolation=cv2.INTER_NEAREST)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.imshow(thumb)
    sp = ax2.imshow(heatmap_thumb, vmin=args.vmin, vmax=1)
    ax1.axis('off')
    ax2.axis('off')
    if not args.binary:
        fig.colorbar(sp)
    if args.superimpose:
        fig.savefig(os.path.join(dn, 'out', slide_name, 'thumb.jpg'), bbox_inches='tight', dpi=1000)
    else:
        fig.savefig(os.path.join(dn, 'out', os.path.splitext(slide_name)[0] + '.jpg'), bbox_inches='tight', dpi=1000)
    plt.close(fig)
    if args.superimpose:
        for i_y in range(N_patches_y):
            for i_x in range(N_patches_x):
                im = slide.read_region((i_x * patch_size * ds, i_y * patch_size * ds), best_slide_level,
                                       (adjusted_tile_size, adjusted_tile_size)).convert('RGB')
                if adjusted_tile_size != patch_size:
                    im = im.resize((patch_size, patch_size))
                heat_im = heatmap_resized[i_y * patch_size:(i_y + 1) * patch_size,
                          i_x * patch_size:(i_x + 1) * patch_size]
                if np.any(heat_im < 1):  # at least one patch
                    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
                    ax.imshow(im, vmin=args.vmin, vmax=1)
                    sp = ax.imshow(heat_im, alpha=0.2, cmap=my_cmap)
                    ax.axis('off')
                    if not args.binary:
                        fig.colorbar(sp)
                    fig.savefig(os.path.join(dn, 'out', slide_name, str(i_x) + '_' + str(i_y) + '.jpg'),
                                bbox_inches='tight', dpi=1000)
                    plt.close(fig)
                print(1)

    print('Finished file ' + slide_name)
