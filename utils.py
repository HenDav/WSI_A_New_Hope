import cv2 as cv
import numpy as np
from PIL import Image
import os
import openslide
import pandas as pd
import shutil
import glob
from tqdm import tqdm
from typing import List, Tuple
import pickle
from random import sample
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import sys
import time


def make_dir(dirname):
    if not dirname in next(os.walk(os.getcwd()))[1]:
        try:
            os.mkdir(dirname)
        except OSError:
            print('Creation of directory ', dirname, ' failed...')
            raise


def make_tiles_hard_copy(data_path: str = 'tcga-data', tile_size: int = 256, how_many_tiles: int = 500):
    """
    This function makes a hard copy of the tile in order to avoid using openslide
    :param data_path:
    :return:
    """

    dirs = _get_tcga_id_list(data_path)
    meta_data = pd.read_excel(os.path.join(data_path, 'slides_data.xlsx'))

    for i in tqdm(range(meta_data.shape[0])):
        if meta_data['Total tiles - 256 compatible @ X20'][i] == -1:
            continue

        slide_file_name = os.path.join(data_path, meta_data['id'][i], meta_data['file'][i])
        slide_tiles = _choose_data(slide_file_name, how_many_tiles, meta_data['Objective Power'][i], tile_size,
                                   resize=True)

        file_name = os.path.join(data_path, meta_data['id'][i], 'tiles.data')
        with open(file_name, 'wb') as filehandle:
            pickle.dump(slide_tiles, filehandle)


def copy_segImages(data_path: str = 'tcga-data', data_format: str = 'TCGA'):
    """
    This function copies the Segmentation Images from it's original location to one specific location, for easy checking
    of the segmentations later on...
    :return:
    """
    print('Copying Segmentation Images...')
    make_dir('Segmentation_Images')

    if data_format == 'TCGA':
        dirs = _get_tcga_id_list(data_path)
        for _, dir in enumerate(dirs):
            if 'segImage.png' in next(os.walk(os.path.join(data_path, dir)))[2]:
                shutil.copy2(os.path.join(data_path, dir, 'segImage.png'),
                             os.path.join('Segmentation_Images', dir + '_SegImage.png'))
            else:
                print('Found no segImage file for {}'.format(dir))
    elif data_format == 'ABCTB' or data_format == 'MIRAX':
        files = [file for file in os.listdir(data_path) if file.endswith("_segImage.png")]
        for file in files:
            shutil.copy2(os.path.join(data_path, file), os.path.join('Segmentation_Images', file))
        # dirs = _get_tcga_id_list(data_path)
        # for _, dir in enumerate(dirs):
        #    if 'segImage.png' in next(os.walk(os.path.join(data_path, dir)))[2]:

        #    else:
        #        print('Found no segImage file for {}'.format(dir))

    print('Finished copying!')


def compute_normalization_values(data_path: 'str' = 'tcga-data/') -> tuple:
    """
    This function runs over a set of images and compute mean and variance of each channel.
    The function computes these statistic values over the thumbnail images which are at X1 magnification
    :return:
    """

    # get a list of all directories with images:
    dirs = _get_tcga_id_list(data_path)
    stats_list = []
    print('Computing image-set Mean and Variance...')
    meta_data = pd.read_excel(os.path.join(data_path, 'slides_data.xlsx'))
    meta_data.set_index('id', inplace=True)

    # gather tissue image values from thumbnail image using the segmentation map:
    # for idx, dir in enumerate(dirs):
    for i in tqdm(range(len(dirs))):
        dir = dirs[i]
        if meta_data.loc[[dir], ['Total tiles - 256 compatible @ X20']].values[0][0] == -1:
            continue

        image_stats = {}
        thumb = np.array(Image.open(os.path.join(data_path, dir, 'thumb.png')))
        segMap = np.array(Image.open(os.path.join(data_path, dir, 'segMap.png')))
        tissue = thumb.transpose(2, 0, 1) * segMap
        tissue_pixels = (tissue[0] != 0).sum()
        tissue_matter = np.where(tissue[0] != 0)
        values = tissue[:, tissue_matter[0], tissue_matter[1]]
        image_stats['Pixels'] = tissue_pixels
        image_stats['Mean'] = values.mean(axis=1)
        image_stats['Var'] = values.var(axis=1)
        stats_list.append(image_stats)

    # Save data to file:
    with open(os.path.join(data_path, 'ImageStatData.data'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(stats_list, filehandle)

    # Compute total mean and var:
    N = 0
    running_mean = 0
    running_mean_squared = 0
    running_var = 0
    for i, item in enumerate(stats_list):
        n = item['Pixels']
        N += n
        running_mean += item['Mean'] * n
        running_mean_squared += (item['Mean'] ** 2) * n
        running_var += item['Var'] * n

    total_mean = running_mean / N
    total_var = (running_mean_squared + running_var) / N - total_mean ** 2
    print('Finished computing statistical data over {} thumbnail slides'.format(i + 1))
    print('Mean: {}'.format(total_mean))
    print('Variance: {}'.format(total_var))
    return total_mean, total_var


def _choose_data(file_name: str, how_many: int, magnification: int = 20, tile_size: int = 256, resize: bool = False):
    """
    This function choose and returns data to be held by DataSet
    :param file_name:
    :param how_many: how_many describes how many tiles to pick from the whole image
    :return:
    """
    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], basic_grid_file_name)
    with open(grid_file, 'rb') as filehandle:
        # read the data as binary data stream
        grid_list = pickle.load(filehandle)

    # Choose locations from the grid:
    loc_num = len(grid_list)

    idxs = sample(range(loc_num), how_many)
    locs = [grid_list[idx] for idx in idxs]

    #  _save_tile_list_to_file(file_name, idxs)

    # print('File: {}, Tiles: {}'.format(file_name, idxs))

    if resize:
        resize_to = tile_size
    else:
        resize_to = adjusted_tile_size

    image_tiles = _get_tiles(file_name, locs, adjusted_tile_size, resize_to=resize_to)

    return image_tiles


def _get_grid_list(file_name: str, magnification: int = 20, tile_size: int = 256):
    """
    This function returns the grid location of tile for a specific slide.
    :param file_name:
    :return:
    """

    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], basic_grid_file_name)
    with open(grid_file, 'rb') as filehandle:
        # read the data as binary data stream
        grid_list = pickle.load(filehandle)

        return grid_list


def _get_tiles(file_name: str, locations: List[Tuple], tile_sz: int, resize_to: int):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """
    # open the .svs file:
    img = openslide.open_slide(file_name)
    tiles_num = len(locations)
    # TODO: Checking Pil vs. np array. Delete one of them...
    # tiles = np.zeros((tiles_num, 3, tile_sz, tile_sz), dtype=int)  # this line is needed when working with nd arrays
    tiles_PIL = []

    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        image = img.read_region((loc[1], loc[0]), 0, (tile_sz, tile_sz)).convert('RGB')
        # tiles[idx, :, :, :] = np.array(image).transpose(2, 0, 1)  # this line is needed when working with nd arrays
        if tile_sz != resize_to:
            image = image.resize((resize_to, resize_to), resample=Image.BILINEAR)

        tiles_PIL.append(image)

    return tiles_PIL


def make_grid(data_path: str = 'tcga-data', tile_sz: int = 256):
    """
    This function creates a location for all top left corners of the grid
    :param data_file: name of main excel data file containing size of images (this file is created by function :"make_slides_xl_file")
    :param tile_sz: size of tiles to be created
    :return:
    """
    data_file = os.path.join(data_path, 'slides_data.xlsx')

    BASIC_OBJ_PWR = 20

    basic_DF = pd.read_excel(data_file)
    files = list(basic_DF['file'])
    objective_power = list(basic_DF['Objective Power'])
    basic_DF.set_index('file', inplace=True)
    tile_nums = []
    total_tiles = []
    print('Starting Grid production...')
    print()
    # for _, file in enumerate(files):
    for i in tqdm(range(len(files))):
        file = files[i]
        data_dict = {}
        height = basic_DF.loc[file, 'Height']
        width = basic_DF.loc[file, 'Width']

        id = basic_DF.loc[file, 'id']
        if objective_power[i] == 'Missing Data':
            print('Grid was not computed for path {}'.format(id))
            tile_nums.append(0)
            total_tiles.append(-1)
            continue

        converted_tile_size = int(tile_sz * (int(objective_power[i]) / BASIC_OBJ_PWR))
        basic_grid = [(row, col) for row in range(0, height, converted_tile_size) for col in
                      range(0, width, converted_tile_size)]
        total_tiles.append((len(basic_grid)))

        # We now have to check, which tiles of this grid are legitimate, meaning they contain enough tissue material.
        legit_grid = _legit_grid(os.path.join(data_file.split('/')[0], id, 'segMap.png'),
                                 basic_grid,
                                 converted_tile_size,
                                 (height, width))

        # create a list with number of tiles in each file
        tile_nums.append(len(legit_grid))

        # Save the grid to file:
        file_name = os.path.join(data_file.split('/')[0], id, 'grid_tlsz' + str(converted_tile_size) + '.data')
        with open(file_name, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(legit_grid, filehandle)

    # Adding the number of tiles to the excel file:
    basic_DF['Legitimate tiles - ' + str(tile_sz) + ' compatible @ X20'] = tile_nums
    basic_DF['Total tiles - ' + str(tile_sz) + ' compatible @ X20'] = total_tiles
    basic_DF['Slide tile usage [%] (for ' + str(tile_sz) + '^2 Pix/Tile)'] = list(
        ((np.array(tile_nums) / np.array(total_tiles)) * 100).astype(int))
    basic_DF.to_excel(data_file)

    print('Finished Grid production phase !')


def _legit_grid(image_file_name: str, grid: List[Tuple], tile_size: int, size: tuple, coverage: int = 0.5) -> List[
    Tuple]:
    """
    This function gets a .svs file name, a basic grid and tile size and returns a list of legitimate grid locations.
    :param image_file_name: .svs file name
    :param grid: basic grid
    :param tile_size: tile size
    :param size: size of original image (height, width)
    :param coverage: Coverage of tissue to make the slide legitimate
    :return:
    """

    # Check if coverage is a number in the range (0, 1]
    if not (coverage > 0 and coverage <= 1):
        raise ValueError('Coverage Parameter should be in the range (0,1]')

    # open the segmentation map image from which the coverage will be calculated:
    segMap = np.array(Image.open(image_file_name))
    rows = size[0] / segMap.shape[0]
    cols = size[1] / segMap.shape[1]

    # the complicated next line only rounds up the numbers
    small_tile = (int(-(-tile_size // rows)), int(-(-tile_size // cols)))
    # computing the compatible grid for the small segmenatation map:
    idx_to_remove = []
    for idx, (row, col) in enumerate(grid):
        new_row = int(-(-(row // rows)))
        new_col = int(-(-(col // cols)))

        # collect the data from the segMap:
        tile = segMap[new_row: new_row + small_tile[0], new_col: new_col + small_tile[1]]
        tile_pixels = small_tile[0] * small_tile[1]
        tissue_coverage = tile.sum() / tile_pixels
        if tissue_coverage < coverage:
            idx_to_remove.append(idx)

    # We'll now remove items from the grid. starting from the end to the beginning in order to keep the indices correct:
    for idx in reversed(idx_to_remove):
        grid.pop(idx)

    return grid


def make_slides_xl_file(path: str = 'tcga-data'):
    """
    This function goes over all directories and makes a table with slides data:
    (1) id
    (2) file name
    (3) ER, PR, Her2 status
    (4) size of image
    (5) MPP (Microns Per Pixel)
    It also erases all 'log' subdirectories
    :return:
    """

    TCGA_BRCA_DF = pd.read_excel(os.path.join(path, 'TCGA_BRCA.xlsx'))
    TCGA_BRCA_DF.set_index('bcr_patient_barcode', inplace=True)

    print('Creating a new data file in path: {}'.format(path))

    id_list = []

    for idx, (root, dirs, files) in enumerate(tqdm(os.walk(path))):
        id_dict = {}
        if idx is 0:
            continue
        else:
            # get all *.svs files in the directory:
            files = glob.glob(os.path.join(root, '*.svs'))
            for _, file in enumerate(files):
                # Create a dictionary to the files and id's:
                id_dict['patient barcode'] = '-'.join(file.split('/')[-1].split('-')[0:3])
                id_dict['id'] = root.split('/')[-1]
                id_dict['file'] = file.split('/')[-1]

                # Get some basic data about the image like MPP (Microns Per Pixel) and size:
                img = openslide.open_slide(file)
                try:
                    id_dict['MPP'] = float(img.properties['aperio.MPP'])
                except:
                    id_dict['MPP'] = 'Missing Data'
                try:
                    id_dict['Width'] = int(img.dimensions[0])
                except:
                    id_dict['Width'] = 'Missing Data'
                try:
                    id_dict['Height'] = int(img.dimensions[1])
                except:
                    id_dict['Height'] = 'Missing Data'
                try:
                    id_dict['Objective Power'] = int(float(img.properties['aperio.AppMag']))
                except:
                    id_dict['Objective Power'] = 'Missing Data'
                try:
                    id_dict['Scan Date'] = img.properties['aperio.Date']
                except:
                    id_dict['Scan Date'] = 'Missing Data'
                img.close()

                # Get data from 'TCGA_BRCA.xlsx' and add to the dictionary ER_status, PR_status, Her2_status
                try:
                    id_dict['ER status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['ER_status']].values[0][0]
                    id_dict['PR status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['PR_status']].values[0][0]
                    id_dict['Her2 status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['Her2_status']].values[0][
                        0]
                    id_dict['test fold idx'] = \
                    TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['Test_fold_idx']].values[0][0]
                except:
                    id_dict['ER status'] = 'Missing Data'
                    id_dict['PR status'] = 'Missing Data'
                    id_dict['Her2 status'] = 'Missing Data'
                    id_dict['test fold idx'] = 'Missing Data'

                id_list.append(id_dict)

    slides_data = pd.DataFrame(id_list)
    slides_data.to_excel(os.path.join(path, 'slides_data.xlsx'))
    print('Created data file {}'.format(os.path.join(path, 'slides_data.xlsx')))


def make_segmentations(data_path: str = 'tcga-data/', rewrite: bool = False, magnification: int = 1,
                       data_format: str = 'TCGA'):
    # RanS, save locally
    make_dir('Segmentation_Images')
    make_dir('Thumbs')
    make_dir('Segmentation_Maps')

    '''if data_format == 'TCGA':
        dirs = _get_tcga_id_list(data_path)
        for _, dir in enumerate(dirs):
            if 'segImage.png' in next(os.walk(os.path.join(data_path, dir)))[2]:
                shutil.copy2(os.path.join(data_path, dir, 'segImage.png'),
                             os.path.join('Segmentation_Images', dir + '_SegImage.png'))
            else:
                print('Found no segImage file for {}'.format(dir))'''

    if data_format == 'TCGA':
        print('Making Segmentation Maps for each .svs file...')
        dirs = _get_tcga_id_list(data_path)
        error_list = []
        # for _, dir in enumerate(dirs):
        for i in tqdm(range(len(dirs))):  # In order to get rid of tqdm, just erase this line and un-comment the line above
            dir = dirs[i]
            fn = dir
            #if (not rewrite and 'segMap.png' in next(os.walk(os.path.join(data_path, dir)))[2]):
            if not rewrite:
                pic1 = os.path.exists(os.path.join(os.getcwd(), 'Thumbs', fn + '_thumb.png'))
                pic2 = os.path.exists(os.path.join(os.getcwd(), 'Segmentation_Maps', fn + '_SegMap.png'))
                pic3 = os.path.exists(os.path.join(os.getcwd(), 'Segmentation_Images', fn + '_SegImage.png'))
                if pic1 and pic2 and pic3:
                    continue

            print('Working on {}'.format(dir))
            slide = _get_slide(os.path.join(data_path, dir), data_format)
            if slide is not None:
                # Get a thunmbnail image to create the segmentation for:
                try:
                    objective_pwr = int(float(slide.properties['aperio.AppMag']))
                except KeyError:
                    print('Couldn\'t find Magnification - Segmentation Map was not Created')
                    continue
                height = slide.dimensions[1]
                width = slide.dimensions[0]
                try:
                    thumb = slide.get_thumbnail(
                        (width / (objective_pwr / magnification), height / (objective_pwr / magnification)))
                except openslide.lowlevel.OpenSlideError as err:
                    error_dict = {}
                    e = sys.exc_info()
                    error_dict['Path'] = dir
                    error_dict['Error'] = err
                    error_dict['Error Details 1'] = e[0]
                    error_dict['Error Details 2'] = e[1]
                    error_list.append(error_dict)
                    print('Exception on path {}'.format(dir))
                    continue

                thmb_seg_map, thmb_seg_image = _make_segmentation_for_image(thumb, magnification)
                slide.close()
                # Saving segmentation map, segmentation image and thumbnail:
                thumb.save(os.path.join('Thumbs', fn + '_thumb.png'))
                thmb_seg_map.save(os.path.join('Segmentation_Maps', fn + '_SegMap.png'))
                thmb_seg_image.save(os.path.join('Segmentation_Images', fn + '_SegImage.png'))

                '''thumb.save(os.path.join(data_path, dir, 'thumb.png'))
                thmb_seg_map.save(os.path.join(data_path, dir, 'segMap.png'))
                thmb_seg_image.save(os.path.join(data_path, dir, 'segImage.png'))'''

            else:
                print('Error: Found no slide in path {}'.format(dir))
                # TODO: implement a case for a slide that cannot be opened.
                continue

    elif data_format == 'ABCTB' or data_format == 'MIRAX':
        if data_format == 'ABCTB':
            print('Making Segmentation Maps for each .ndpi file...')
            files = [file for file in os.listdir(data_path) if file.endswith(".ndpi")]
        elif data_format == 'MIRAX':
            print('Making Segmentation Maps for each .mrxs file...')
            files = [file for file in os.listdir(data_path) if file.endswith(".mrxs")]
        error_list = []
        for file in files:
            fn = file[:-5]
            if not rewrite:
                pic1 = os.path.exists(os.path.join(os.getcwd(), 'Thumbs', fn + '_thumb.png'))
                pic2 = os.path.exists(os.path.join(os.getcwd(), 'Segmentation_Maps', fn + '_SegMap.png'))
                pic3 = os.path.exists(os.path.join(os.getcwd(), 'Segmentation_Images', fn + '_SegImage.png'))
                if pic1 and pic2 and pic3:
                    continue

            print('Working on {}'.format(file))
            slide = _get_slide(os.path.join(data_path, file), data_format)
            if slide is not None:
                # Get a thunmbnail image to create the segmentation for:
                try:
                    if data_format == 'ABCTB':
                        objective_pwr = int(float(slide.properties['hamamatsu.SourceLens']))
                    elif data_format == 'MIRAX':
                        objective_pwr = int(float(slide.properties['openslide.objective-power']))
                except KeyError:
                    print('Couldn\'t find Magnification - Segmentation Map was not Created')
                    continue
                height = slide.dimensions[1]
                width = slide.dimensions[0]
                try:
                    thumb = slide.get_thumbnail(
                        (width / (objective_pwr / magnification), height / (objective_pwr / magnification)))
                except openslide.lowlevel.OpenSlideError as err:
                    error_dict = {}
                    e = sys.exc_info()
                    error_dict['Path'] = file
                    error_dict['Error'] = err
                    error_dict['Error Details 1'] = e[0]
                    error_dict['Error Details 2'] = e[1]
                    error_list.append(error_dict)
                    print('Exception on file {}'.format(file))
                    continue

                thmb_seg_map, thmb_seg_image = _make_segmentation_for_image(thumb, magnification)
                slide.close()
                # Saving segmentation map, segmentation image and thumbnail:
                thumb.save(os.path.join('Thumbs', fn + '_thumb.png'))
                thmb_seg_map.save(os.path.join('Segmentation_Maps', fn + '_SegMap.png'))
                thmb_seg_image.save(os.path.join('Segmentation_Images', fn + '_SegImage.png'))
                '''thumb.save(os.path.join(data_path, fn + '_thumb.png'))
                thmb_seg_map.save(os.path.join(data_path, fn + '_segMap.png'))
                thmb_seg_image.save(os.path.join(data_path, fn + '_segImage.png'))'''

            else:
                print('Error: Found no slide in path {}'.format(data_path))
                # TODO: implement a case for a slide that cannot be opened.
                continue

    if len(error_list) != 0:
        # Saving all error data to excel file:
        error_DF = pd.DataFrame(error_list)
        error_DF.to_excel('Segmentation_Errors.xlsx')
        print('Segmentation Process finished WITH EXCEPTIONS!!!!')
        print('Check "Segmenatation_Errors.xlsx" file for details...')
    else:
        print('Segmentation Process finished without exceptions!')


def _get_slide(path: 'str', data_format: str = 'TCGA') -> openslide.OpenSlide:
    """
    This function returns an OpenSlide object from the file within the directory
    :param path:
    :return:
    """
    if data_format == 'TCGA':
        # file = next(os.walk(path))[2]  # TODO: this line can be erased since we dont use file. also check the except part...
        # if '.DS_Store' in file: file.remove('.DS_Store')
        slide = None
        try:
            # slide = openslide.open_slide(os.path.join(path, file[0]))
            slide = openslide.open_slide(glob.glob(os.path.join(path, '*.svs'))[0])
        except:
            print('Cannot open slide at location: {}'.format(path))
    elif data_format == 'ABCTB' or data_format == 'MIRAX':
        slide = None
        try:
            slide = openslide.open_slide(path)
        except:
            print('Cannot open slide at location: {}'.format(path))

    return slide


def _get_tcga_id_list(path: str = 'tcga-data'):
    """
    This function returns the id of all images in the TCGA data directory given by 'path'
    :return:
    """
    return next(os.walk(path))[1]


def _make_segmentation_for_image(image: Image, magnification: int) -> (Image, Image):
    """
    This function creates a segmentation map for an Image
    :param magnification:
    :return:
    """
    # Converting the image from RGBA to HSV and to a numpy array (from PIL):
    image_array = np.array(image.convert('HSV'))
    # otsu Thresholding:
    _, seg_map = cv.threshold(image_array[:, :, 1], 0, 255, cv.THRESH_OTSU)

    # Smoothing the tissue segmentation imaqe:
    size = 30 * magnification
    kernel_smooth = np.ones((size, size), dtype=np.float32) / size ** 2
    seg_map = cv.filter2D(seg_map, -1, kernel_smooth)

    th_val = 5
    seg_map[seg_map > th_val] = 255
    seg_map[seg_map <= th_val] = 0

    # find small contours and delete them from segmentation map
    size_thresh = 10000
    contours, _ = cv.findContours(seg_map, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # drawContours = cv.drawContours(image_array, contours, -1, (0, 0, 255), -1)
    # cv.imshow("Contours", drawContours)
    # cv.waitKey()
    small_contours = []
    for contour in contours:
        contour_area = cv.contourArea(contour)
        if contour_area < size_thresh:
            small_contours.append(contour)
    seg_map = cv.drawContours(seg_map, small_contours, -1, (0, 0, 255), -1)

    seg_map_PIL = Image.fromarray(seg_map)
    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))  # cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert('RGB')
    seg_image = Image.blend(image, edge_image, 0.5)

    return seg_map_PIL, seg_image


def device_gpu_cpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using cpu')

    return device


def get_cpu():
    platform = sys.platform
    if platform == 'linux':
        cpu = len(os.sched_getaffinity(0))
    elif platform == 'darwin':
        cpu = 2
        platform = 'MacOs'
    else:
        cpu = 1
        platform = 'Unrecognized'

    # if cpu > 6:
    #    cpu -= 2
    print('Running on {} with {} workers'.format(platform, cpu))
    return cpu


def run_data(experiment: str = None, test_fold: int = 1, transformations: bool = False, tile_size: int = 256,
             tiles_per_bag: int = 50):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :return:
    """

    run_file_name = 'runs/run_data.xlsx'
    run_DF = pd.read_excel(run_file_name)
    try:
        run_DF.drop(labels='Unnamed: 0', axis='columns', inplace=True)
    except KeyError:
        pass

    run_DF_exp = run_DF.set_index('Experiment', inplace=False)

    # If a new experiment is conducted:
    if experiment is None:
        experiment = run_DF_exp.index.values.max() + 1
        location = 'runs/Exp_' + str(experiment) + '-TestFold_' + str(test_fold)
        run_dict = {'Experiment': experiment,
                    'Test Fold': test_fold,
                    'Transformations': transformations,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'Location': location
                    }
        run_DF = run_DF.append([run_dict], ignore_index=True)
        run_DF.to_excel(run_file_name)
        print('Created a new Experiment (number {}). It will be saved at location: {}'.format(experiment, location))

        return location
    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        transformations = bool(run_DF_exp.loc[[experiment], ['Transformations']].values[0][0])
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])

        return location, test_fold, transformations, tile_size, tiles_per_bag


class WSI_MILdataset(Dataset):
    def __init__(self,
                 data_path: str = 'tcga-data',
                 tile_size: int = 256,
                 num_of_tiles_from_slide: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform=False):

        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        meta_data_file = os.path.join(data_path, 'slides_data.xlsx')
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)

        # self.meta_data_DF.set_index('id')
        self.data_path = data_path
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.num_of_tiles_from_slide = num_of_tiles_from_slide
        self.train = train
        self.print_time = print_timing
        self.transform = transform

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            self.image_file_names.append(all_image_file_names[index])
            self.image_path_names.append(all_image_path_names[index])
            self.in_fold.append(all_in_fold[index])
            self.tissue_tiles.append(all_tissue_tiles[index])
            self.target.append(all_targets[index])
            self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        if self.transform:
            # TODO: Consider using - torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: Consider transforms.RandomHorizontalFlip()

            self.transform = \
                transforms.Compose([  # transforms.RandomRotation([self.rotate_by, self.rotate_by]),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255),
                                         std=(
                                         40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255))
                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.22826, 0.37736, 0.275547),
                                                                      std=(0.158447, 0.231005, 0.1768365))
                                                 ])

        print(
            'Initiation of WSI {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}'
            .format('Train' if self.train else 'Test',
                    self.__len__(),
                    self.tile_size,
                    self.num_of_tiles_from_slide,
                    'Without' if transform is False else 'With',
                    self.test_fold))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        start = time.time()
        file_name = os.path.join(self.data_path, self.image_path_names[idx], self.image_file_names[idx])
        tiles = _choose_data(file_name, self.num_of_tiles_from_slide, self.magnification[idx], self.tile_size)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        """    
        # This is for images in an nd.array:        
        shape = tiles.shape
        X = torch.zeros(shape)
        """

        # The following section is written for tiles in PIL format
        X = torch.zeros([self.num_of_tiles_from_slide, 3, self.tile_size, self.tile_size])

        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([transforms.RandomRotation([rotate_by, rotate_by]),
                                            self.transform
                                            ])
        else:
            transform = self.transform

        """
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.22826, 0.37736, 0.275547),
                                                                      std=(0.158447, 0.231005, 0.1768365))
                                                 ])
        else:
            # TODO: Consider using - torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: Consider transforms.RandomHorizontalFlip()

            
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            self.transform = \
                transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255),
                                                          std=(40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255))
                                     ])
        """

        # tiles = tiles.transpose(0, 2, 3, 1)  # When working with PIL, this line is not needed
        # Check the need to resize the images (in case of different magnification):

        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([transforms.Resize(self.tile_size), transform])

        for i in range(self.num_of_tiles_from_slide):
            #  X[i] = self.transform(tiles[i])  # This line is for nd.array
            X[i] = transform(tiles[i])

        if self.print_time:
            end = time.time()
            print('WSI: Time to prepare item is {:.2f} s'.format(end - start))
        return X, label


"""
class PreSavedTiles_MILdataset(Dataset):
    def __init__(self,
                 data_path: str = 'tcga-data',
                 tile_size: int = 256,
                 num_of_tiles_from_slide: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 transform = None):

        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        meta_data_file = os.path.join(data_path, 'slides_data.xlsx')
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        # self.meta_data_DF.set_index('id')
        self.data_path = data_path
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.num_of_tiles_from_slide = num_of_tiles_from_slide
        self.train = train
        self.transform = transform
        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]
        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - 256 compatible @ X20'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            self.image_file_names.append(all_image_file_names[index])
            self.image_path_names.append(all_image_path_names[index])
            self.in_fold.append(all_in_fold[index])
            self.tissue_tiles.append(all_tissue_tiles[index])
            self.target.append(all_targets[index])
            self.magnification.append(all_magnifications[index])

        print('Initiation of {} DataSet is Complete. Working with PreSaved tiles'.format('Train' if self.train else 'Test'))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start = time.time()
        tiles_file_name = os.path.join(self.data_path, self.image_path_names[idx], 'tiles.data')

        with open(tiles_file_name, 'rb') as filehandle:
            tiles = pickle.load(filehandle)

        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        X = torch.zeros([self.num_of_tiles_from_slide, 3, self.tile_size, self.tile_size])
        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor()])

        sample_idx = sample(range(len(tiles)), self.num_of_tiles_from_slide)
        for i, idx in enumerate(sample_idx):
            X[i] = self.transform(tiles[idx])

        end = time.time()
        print('PreSaved: Time to prepare item is {:.2f} s'.format(end - start))
        return X, label
"""

"""
def get_transform():
    # TODO: Consider using - torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    # TODO: Consider transforms.RandomHorizontalFlip()

    transform = transforms.Compose([#transforms.RandomRotation([180, 180]),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255),
                                                         std=(40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255))
                                    ])
    return transform
"""

"""
def _save_tile_list_to_file(slide_name: str, tile_list: list, path: str = 'tcga-data'):
    tile_list.sort()
    tile_list_dict = {'Slide': slide_name.split('/')[1],
                      'Tiles': str(tile_list)}

    if os.path.isfile(os.path.join(path, 'train_tile_selection.xlsx')):
        tile_list_DF = pd.read_excel(os.path.join(path, 'train_tile_selection.xlsx'))
        tile_list_DF = tile_list_DF.append([tile_list_dict], ignore_index=False)
        try:
            tile_list_DF.drop(labels='Unnamed: 0', axis='columns',  inplace=True)
        except:
            pass
    else:
        tile_list_DF = pd.DataFrame([tile_list_dict])

    tile_list_DF.to_excel(os.path.join(path, 'train_tile_selection.xlsx'))
"""


class Infer_WSI_MILdataset(Dataset):
    def __init__(self,
                 data_path: str = 'tcga-data',
                 tile_size: int = 256,
                 tiles_per_bag: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = False,
                 print_timing: bool = True,
                 transform=False):

        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        meta_data_file = os.path.join(data_path, 'slides_data.xlsx')
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)

        # self.meta_data_DF.set_index('id')
        self.data_path = data_path
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.tiles_per_bag = tiles_per_bag
        self.train = train
        self.print_time = print_timing
        self.transform = transform

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.bags_per_slide = []
        self.grid_list = []

        for idx, index in enumerate(valid_slide_indices):
            self.image_file_names.append(all_image_file_names[index])
            self.image_path_names.append(all_image_path_names[index])
            self.in_fold.append(all_in_fold[index])
            self.tissue_tiles.append(all_tissue_tiles[index])
            self.target.append(all_targets[index])
            self.magnification.append(all_magnifications[index])
            self.bags_per_slide.append(-(-all_tissue_tiles[index] // self.tiles_per_bag))  # round up

            grid = _get_grid_list(file_name=os.path.join(self.data_path,
                                                         self.image_path_names[idx],
                                                         self.image_file_names[idx]),
                                  magnification=all_magnifications[index],
                                  tile_size=self.tile_size)
            bag = []
            for i in range(self.bags_per_slide[idx]):
                if bag == self.bags_per_slide[idx] - 1:
                    bag.append(grid[(i + 1) * 50:])
                else:
                    bag.append(grid[i * 50: (i + 1) * 50])

            self.grid_list.append(bag)

        # Setting the transformation:
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.22826, 0.37736, 0.275547),
                                                                  std=(0.158447, 0.231005, 0.1768365))
                                             ])

        self.last_bag = False

        print(
            'Initiation of Inference WSI {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform'
            .format('Train' if self.train else 'Test',
                    len(self.target),
                    self.tile_size,
                    self.tiles_per_bag,
                    'Without' if transform is False else 'With'))

    def __len__(self):
        return sum(self.bags_per_slide)

    def __getitem__(self, idx):
        start = time.time()
        # According to idx we need to find which slide and which bag we're in.

        grid_locations = self.grid_list

        file_name = os.path.join(self.data_path, self.image_path_names[idx], self.image_file_names[idx])
        tiles = _choose_data(file_name, self.tiles_per_bag, self.magnification[idx], self.tile_size)

        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # The following section is written for tiles in PIL format
        X = torch.zeros([self.tiles_per_bag, 3, self.tile_size, self.tile_size])

        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([transforms.Resize(self.tile_size), self.transform])

        for i in range(self.tiles_per_bag):
            X[i] = transform(tiles[i])

        if self.print_time:
            end = time.time()
            print('Infer WSI: Time to prepare item is {:.2f} s'.format(end - start))
        return X, label, self.last_bag
