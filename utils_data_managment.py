"""
This file contains all functions needed for data Management, Pre Processing etc.
That includes:
- Organization in directories.
- Image segmentation.
- Grid production
"""

import utils
import pandas as pd
from tqdm import tqdm
import os
import pickle
import shutil
import numpy as np
from PIL import Image
from typing import List, Tuple
import openslide
import glob
import sys
import cv2 as cv


'''
def make_dir(dirname):
    if not dirname in next(os.walk(os.getcwd()))[1]:
        try:
            os.mkdir(dirname)
        except OSError:
            print('Creation of directory ', dirname, ' failed...')
            raise
'''
'''
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
    seg_map_PIL = Image.fromarray(seg_map)

    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))  #cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert('RGB')
    seg_image = Image.blend(image, edge_image, 0.5)

    return seg_map_PIL, seg_image
'''
'''
def TCGA_dirs_2_files():
    dirs = utils._get_tcga_id_list()
    print('Creating one directory for all TCGA slides...')
    if not os.path.isdir('All Data'):
        os.mkdir('All Data')
    if not os.path.isdir('All Data/TCGA'):
        os.mkdir('All Data/TCGA')

    for _, dir in enumerate(tqdm(dirs)):
        files = glob.glob(os.path.join('tcga-data', dir, '*.svs'))
        for _, path_file in enumerate(files):
            shutil.copy2(path_file, os.path.join('All Data/TCGA', path_file.split('/')[-1]))

    print('Finished moving all TCGA data to folder \'All Data\TCGA\'')
'''

def make_tiles_hard_copy(data_path: str = 'tcga-data', tile_size: int = 256, how_many_tiles: int = 500):
    """
    This function makes a hard copy of the tile in order to avoid using openslide
    :param data_path:
    :return:
    """

    dirs = utils._get_tcga_id_list(data_path)
    meta_data = pd.read_excel(os.path.join(data_path, 'slides_data.xlsx'))

    for i in tqdm(range(meta_data.shape[0])):
        if meta_data['Total tiles - 256 compatible @ X20'][i] == -1:
            print('Could not find tile data for slide XXXXXXX')
            continue

        slide_file_name = os.path.join(data_path, meta_data['id'][i], meta_data['file'][i])
        # slide_tiles = _choose_data(slide_file_name, how_many_tiles, meta_data['Objective Power'][i], tile_size, resize=True)
        tiles_basic_file_name = os.path.join(data_path, meta_data['id'][i], 'tiles')
        _make_HC_tiles_from_slide(slide_file_name, 0, how_many_tiles, tiles_basic_file_name, meta_data['Objective Power'][i], tile_size)


        """
        file_name = os.path.join(data_path, meta_data['id'][i], 'tiles', 'tiles.data')
        with open(file_name, 'wb') as filehandle:
            pickle.dump(slide_tiles, filehandle)
        """


def _make_HC_tiles_from_slide(file_name: str, from_tile: int, num_tiles: int, tile_basic_file_name: str, magnification: int = 20, tile_size: int = 256):
    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], basic_grid_file_name)
    with open(grid_file, 'rb') as filehandle:
        # read the data as binary data stream
        grid_list = pickle.load(filehandle)

    if not os.path.isdir(tile_basic_file_name):
        os.mkdir(tile_basic_file_name)
        os.mkdir(os.path.join(tile_basic_file_name, str(tile_size)))
    if not os.path.isdir(os.path.join(tile_basic_file_name, str(tile_size))):
        os.mkdir(os.path.join(tile_basic_file_name, str(tile_size)))

    for tile_idx in range(from_tile, from_tile + num_tiles):
        tile, _ = utils._get_tiles_2(file_name, [grid_list[tile_idx]], adjusted_tile_size)
        tile_file_name = os.path.join(tile_basic_file_name, str(tile_size), str(tile_idx) + '.data')
        with open(tile_file_name, 'wb') as filehandle:
            pickle.dump(tile, filehandle)


def compute_normalization_values(data_path: 'str'= 'tcga-data/') -> tuple:
    """
    This function runs over a set of images and compute mean and variance of each channel.
    The function computes these statistic values over the thumbnail images which are at X1 magnification
    :return:
    """

    # get a list of all directories with images:
    dirs = utils._get_tcga_id_list(data_path)
    stats_list =[]
    print('Computing image-set Mean and Variance...')
    meta_data = pd.read_excel(os.path.join(data_path, 'slides_data.xlsx'))
    meta_data.set_index('id', inplace=True)

    # gather tissue image values from thumbnail image using the segmentation map:
    #for idx, dir in enumerate(dirs):
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
    print('Finished computing statistical data over {} thumbnail slides'.format(i+1))
    print('Mean: {}'.format(total_mean))
    print('Variance: {}'.format(total_var))
    return total_mean, total_var


def make_grid(main_data_path: str = 'All Data', tile_sz: int = 256):
    """
    This function creates a location for all top left corners of the grid
    :param data_file: name of main excel data file containing size of images (this file is created by function :"make_slides_xl_file")
    :param tile_sz: size of tiles to be created
    :return:
    """
    data_file = os.path.join(main_data_path, 'slides_data.xlsx')

    #for data_file in data_files:
    BASIC_OBJ_PWR = 20

    basic_DF = pd.read_excel(data_file)
    files = list(basic_DF['file'])
    objective_power = list(basic_DF['Objective Power'])
    basic_DF.set_index('file', inplace=True)
    tile_nums = []
    file_inds = []
    total_tiles =[]
    print('Starting Grid production...')
    print()
    #for _, file in enumerate(files):
    for i in tqdm(range(len(files))):
        file = files[i]
        database = basic_DF.loc[file, 'id']
        if os.path.isfile(os.path.join(main_data_path, database, file)): #make sure file exists
            data_dict = {}
            height = basic_DF.loc[file, 'Height']
            width  = basic_DF.loc[file, 'Width']

            if objective_power[i] == 'Missing Data':
                print('Grid was not computed for file {}'.format(file))
                tile_nums.append(0)
                total_tiles.append(-1)
                continue

            converted_tile_size = int(tile_sz * (int(objective_power[i]) / BASIC_OBJ_PWR))
            basic_grid = [(row, col) for row in range(0, height, converted_tile_size) for col in range(0, width, converted_tile_size)]
            total_tiles.append((len(basic_grid)))

            # We now have to check, which tiles of this grid are legitimate, meaning they contain enough tissue material.
            #legit_grid = _legit_grid(os.path.join(data_file.split('/')[0], id, 'SegData', file[:-4] + '-segMap.png'),
            legit_grid = _legit_grid(os.path.join(main_data_path, database, 'SegData', 'SegMaps', file[:-4] + '_SegMap.png'),
                                     basic_grid,
                                     converted_tile_size,
                                     (height, width))

            # create a list with number of tiles in each file
            tile_nums.append(len(legit_grid))
            file_inds.append(i)

            # Save the grid to file:
            #if not os.path.isdir(os.path.join(data_path.split(splitter)[0], id, 'Grids')):
            #    os.mkdir(os.path.join(data_path.split(splitter)[0], id, 'Grids'))
            if not os.path.isdir(os.path.join(main_data_path, database, 'Grids')):
                os.mkdir(os.path.join(main_data_path, database, 'Grids'))

            #file_name = os.path.join(data_file.split(splitter)[0], id, 'Grids', file[:-4] + '--tlsz' + str(tile_sz) + '.data')
            file_name = os.path.join(main_data_path, database, 'Grids', file[:-4] + '--tlsz' + str(tile_sz) + '.data')
            with open(file_name, 'wb') as filehandle:
                # store the data as binary data stream
                pickle.dump(legit_grid, filehandle)


    # Adding the number of tiles to the excel file:
    #TODO - support adding grids to a half-filled excel files? (currently erases everything) RanS 26.10.20
    tile_nums_all = np.zeros(basic_DF.shape[0])
    total_tiles_all = np.zeros(basic_DF.shape[0])
    slide_usage_all = np.zeros(basic_DF.shape[0])

    tile_nums_all[file_inds] = tile_nums
    total_tiles_all[file_inds] = total_tiles
    slide_usage_all[file_inds] = list(((np.array(tile_nums) / np.array(total_tiles)) * 100).astype(int))

    basic_DF['Legitimate tiles - ' + str(tile_sz) + ' compatible @ X20'] = tile_nums_all
    basic_DF['Total tiles - ' + str(tile_sz) + ' compatible @ X20'] = total_tiles_all
    basic_DF['Slide tile usage [%] (for ' + str(tile_sz) + '^2 Pix/Tile)'] = slide_usage_all
    #basic_DF['Legitimate tiles - ' + str(tile_sz) + ' compatible @ X20'] = tile_nums
    #basic_DF['Total tiles - ' + str(tile_sz) + ' compatible @ X20'] = total_tiles
    #basic_DF['Slide tile usage [%] (for ' + str(tile_sz) + '^2 Pix/Tile)'] = list(((np.array(tile_nums) / np.array(total_tiles)) * 100).astype(int))
    basic_DF.to_excel(data_file)

    print('Finished Grid production phase !')


def _legit_grid(image_file_name: str, grid: List[Tuple], tile_size: int, size: tuple, coverage: int = 0.5) -> List[Tuple]:
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
    small_tile = (int(-(-tile_size//rows)), int(-(-tile_size//cols)))
    # computing the compatible grid for the small segmenatation map:
    idx_to_remove =[]
    for idx, (row, col) in enumerate(grid):
        new_row = int(-(-(row // rows)))
        new_col = int(-(-(col // cols)))

        # collect the data from the segMap:
        tile = segMap[new_row : new_row + small_tile[0], new_col : new_col + small_tile[1]]
        tile_pixels = small_tile[0] * small_tile[1]
        tissue_coverage = tile.sum() / tile_pixels
        if tissue_coverage < coverage:
            idx_to_remove.append(idx)

    # We'll now remove items from the grid. starting from the end to the beginning in order to keep the indices correct:
    for idx in reversed(idx_to_remove):
        grid.pop(idx)

    return grid


def make_slides_xl_file(root_dir: str = 'All Data', DataSet: str = 'TCGA'):
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
    SLIDES_DATA_FILE = 'slides_data.xlsx'
    META_DATA_FILE = {}
    META_DATA_FILE['TCGA'] = 'TCGA_BRCA.xlsx'
    META_DATA_FILE['HEROHE'] = 'HEROHE_HER2_STATUS.xlsx'


    data_file = os.path.join(root_dir, SLIDES_DATA_FILE)
    new_file = False if os.path.isfile(data_file) else True

    meta_data_DF = pd.read_excel(os.path.join(root_dir, DataSet, META_DATA_FILE[DataSet]))
    # meta_data_DF = pd.read_excel(os.path.join(root_dir, 'TCGA_BRCA.xlsx'))

    meta_data_DF.set_index('bcr_patient_barcode', inplace=True)

    if new_file:
        print('Creating a new data file in in \'{}\''.format(data_file))
    else:
        print('Adding data to file \'{}\''.format(data_file))
        slides_data_DF = pd.read_excel(data_file)
        try:
            slides_data_DF.drop(labels='Unnamed: 0', axis='columns',  inplace=True)
        except KeyError:
            pass

    id_list = []

    #RanS 27.10.20, support slide types
    slide_files_svs = glob.glob(os.path.join(root_dir, DataSet, '*.svs'))
    slide_files_ndpi = glob.glob(os.path.join(root_dir, DataSet, '*.ndpi'))
    slide_files_mrxs = glob.glob(os.path.join(root_dir, DataSet, '*.mrxs'))
    slides = slide_files_svs + slide_files_ndpi + slide_files_mrxs
    mag_dict = {'.svs': 'aperio.AppMag', '.ndpi': 'hamamatsu.SourceLens', '.mrxs': 'openslide.objective-power'}
    mpp_dict = {'.svs': 'aperio.MPP', '.ndpi': 'openslide.mpp-x', '.mrxs': 'openslide.mpp-x'}
    date_dict = {'.svs': 'aperio.Date', '.ndpi': 'tiff.DateTime', '.mrxs': 'mirax.GENERAL.SLIDE_CREATIONDATETIME'}

    #slides = glob.glob(os.path.join(path, '*.svs'))
    for idx, file in enumerate(tqdm(slides)):
        fn, data_format = os.path.splitext(os.path.basename(file))
        id_dict = {}

        # Create a dictionary to the files and id's:
        if DataSet is 'TCGA':
            id_dict['patient barcode'] = '-'.join(file.split('/')[-1].split('-')[0:3])
        elif DataSet is 'HEROHE':
            id_dict['patient barcode'] = int(file.split('/')[-1].split('.')[0])

        # id_dict['id'] = root.split('/')[-1]
        id_dict['id'] = DataSet
        id_dict['file'] = file.split('/')[-1]
        id_dict['DX'] = True if (file.find('DX') != -1 or DataSet is not 'TCGA') else False

        # Get some basic data about the image like MPP (Microns Per Pixel) and size:
        img = openslide.open_slide(file)
        try:
            #id_dict['MPP'] = float(img.properties['aperio.MPP'])
            id_dict['MPP'] = float(img.properties[mpp_dict[data_format]])
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
            #id_dict['Objective Power'] = int(float(img.properties['aperio.AppMag']))
            id_dict['Objective Power'] = int(float(img.properties[mag_dict[data_format]]))
        except:
            id_dict['Objective Power'] = 'Missing Data'
        try:
            #id_dict['Scan Date'] = img.properties['aperio.Date']
            id_dict['Scan Date'] = img.properties[date_dict[data_format]]
        except:
            id_dict['Scan Date'] = 'Missing Data'
        img.close()

        # Get data from META_DATA_FILE and add to the dictionary ER_status, PR_status, Her2_status
        try:
            id_dict['ER status'] = meta_data_DF.loc[[id_dict['patient barcode']], ['ER_status']].values[0][0]
        except:
            id_dict['ER status'] = 'Missing Data'
        try:
            id_dict['PR status'] = meta_data_DF.loc[[id_dict['patient barcode']], ['PR_status']].values[0][0]
        except:
            id_dict['PR status'] = 'Missing Data'
        try:
            id_dict['Her2 status'] = meta_data_DF.loc[[id_dict['patient barcode']], ['Her2_status']].values[0][0]
        except:
            id_dict['Her2 status'] = 'Missing Data'
        try:
            id_dict['test fold idx'] = meta_data_DF.loc[[id_dict['patient barcode']], ['Test_fold_idx']].values[0][0]
        except:
            id_dict['test fold idx'] = 1

        id_list.append(id_dict)

    if new_file:
        slides_data_DF = pd.DataFrame(id_list)
        messege_prefix = 'Creating'
    else:
        slides_data_DF = slides_data_DF.append(id_list)
        messege_prefix = 'Updated'

    slides_data_DF.to_excel(data_file)
    print('{} data file \'{}\''.format(messege_prefix, data_file))


def make_segmentations(data_path: str = 'All Data/TCGA/', rewrite: bool = False, magnification: int = 1):
    print('Making Segmentation Maps for each slide file...')
    if not os.path.isdir(os.path.join(data_path, 'SegData')):
        os.mkdir(os.path.join(data_path, 'SegData'))
    if not os.path.isdir(os.path.join(data_path, 'SegData','Thumbs')):
        os.mkdir(os.path.join(data_path, 'SegData','Thumbs'))
    if not os.path.isdir(os.path.join(data_path, 'SegData', 'SegMaps')):
        os.mkdir(os.path.join(data_path, 'SegData', 'SegMaps'))
    if not os.path.isdir(os.path.join(data_path, 'SegData', 'SegImages')):
        os.mkdir(os.path.join(data_path, 'SegData', 'SegImages'))

    slide_files_svs = glob.glob(os.path.join(data_path, '*.svs'))
    slide_files_ndpi = glob.glob(os.path.join(data_path, '*.ndpi'))
    slide_files_mrxs = glob.glob(os.path.join(data_path, '*.mrxs'))
    slide_files = slide_files_svs + slide_files_ndpi + slide_files_mrxs
    mag_dict = {'.svs': 'aperio.AppMag', '.ndpi': 'hamamatsu.SourceLens', '.mrxs': 'openslide.objective-power'}

    error_list = []
    for idx, file in enumerate(tqdm(slide_files)):

        fn, data_format = os.path.splitext(os.path.basename(file))

        if not rewrite:
            pic1 = os.path.exists(os.path.join(data_path, 'SegData', 'Thumbs', fn + '_thumb.png'))
            pic2 = os.path.exists(os.path.join(data_path, 'SegData', 'SegMaps', fn + '_SegMap.png'))
            pic3 = os.path.exists(os.path.join(data_path, 'SegData', 'SegImages', fn + '_SegImage.png'))
            if pic1 and pic2 and pic3:
                continue

        slide = None
        try:
            slide = openslide.open_slide(file)
        except:
            print('Cannot open slide at location: {}'.format(file))

        if slide is not None:
            # Get a thunmbnail image to create the segmentation for:
            try:
                #objective_pwr = int(float(slide.properties['aperio.AppMag']))
                objective_pwr = int(float(slide.properties[mag_dict[data_format]]))
            except KeyError:
                print('Couldn\'t find Magnification - Segmentation Map was not Created')
                continue

            height = slide.dimensions[1]
            width = slide.dimensions[0]
            try:
                thumb = slide.get_thumbnail((width / (objective_pwr / magnification), height / (objective_pwr / magnification)))
            except openslide.lowlevel.OpenSlideError as err:
                error_dict = {}
                e = sys.exc_info()
                error_dict['File'] = file
                error_dict['Error'] = err
                error_dict['Error Details 1'] = e[0]
                error_dict['Error Details 2'] = e[1]
                error_list.append(error_dict)
                print('Exception for file {}'.format(file))
                continue

            thmb_seg_map, thmb_seg_image = _make_segmentation_for_image(thumb, magnification)
            slide.close()
            # Saving segmentation map, segmentation image and thumbnail:
            thumb.save(os.path.join(data_path, 'SegData',  'Thumbs', fn + '_thumb.png'))
            thmb_seg_map.save(os.path.join(data_path, 'SegData', 'SegMaps', fn + '_SegMap.png'))
            thmb_seg_image.save(os.path.join(data_path, 'SegData', 'SegImages', fn + '_SegImage.png'))

        else:
            print('Error: Found no slide in path {}'.format(dir))
            # TODO: implement a case for a slide that cannot be opened.
            continue


    if len(error_list) != 0:
        # Saving all error data to excel file:
        error_DF = pd.DataFrame(error_list)
        error_DF.to_excel(os.path.join('All Data', 'Segmentation_Errors.xlsx'))
        print('Segmentation Process finished WITH EXCEPTIONS!!!!')
        print('Check "Segmenatation_Errors.xlsx" file for details...')
    else:
        print('Segmentation Process finished without exceptions!')


def otsu3(img):
    #blur = cv.GaussianBlur(img,(5,5),0)
    # find normalized_histogram, and its cumulative distribution function
    #hist = cv.calcHist([blur],[0],None,[256],[0,256])
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1,-1
    for i in range(1,256):
        for j in range(i+1, 256):
            p1, p2, p3 = np.hsplit(hist_norm,[i,j]) # probabilities
            q1, q2, q3 = Q[i], Q[j]-Q[i], Q[255]-Q[j] # cum sum of classes
            if q1 < 1.e-6 or q2 < 1.e-6 or q3 < 1.e-6:
                continue
            b1, b2, b3 = np.hsplit(bins,[i,j]) # weights
            # finding means and variances
            m1, m2, m3 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2, np.sum(p3*b3)/q3
            v1, v2, v3 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2, np.sum(((b3-m3)**2)*p3)/q3
            # calculates the minimization function
            fn = v1*q1 + v2*q2 + v3*q3
            if fn < fn_min:
                fn_min = fn
                thresh = i,j
    return thresh


def _make_segmentation_for_image(image: Image, magnification: int) -> (Image, Image):
    """
    This function creates a segmentation map for an Image
    :param magnification:
    :return:
    """
    # Converting the image from RGBA to HSV and to a numpy array (from PIL):
    image_array = np.array(image.convert('HSV'))
    # otsu Thresholding:
    use_otsu3 = True
    if use_otsu3:
        # RanS 25.10.20 - 3way binarization
        thresh = otsu3(image_array[:, :, 1])
        _, seg_map = cv.threshold(image_array[:, :, 1], thresh[0], 255, cv.THRESH_BINARY)
    else:
        _, seg_map = cv.threshold(image_array[:, :, 1], 0, 255, cv.THRESH_OTSU)

    # Smoothing the tissue segmentation imaqe:
    size = 30 * magnification
    kernel_smooth = np.ones((size, size), dtype=np.float32) / size ** 2
    seg_map = cv.filter2D(seg_map, -1, kernel_smooth)

    th_val = 5
    seg_map[seg_map > th_val] = 255
    seg_map[seg_map <= th_val] = 0

    # find small contours and delete them from segmentation map
    size_thresh = 30000 #10000
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
    kernel_dilation = np.ones((3, 3))  #cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert('RGB')
    seg_image = Image.blend(image, edge_image, 0.5)

    return seg_map_PIL, seg_image


'''def _make_segmentation_for_image(image: Image, magnification: int) -> (Image, Image):
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
    seg_map_PIL = Image.fromarray(seg_map)

    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))  #cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=magnification * 2)).convert('RGB')
    seg_image = Image.blend(image, edge_image, 0.5)

    return seg_map_PIL, seg_image'''


def TCGA_dirs_2_files():
    dirs = utils._get_tcga_id_list()
    print('Creating one directory for all TCGA slides...')
    if not os.path.isdir('All Data'):
        os.mkdir('All Data')
    if not os.path.isdir('All Data/TCGA'):
        os.mkdir('All Data/TCGA')

    for _, dir in enumerate(tqdm(dirs)):
        files = glob.glob(os.path.join('tcga-data', dir, '*.svs'))
        for _, path_file in enumerate(files):
            shutil.copy2(path_file, os.path.join('All Data/TCGA', path_file.split('/')[-1]))

    print('Finished moving all TCGA data to folder \'All Data\TCGA\'')
