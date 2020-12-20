import numpy as np
from PIL import Image
import os
import openslide
import pandas as pd
import glob
import pickle
from random import sample
import random
import torch
from torchvision import transforms
import sys
import time
from typing import List, Tuple
from xlrd.biffh import XLRDError
from zipfile import BadZipFile


MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
        'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
        'Ron': [0.8998, 0.8253, 0.9357]
        }

STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
       'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
       'Ron': [0.1125, 0.1751, 0.0787]
       }


def chunks(list: List, length: int):
    new_list = [ list[i * length:(i + 1) * length] for i in range((len(list) + length - 1) // length )]
    return new_list


def make_dir(dirname):
    if not dirname in next(os.walk(os.getcwd()))[1]:
        try:
            os.mkdir(dirname)
        except OSError:
            print('Creation of directory ', dirname, ' failed...')
            raise


def _choose_data(file_name: str, how_many: int, magnification: int = 20, tile_size: int = 256, resize: bool = False, print_timing: bool = False):
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


    image_tiles = _get_tiles(file_name, locs, adjusted_tile_size, resize_to=resize_to, print_timing=print_timing)

    return image_tiles


def _choose_data_2(grid_file: str, image_file: str, how_many: int, magnification: int = 20, tile_size: int = 256, print_timing: bool = False):
    """
    This function choose and returns data to be held by DataSet
    :param file_name:
    :param how_many: how_many describes how many tiles to pick from the whole image
    :return:
    """
    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    ### basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    ### grid_file = os.path.join(data_path, 'Grids', file_name.split('.')[0] + '--tlsz' + str(tile_size) + '.data')

    #grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], 'Grids', file_name.split('/')[2][:-4] + '--tlsz' + str(tile_size) + '.data')
    with open(grid_file, 'rb') as filehandle:
        grid_list = pickle.load(filehandle)

    # Choose locations from the grid:
    loc_num = len(grid_list)

    idxs = sample(range(loc_num), how_many)
    locs = [grid_list[idx] for idx in idxs]

    ### image_file = os.path.join(data_path, file_name)
    image_tiles, time_list = _get_tiles_2(image_file, locs, adjusted_tile_size, print_timing=print_timing)

    return image_tiles, time_list


def _get_tiles_2(file_name: str, locations: List[Tuple], tile_sz: int, print_timing: bool = False):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """

    # open the .svs file:
    start_openslide = time.time()
    img = openslide.open_slide(file_name)
    end_openslide = time.time()

    tiles_num = len(locations)
    # TODO: Checking Pil vs. np array. Delete one of them...
    #tiles = np.zeros((tiles_num, 3, tile_sz, tile_sz), dtype=int)  # this line is needed when working with nd arrays
    tiles_PIL = []

    start_gettiles = time.time()
    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        image = img.read_region((loc[1], loc[0]), 0, (tile_sz, tile_sz)).convert('RGB')
        #tiles[idx, :, :, :] = np.array(image).transpose(2, 0, 1)  # this line is needed when working with nd arrays
        tiles_PIL.append(image)

    end_gettiles = time.time()


    if print_timing:
        time_list = [end_openslide - start_openslide, (end_gettiles - start_gettiles) / tiles_num]
        # print('WSI: Time to Openslide is: {:.2f} s, Time to Prepare {} tiles: {:.2f} s'.format(end_openslide - start_openslide,
        #                                                                   tiles_num,
        #                                                                   end_tiles - start_tiles))
    else:
        time_list = [0]

    return tiles_PIL, time_list


def _choose_data_3(grid_file: str, image_file: str, how_many: int, magnification: int = 20, tile_size: int = 256, print_timing: bool = False):
    """
    This function choose and returns data to be held by DataSet
    :param file_name:
    :param how_many: how_many describes how many tiles to pick from the whole image
    :return:
    """
    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    ### basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    ### grid_file = os.path.join(data_path, 'Grids', file_name.split('.')[0] + '--tlsz' + str(tile_size) + '.data')

    #grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], 'Grids', file_name.split('/')[2][:-4] + '--tlsz' + str(tile_size) + '.data')
    with open(grid_file, 'rb') as filehandle:
        grid_list = pickle.load(filehandle)

    # Choose locations from the grid:
    loc_num = len(grid_list)

    idxs = sample(range(loc_num), how_many)
    locs = [grid_list[idx] for idx in idxs]

    ### image_file = os.path.join(data_path, file_name)
    image_tiles, time_list = _get_tiles_3(image_file, locs, adjusted_tile_size, print_timing=print_timing)

    return image_tiles, time_list


def _get_tiles_3(file_name: str, locations: List[Tuple], tile_sz: int, print_timing: bool = False):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """

    # open the .svs file:
    start_openslide = time.time()
    img = openslide.open_slide(file_name)
    end_openslide = time.time()

    tiles_num = len(locations)
    # TODO: Checking Pil vs. np array. Delete one of them...
    #tiles = np.zeros((tiles_num, 3, tile_sz, tile_sz), dtype=int)  # this line is needed when working with nd arrays
    tiles_PIL = []

    start_gettiles = time.time()
    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        image = img.read_region((loc[1], loc[0]), 0, (tile_sz, tile_sz)).convert('RGB')
        #tiles[idx, :, :, :] = np.array(image).transpose(2, 0, 1)  # this line is needed when working with nd arrays
        tiles_PIL.append(image.resize((256, 256)))

    end_gettiles = time.time()


    if print_timing:
        time_list = [end_openslide - start_openslide, (end_gettiles - start_gettiles) / tiles_num]
        # print('WSI: Time to Openslide is: {:.2f} s, Time to Prepare {} tiles: {:.2f} s'.format(end_openslide - start_openslide,
        #                                                                   tiles_num,
        #                                                                   end_tiles - start_tiles))
    else:
        time_list = [0]

    return tiles_PIL, time_list


def _get_tile(file_name: str, locations: Tuple, tile_sz: int, print_timing: bool = False):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """

    # open the .svs file:
    start_openslide = time.time()
    img = openslide.open_slide(file_name)
    end_openslide = time.time()

    start_gettiles = time.time()
    image = img.read_region((locations[1], locations[0]), 0, (tile_sz, tile_sz)).convert('RGB')
    end_gettiles = time.time()


    if print_timing:
        time_list = [end_openslide - start_openslide, end_gettiles - start_gettiles]
    else:
        time_list = [0]

    return image, time_list


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


def _get_tiles(file_name: str, locations: List[Tuple], tile_sz: int, resize_to: int, print_timing: bool = False):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """
    # open the .svs file:
    start_openslide = time.time()
    img = openslide.open_slide(file_name)
    end_openslide = time.time()

    tiles_num = len(locations)
    # TODO: Checking Pil vs. np array. Delete one of them...
    #tiles = np.zeros((tiles_num, 3, tile_sz, tile_sz), dtype=int)  # this line is needed when working with nd arrays
    tiles_PIL = []

    start_tiles = time.time()
    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        image = img.read_region((loc[1], loc[0]), 0, (tile_sz, tile_sz)).convert('RGB')
        #tiles[idx, :, :, :] = np.array(image).transpose(2, 0, 1)  # this line is needed when working with nd arrays
        # TODO : try to remove this if !!!!!!!!
        if tile_sz != resize_to:
            image = image.resize((resize_to, resize_to), resample=Image.BILINEAR)

        tiles_PIL.append(image)
        end_tiles = time.time()
    if print_timing:
        print('WSI: Time to Openslide is: {:.2f} s, Time to Prepare {} tiles: {:.2f} s'.format(end_openslide - start_openslide,
                                                                           tiles_num,
                                                                           end_tiles - start_tiles))
    return tiles_PIL


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


    #cpu = 20
    print('Running on {} with {} workers'.format(platform, cpu))
    return cpu


def run_data(experiment: str = None, test_fold: int = 1, transform_type: str = 'none',#transformations: bool = False,
             tile_size: int = 256, tiles_per_bag: int = 50, num_bags: int = 1, DX: bool = False, DataSet: str = 'TCGA',
             epoch: int = None, model: str = None, transformation_string: str = None, Receptor: str = None,
             MultiSlide: bool = False):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :return:
    """

    run_file_name = 'runs/run_data.xlsx'
    if os.path.isfile(run_file_name):
        try:
            run_DF = pd.read_excel(run_file_name)
        except (XLRDError, BadZipFile):
            print('Couldn\'t open file {}'.format(run_file_name))
            return
        try:
            run_DF.drop(labels='Unnamed: 0', axis='columns',  inplace=True)
        except KeyError:
            pass

        run_DF_exp = run_DF.set_index('Experiment', inplace=False)
    else:
        run_DF = pd.DataFrame()

    # If a new experiment is conducted:
    if experiment is None:
        if os.path.isfile(run_file_name):
            experiment = run_DF_exp.index.values.max() + 1
        else:
            experiment = 1

        location = 'runs/Exp_' + str(experiment) + '-TestFold_' + str(test_fold)
        run_dict = {'Experiment': experiment,
                    'Test Fold': test_fold,
                    #'Transformations': transformations,
                    'Transformations': transform_type,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'MultiSlide Per Bag': MultiSlide,
                    'No. of Bags': num_bags,
                    'Location': location,
                    'DX': DX,
                    'DataSet': DataSet,
                    'Receptor': Receptor,
                    'Model': 'None',
                    'Last Epoch': 0,
                    'Transformation String': 'None'
                    }
        run_DF = run_DF.append([run_dict], ignore_index=True)
        if not os.path.isdir('runs'):
            os.mkdir('runs')

        run_DF.to_excel(run_file_name)
        print('Created a new Experiment (number {}). It will be saved at location: {}'.format(experiment, location))

        return location, experiment

    elif experiment is not None and epoch is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Last Epoch'] = epoch
        run_DF.to_excel(run_file_name)
    elif experiment is not None and model is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Model'] = model
        run_DF.to_excel(run_file_name)
    elif experiment is not None and transformation_string is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Transformation String'] = transformation_string
        run_DF.to_excel(run_file_name)


    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        #transformations = bool(run_DF_exp.loc[[experiment], ['Transformations']].values[0][0])
        transformations = run_DF_exp.loc[[experiment], ['Transformations']].values[0][0] #RanS 9.12.20
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])
        DX = bool(run_DF_exp.loc[[experiment], ['DX']].values[0][0])
        DataSet = str(run_DF_exp.loc[[experiment], ['DataSet']].values[0][0])
        Receptor = str(run_DF_exp.loc[[experiment], ['Receptor']].values[0][0])
        MultiSlide = str(run_DF_exp.loc[[experiment], ['MultiSlide Per Bag']].values[0][0])

        return location, test_fold, transformations, tile_size, tiles_per_bag, DX, DataSet, Receptor, MultiSlide


def get_concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = torch.randint(low=0, high=h, size=(1,)).numpy()[0]
            x = torch.randint(low=0, high=w, size=(1,)).numpy()[0]
            '''
            # Numpy random numbers will produce the same numbers in every epoch - I changed the random number producer
            # to torch.random to overcome this issue. 
            y = np.random.randint(h)            
            x = np.random.randint(w)
            '''


            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MyRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)