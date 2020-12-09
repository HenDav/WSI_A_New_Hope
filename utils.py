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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple
import shutil
import cv2 as cv
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
             tile_size: int = 256, tiles_per_bag: int = 50, DX: bool = False, DataSet: str = 'TCGA',
             epoch: int = None, model: str = None, transformation_string: str = None):
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
                    'Location': location,
                    'DX': DX,
                    'DataSet': DataSet,
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

        return location, test_fold, transformations, tile_size, tiles_per_bag, DX, DataSet


def get_concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class WSI_MILdataset(Dataset):
    def __init__(self,
                 #data_path: str = '/Users/wasserman/Developer/All data - outer scope',
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False):

        self.ROOT_PATH = 'All Data'

        if target_kind not in ['ER', 'PR', 'Her2', 'PDL1']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if DataSet == 'HEROHE':
            target_kind = 'Her2'
            #test_fold = 2

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        ### self.data_path = os.path.join(self.ROOT_PATH, self.DataSet)
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ MyRotation(angles=[0, 90, 180, 270]),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     Cutout(n_holes=1, length=100),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()
        #data_path_extended = os.path.join(self.ROOT_PATH, self.image_path_names[idx])
        # file_name = os.path.join(self.data_path, self.image_path_names[idx], self.image_file_names[idx])
        # tiles = _choose_data(file_name, self.num_of_tiles_from_slide, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        #tiles, time_list = _choose_data_2(self.data_path, file_name, self.bag_size, self.magnification[idx], self.tile_size, print_timing=self.print_time)
        '''
        tiles, time_list = _choose_data_2(data_path_extended, self.image_file_names[idx], self.bag_size, self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        '''
        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])

        tiles, time_list = _choose_data_2(grid_file, image_file, self.bag_size,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # The following section is written for tiles in PIL format
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        '''
        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform
        '''
        transform = self.transform


        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])

        start_aug = time.time()
        ### trans = transforms.ToPILImage()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])
            '''
            img = get_concat(tiles[i], trans(X[i]))
            img.show()
            time.sleep(3)
            '''

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        trans = transforms.ToTensor()
        return X, label, time_list, self.image_file_names[idx], trans(tiles[0])


'''
class PreSavedTiles_MILdataset(Dataset):
    def __init__(self,
                 data_path: str = 'tcga-data',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform = False):

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
        self.bag_size = bag_size
        self.train = train
        self.print_timing = print_timing
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
        if self.transform and self.train:
            self.transform = \
                transforms.Compose([ transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255),
                                                          std=(40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255))
                                     ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.22826, 0.37736, 0.275547),
                                                                      std=(0.158447, 0.231005, 0.1768365))
                                                 ])

        print('Initiation of PreSaved Tiles {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}'
              .format('Train' if self.train else 'Test',
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold))


    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        start_load_tiles = time.time()
        tiles_basic_file_name = os.path.join(self.data_path, self.image_path_names[idx], 'tiles', str(self.tile_size))

        tile_idxs = sample(range(500), self.bag_size)
        tiles_PIL = []
        for _, tile_idx in enumerate(tile_idxs):
            tile_file_name = os.path.join(tiles_basic_file_name, str(tile_idx) + '.data')
            with open(tile_file_name, 'rb') as filehandle:
                tile = pickle.load(filehandle)
                tiles_PIL.append(tile[0])

        time_load_tiles = (time.time() - start_load_tiles) / self.bag_size
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        # Adding rotation as data aug. for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform

        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([transforms.Resize(self.tile_size), transform])

        time_start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = transform(tiles_PIL[i])

        aug_time = (time.time() - time_start_aug) / self.bag_size

        total_time = time.time() - start_load_tiles
        if self.print_timing:
            time_list = (time_load_tiles, aug_time, total_time)
        else:
            time_list = [0]

        return X, label, time_list

'''

'''
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
'''


'''
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
'''



class Infer_WSI_MILdataset(Dataset):
    def __init__(self,
                 DataSet: str = 'HEROHE',
                 tile_size: int = 256,
                 tiles_per_iter: int = 400,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 print_timing: bool = False,
                 DX: bool = False,
                 num_tiles: int = 500):

        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if DataSet == 'HEROHE':
            target_kind = 'Her2'

        self.ROOT_PATH = 'All Data'
        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        self.tiles_per_iter = tiles_per_iter
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.folds = folds
        self.print_time = print_timing
        self.DX = DX
        # The following attributes will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = 0
        self.current_file = None

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.num_patches = []

        self.image_full_filenames = []
        self.slide_grids = []


        for _, slide_num in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= all_tissue_tiles[slide_num]:
                    self.num_patches.append(num_tiles)
                else:
                    self.num_patches.append(all_tissue_tiles[slide_num])
                    print('{} Slide available patches are less than {}'.format(all_image_file_names[slide_num], num_tiles))

                self.image_file_names.append(all_image_file_names[slide_num])
                self.image_path_names.append(all_image_path_names[slide_num])
                self.in_fold.append(all_in_fold[slide_num])
                self.tissue_tiles.append(all_tissue_tiles[slide_num])
                self.target.append(all_targets[slide_num])
                self.magnification.extend([all_magnifications[slide_num]] * self.num_patches[-1])

                full_image_filename = os.path.join(self.ROOT_PATH, self.image_path_names[-1], self.image_file_names[-1])
                self.image_full_filenames.append(full_image_filename)
                basic_file_name = '.'.join(self.image_file_names[-1].split('.')[:-1])
                grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[-1], 'Grids',
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                which_patches = sample(range(int(self.tissue_tiles[-1])), self.num_patches[-1])

                with open(grid_file, 'rb') as filehandle:
                    grid_list = pickle.load(filehandle)
                chosen_locations = [ grid_list[loc] for loc in which_patches ]
                chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)
                ### self.slide_multiple_filenames.extend([full_image_filename] * self.num_patches[-1])

        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(
                                                 mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                 std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                             ])
        '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
        '''transforms.Normalize(mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                                  std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''

        print("Normalization Values are:")
        print(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2], STD['Ron'][0], STD['Ron'][1], STD['Ron'][2])
        print(
            'Initiation of WSI INFERENCE for {} DataSet of folds {} is Complete. {} Slides, Working on Tiles of size {}^2. {} Tiles per slide, {} tiles per iteration, {} iterations to complete full inference'
            .format(self.DataSet,
                    str(self.folds),
                    len(self.image_file_names),
                    self.tile_size,
                    num_tiles,
                    self.tiles_per_iter,
                    self.__len__()))

    def __len__(self):
        ### return len(self.slide_multiple_filenames)
        return int(np.ceil(np.array(self.num_patches)/self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.tiles_to_go = self.num_patches[self.slide_num]
            self.current_file = self.image_full_filenames[self.slide_num]
            self.initial_num_patches = self.num_patches[self.slide_num]

        label = [1] if self.target[self.slide_num] == 'Positive' else [0]
        label = torch.LongTensor(label)

        if self.tiles_to_go <= self.tiles_per_iter:
            ### tiles_next_iter = self.tiles_to_go
            self.tiles_to_go = None
            self.slide_num += 1
        else:
            ### tiles_next_iter = self.tiles_per_iter
            self.tiles_to_go -= self.tiles_per_iter

        adjusted_tile_size = self.tile_size * (self.magnification[idx] // self.BASIC_MAGNIFICATION)
        tiles, time_list = _get_tiles_2(self.current_file,
                                       self.slide_grids[idx],
                                       adjusted_tile_size,
                                       self.print_time)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([transforms.Resize(self.tile_size), self.transform])
        else:
            transform = self.transform

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = transform(tiles[i])

        aug_time = time.time() - start_aug
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]
        if self.tiles_to_go is None:
            last_batch = True
        else:
            last_batch = False

        print('Slide: {}, tiles: {}'.format(self.current_file, self.slide_grids[idx]))
        return X, label, time_list, last_batch, self.initial_num_patches

class WSI_REGdataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False,
                 n_patches_test: int = 1,
                 n_patches_train: int = 50,
                 transform_type: str = 'flip'):

        #define data root
        if sys.platform == 'linux': #GIPdeep
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'/home/womer/project/All Data'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'/home/rschley/All_Data/LUNG'
        elif sys.platform == 'win32': #Ran local
            if DataSet == 'HEROHE':
                self.ROOT_PATH = r'C:\ran_data\HEROHE_examples'
            elif DataSet == 'TCGA':
                self.ROOT_PATH = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat'
            elif DataSet == 'LUNG':
                self.ROOT_PATH = r'C:\ran_data\Lung_examples'
        else: #Omer local
            if (DataSet == 'HEROHE') or (DataSet == 'TCGA'):
                self.ROOT_PATH = r'All Data'
            elif DataSet == 'LUNG':
                pass #TODO omer, add your lung path if needed

        if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
            raise ValueError('target should be one of: PDL1, EGFR')
        elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = 1
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            #folds = list(range(1, 7))
            folds = list(self.meta_data_DF['test fold idx'].unique())  # RanS 17.11.20
            folds.remove(self.test_fold)
            if 'test' in folds:
                folds.remove('test')
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []

        for _, index in enumerate(valid_slide_indices):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

        # Setting the transformation:
        final_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))])
        if self.transform and self.train:
            # TODO: Consider using - torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            if transform_type=='flip':
                self.scale_factor = 0
                transform1 = \
                    transforms.Compose([ transforms.RandomVerticalFlip(),
                                         transforms.RandomHorizontalFlip()])
            elif transform_type == 'wcfrs': #weak color, flip, rotate, scale
                self.scale_factor = 0.2
                transform1 = \
                    transforms.Compose([
                        # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                        transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                               saturation=0.1, hue=(-0.1, 0.1)),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        MyRotation(angles=[0, 90, 180, 270]),
                        transforms.RandomAffine(degrees=0, scale=(1 - self.scale_factor, 1 + self.scale_factor)),
                        transforms.CenterCrop(self.tile_size),  #fix boundary when scaling<1
                    ])
            self.transform = transforms.Compose([transform1,
                                                 final_transform])
        else:
            self.scale_factor = 0
            self.transform = final_transform

        #self.factor = 10 if self.train else 1
        self.factor = n_patches_train if self.train else n_patches_test  # RanS 7.12.20
        self.real_length = int(self.__len__() / self.factor)


        print('Initiation of REG {} {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.real_length,
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))


    def __len__(self):
        return len(self.target) * self.factor


    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length
        data_path_extended = os.path.join(self.ROOT_PATH, self.image_path_names[idx])
        '''
        tiles, time_list = _choose_data_2(data_path_extended, self.image_file_names[idx], self.bag_size, self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        '''
        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])

        tiles, time_list = _choose_data_2(grid_file, image_file, self.bag_size,
                                          self.magnification[idx],
                                          #self.tile_size,
                                          int(self.tile_size / (1 - self.scale_factor)), # RanS 7.12.20, fix boundaries with scale
                                          print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # The following section is written for tiles in PIL format
        X = torch.zeros([1, 3, self.tile_size, self.tile_size])

        '''
        # Updating RandomRotation angle in the data transformations only for train set:
        if self.train:
            rotate_by = sample([0, 90, 180, 270], 1)[0]
            transform = transforms.Compose([ transforms.RandomRotation([rotate_by, rotate_by]),
                                             self.transform
                                             ])
        else:
            transform = self.transform
        '''
        transform = self.transform


        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])

        start_aug = time.time()
        trans = transforms.ToPILImage()

        X = transform(tiles[0])
        '''
        img = get_concat(tiles[0], trans(X))
        img.show()
        time.sleep(3)
        '''

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
            # print('Data Augmentation time is: {:.2f} s'.format(aug_time))
            # print('WSI: TOTAL Time to prepare item is: {:.2f} s'.format(total_time))
        else:
            time_list = [0]

        slide_name = self.image_file_names[idx] #RanS 8.12.20
        return X, label, time_list, slide_name

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


class WSI_MIL2_dataset(Dataset):
    """
    This DataSet class is used for MIL paradigm training.
    This class uses patches from different slides (corresponding to the same label) is a bag.
    """
    def __init__(self,
                 #data_path: str = '/Users/wasserman/Developer/All data - outer scope',
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 TPS: int = 10, # Tiles Per Slide
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform : bool = False,
                 DX : bool = False):

        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')
        if DataSet == 'HEROHE':
            target_kind = 'Her2'
            #test_fold = 2

        self.ROOT_PATH = 'All Data'
        meta_data_file = os.path.join(self.ROOT_PATH, 'slides_data.xlsx')
        self.DataSet = DataSet
        self.BASIC_MAGNIFICATION = 20
        self.meta_data_DF = pd.read_excel(meta_data_file)
        if self.DataSet is not 'ALL':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        ### self.data_path = os.path.join(self.ROOT_PATH, self.DataSet)
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.transform = transform
        self.DX = DX

        # In test mode we want all tiles to be from the same slide:
        if self.train:
            self.TPS = TPS
        else:
            self.TPS = self.bag_size
        self.slides_in_bag = int(bag_size / self.TPS)

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if self.train:
            folds = list(range(1, 7))
            folds.remove(self.test_fold)
        else:
            folds = [self.test_fold]
        self.folds = folds

        correct_folds = self.meta_data_DF['test fold idx'][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_path_names = list(self.meta_data_DF['id'])
        all_in_fold = list(self.meta_data_DF['test fold idx'])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet is not 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Objective Power'])

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.POS_slides, self.NEG_slides = [], []


        for i, index in enumerate(valid_slide_indices):
            if all_targets[index] == 'Positive':
                self.POS_slides.append(i)
            else:
                self.NEG_slides.append(i)

            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])


        # Setting the transformation:
        '''
        mean = {}
        std = {}

        mean['TCGA'] = [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255]
        std['TCGA'] = [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255]

        mean['HEROHE'] = [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255]
        std['HEROHE'] = [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255]

        mean['Ron'] = [0.8998, 0.8253, 0.9357]
        std['Ron'] = [0.1125, 0.1751, 0.0787]
        '''
        if self.transform and self.train:

            # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            # TODO: transforms.RandomRotation([self.rotate_by, self.rotate_by]),
            # TODO: Cutout(n_holes=1, length=100),
            # TODO: transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5), saturation=(0.1), hue=(-0.1, 0.1)),
            # TODO: transforms.RandomHorizontalFlip(),

            self.transform = \
                transforms.Compose([ transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                         std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                     ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                         mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                         std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                     std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                                 ])
            '''transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))'''
            '''transforms.Normalize(
                                                     mean=(MEAN[self.DataSet][0], MEAN[self.DataSet][1], MEAN[self.DataSet][2]),
                                                     std=(STD[self.DataSet][0], STD[self.DataSet][1], STD[self.DataSet][2]))'''





        print('Initiation of WSI(MIL) {} {} DataSet is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
              .format('Train' if self.train else 'Test',
                      self.DataSet,
                      self.__len__(),
                      self.tile_size,
                      self.bag_size,
                      'Without' if transform is False else 'With',
                      self.test_fold,
                      'ON' if self.DX else 'OFF'))
        print('Tiles in a bag are gathered from {} slides'.format(self.slides_in_bag))

    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        start_getitem = time.time()

        basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])
        tiles, time_list = _choose_data_3(grid_file, image_file, self.TPS,
                                          self.magnification[idx],
                                          self.tile_size, print_timing=self.print_time)
        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # Sample tiles from other slides with same label - ONLY IN TRAIN MODE:
        if self.train:
            if self.target[idx] == 'Positive':
                slide_list = self.POS_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)
            else:
                slide_list = self.NEG_slides
                slide_list.remove(idx)
                slides_idx_other = sample(slide_list, self.bag_size - 1)

        if self.train:
            for _, index in enumerate(slides_idx_other):
                basic_file_name_other = '.'.join(self.image_file_names[index].split('.')[:-1])
                grid_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], 'Grids',
                                           basic_file_name_other + '--tlsz' + str(self.tile_size) + '.data')
                image_file_other = os.path.join(self.ROOT_PATH, self.image_path_names[index], self.image_file_names[index])

                tiles_other, time_list_other = _choose_data_3(grid_file_other, image_file_other, self.TPS,
                                                          self.magnification[index],
                                                          self.tile_size, print_timing=self.print_time)
                tiles.extend(tiles_other)

        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        transform = self.transform
        '''
        magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        if magnification_relation != 1:
            transform = transforms.Compose([ transforms.Resize(self.tile_size), transform ])
        '''
        start_aug = time.time()
        ### trans = transforms.ToPILImage()
        for i in range(self.bag_size):
            X[i] = transform(tiles[i])

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        trans = transforms.ToTensor()
        return X, label, time_list, self.image_file_names[idx], trans(tiles[0])
