import numpy as np
import os
import pandas as pd
import pickle
from random import sample
import torch
from torchvision import transforms
import time
from torch.utils.data import Dataset
from typing import List
from utils import MyRotation, Cutout, _get_tiles, _choose_data, chunks
from utils import HEDColorJitter, define_transformations, define_data_root, assert_dataset_target
from utils import show_patches_and_transformations, get_datasets_dir_dict
import openslide
from tqdm import tqdm
import sys
from math import isclose


class WSI_Master_Dataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 20,
                 slide_repetitions: int = 1):

        # Check if the target receptor is available for the requested train DataSet:
        assert_dataset_target(DataSet, target_kind)

        print('Initializing {} DataSet....'.format('Train' if train else 'Test'))

        # Define data root:
        self.ROOT_PATH = define_data_root(DataSet)

        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.DX = DX
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param

        # Get DataSets location:
        dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        print('Train Data will be taken from these locations:')
        print(dir_dict)
        locations_list = []

        for _, key in enumerate(dir_dict):
            locations_list.append(dir_dict[key])
            slide_meta_data_file = os.path.join(dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(dir_dict[key], 'Grids/Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)
        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

        if self.DataSet == 'LUNG':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        # Define number of tiles to be used
        if train_type == 'REG':
            n_minimal_tiles = n_tiles
        else:
            n_minimal_tiles = self.bag_size

        slides_with_few_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] < n_minimal_tiles])
        # FIXME: find a way to use slides with less than the minimal amount of slides. and than delete the following if.
        if len(slides_with_few_tiles) > 0:
            print(
                '{} Slides were excluded from DataSet because they had less than {} available tiles or are non legitimate for training'
                .format(len(slides_with_few_tiles), n_minimal_tiles))
        valid_slide_indices = np.array(
            list(set(valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'Breast':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type in ['REG', 'MIL']:
            if self.train:
                folds = list(self.meta_data_DF[fold_column_name].unique())
                folds.remove(self.test_fold)
                if 'test' in folds:
                    folds.remove('test')
                if 'val' in folds:
                    folds.remove('val')
            else:
                folds = [self.test_fold]
                folds.append('val')
        elif self.train_type == 'Infer':
            folds = infer_folds
        else:
            raise ValueError('Variable train_type is not defined')

        self.folds = folds

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])

        '''if DataSet == 'Breast':
            all_image_path_names = [os.path.join(dir_dict[ii], ii) for ii in self.meta_data_DF['id']]
        else:
            all_image_path_names = list(self.meta_data_DF['id'])'''

        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])
        if self.DataSet != 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type == 'Infer':
            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                # self.image_path_names.append(all_image_path_names[index])
                self.image_path_names.append(dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

                # Preload slides - improves speed during training.
                try:
                    # image_file = os.path.join(self.ROOT_PATH, all_image_path_names[index], all_image_file_names[index])
                    image_file = os.path.join(dir_dict[all_image_ids[index]], all_image_file_names[index])
                    self.slides.append(openslide.open_slide(image_file))
                    basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])
                    # grid_file = os.path.join(self.ROOT_PATH, all_image_path_names[index], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    grid_file = os.path.join(dir_dict[all_image_ids[index]], 'Grids',
                                             basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    with open(grid_file, 'rb') as filehandle:
                        grid_list = pickle.load(filehandle)
                        self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or it\'s Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform, self.scale_factor = define_transformations(transform_type, self.train, self.tile_size,
                                                                   self.color_param)


        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)


        '''if train_type == 'REG':
            self.factor = n_tiles
        elif train_type == 'MIL':
            self.factor = slide_repetitions
        
        if train is False and test_time_augmentation:
            self.factor = 4

        self.real_length = int(self.__len__() / self.factor)'''

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        tiles, time_list = _choose_data(grid_list=self.grid_lists[idx],
                                        slide=self.slides[idx],
                                        how_many=self.bag_size,
                                        magnification=self.magnification[idx],
                                        tile_size=int(self.tile_size / (1 - self.scale_factor)),
                                        # Fix boundaries with scale
                                        print_timing=self.print_time,
                                        desired_mag=self.desired_magnification)

        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])
            for i in range(self.bag_size):
                images[i] = trans(tiles[i])
        else:
            images = 0

        aug_time = (time.time() - start_aug) / self.bag_size
        total_time = time.time() - start_getitem
        if self.print_time:
            time_list = (time_list[0], time_list[1], aug_time, total_time)
        else:
            time_list = [0]

        # TODO: check what this function does
        debug_patches_and_transformations = False
        if debug_patches_and_transformations and images != 0:
            show_patches_and_transformations(X, images, tiles, self.scale_factor, self.tile_size)

        return X, label, time_list, self.image_file_names[idx], images


########################################################################################################################

class WSI_MILdataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 10,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 test_time_augmentation: bool = False,
                 desired_slide_magnification: int = 20,
                 slide_repetitions: int = 1
                 ):
        super(WSI_MILdataset, self).__init__(DataSet=DataSet,
                                             tile_size=tile_size,
                                             bag_size=bag_size,
                                             target_kind=target_kind,
                                             test_fold=test_fold,
                                             train=train,
                                             print_timing=print_timing,
                                             transform_type=transform_type,
                                             DX=DX,
                                             get_images=get_images,
                                             train_type='MIL',
                                             color_param=color_param,
                                             test_time_augmentation=test_time_augmentation,
                                             desired_slide_magnification=desired_slide_magnification,
                                             slide_repetitions=slide_repetitions)

        print(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))


class WSI_REGdataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 20
                 ):
        super(WSI_REGdataset, self).__init__(DataSet=DataSet,
                                             tile_size=tile_size,
                                             bag_size=1,
                                             target_kind=target_kind,
                                             test_fold=test_fold,
                                             train=train,
                                             print_timing=print_timing,
                                             transform_type=transform_type,
                                             DX=DX,
                                             get_images=get_images,
                                             train_type='REG',
                                             color_param=color_param,
                                             n_tiles=n_tiles,
                                             desired_slide_magnification=desired_slide_magnification)

        print(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.desired_magnification,
                        self.real_length,
                        self.tile_size,
                        self.bag_size,
                        'Without' if transform_type == 'none' else 'With',
                        self.test_fold,
                        'ON' if self.DX else 'OFF'))

    def __getitem__(self, idx):
        X, label, time_list, image_file_names, images = super(WSI_REGdataset, self).__getitem__(idx=idx)
        X = torch.reshape(X, (3, self.tile_size, self.tile_size))

        return X, label, time_list, image_file_names, images


class Infer_Dataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 dx: bool = False,
                 desired_slide_magnification: int = 20
                 ):
        super(Infer_Dataset, self).__init__(DataSet=DataSet,
                                            tile_size=tile_size,
                                            bag_size=None,
                                            target_kind=target_kind,
                                            test_fold=1,
                                            infer_folds=folds,
                                            train=True,
                                            print_timing=False,
                                            transform_type='none',
                                            DX=dx,
                                            get_images=False,
                                            train_type='Infer',
                                            desired_slide_magnification=desired_slide_magnification)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.magnification = []
        self.num_tiles = []
        self.slide_grids = []

        ind = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    # self.num_patches.append(self.all_tissue_tiles[slide_num])
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))  # RanS 10.3.21
                    print('{} Slide available tiles are less than {}'.format(self.all_image_file_names[slide_num],
                                                                             num_tiles))

                # self.magnification.extend([self.all_magnifications[slide_num]] * self.num_patches[-1])
                self.magnification.extend([self.all_magnifications[slide_num]])  # RanS 11.3.21

                basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                grid_file = os.path.join(self.image_path_names[ind], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')

                which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                with open(grid_file, 'rb') as filehandle:
                    grid_list = pickle.load(filehandle)
                chosen_locations = [grid_list[loc] for loc in which_patches]
                chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)

                ind += 1  # RanS 29.1.21

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = 0
        self.current_file = None
        print(
            'Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete. {} Slides, Working on Tiles of size {}^2. {} Tiles per slide, {} tiles per iteration, {} iterations to complete full inference'
                .format(self.DataSet,
                        self.target_kind,
                        str(self.folds),
                        len(self.image_file_names),
                        self.tile_size,
                        num_tiles,
                        self.tiles_per_iter,
                        self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.tiles_to_go = self.num_tiles[self.slide_num]
            '''self.current_file = os.path.join(self.ROOT_PATH, self.image_path_names[self.slide_num], self.image_file_names[self.slide_num])
            self.current_slide = openslide.open_slide(self.current_file)'''

            if sys.platform == 'win32':
                image_file = os.path.join(self.image_path_names[self.slide_num], self.image_file_names[self.slide_num])
                self.current_slide = openslide.open_slide(image_file)
            else:
                self.current_slide = self.slides[self.slide_num]

            self.initial_num_patches = self.num_tiles[self.slide_num]

            # RanS 11.3.21
            desired_downsample = self.magnification[self.slide_num] / self.desired_magnification

            level, best_next_level = -1, -1
            for index, downsample in enumerate(self.current_slide.level_downsamples):
                if isclose(desired_downsample, downsample, rel_tol=1e-3):
                    level = index
                    level_downsample = 1
                    break

                elif downsample < desired_downsample:
                    best_next_level = index
                    level_downsample = int(desired_downsample / self.current_slide.level_downsamples[best_next_level])

            self.adjusted_tile_size = self.tile_size * level_downsample
            self.best_slide_level = level if level > best_next_level else best_next_level
            self.level_0_tile_size = int(desired_downsample) * self.tile_size

        label = [1] if self.target[self.slide_num] == 'Positive' else [0]
        label = torch.LongTensor(label)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
            self.slide_num += 1
        else:
            self.tiles_to_go -= self.tiles_per_iter

        '''
        #adjusted_tile_size = self.tile_size * (self.magnification[idx] // self.BASIC_MAGNIFICATION)
        downsample = int(self.magnification[idx] / self.basic_magnification)
        adjusted_tile_size = int(self.tile_size * downsample) #RanS 30.12.20

        tiles, time_list = _get_tiles(self.current_slide, #RanS 9.2.21, preload slides
                                      self.slide_grids[idx],
                                      adjusted_tile_size,
                                      self.print_time,
                                      downsample)'''

        # desired_downsample = self.magnification[idx] / self.basic_magnification  # downsample needed for each dimension (reflected by level_downsamples property)
        # RanS 11.3.21
        '''desired_downsample = self.magnification[self.slide_num] / self.basic_magnification  # downsample needed for each dimension (reflected by level_downsamples property)

        level, best_next_level = -1, -1
        for index, downsample in enumerate(self.current_slide.level_downsamples):
            if isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = index
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                best_next_level = index
                level_downsample = int(desired_downsample / self.current_slide.level_downsamples[best_next_level])

        adjusted_tile_size = self.tile_size * level_downsample
        best_slide_level = level if level > best_next_level else best_next_level
        level_0_tile_size = int(desired_downsample) * self.tile_size'''

        tiles, time_list = _get_tiles(slide=self.current_slide,
                                      locations=self.slide_grids[idx],
                                      tile_size_level_0=self.level_0_tile_size,
                                      adjusted_tile_sz=self.adjusted_tile_size,
                                      output_tile_sz=self.tile_size,
                                      best_slide_level=self.best_slide_level)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        start_aug = time.time()
        for i in range(len(tiles)):
            X[i] = self.transform(tiles[i])

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

        debug_patches_and_transformations = False
        if debug_patches_and_transformations:
            images = torch.zeros_like(X)
            trans = transforms.Compose(
                [transforms.CenterCrop(self.tile_size), transforms.ToTensor()])  # RanS 21.12.20

            for i in range(self.tiles_per_iter):
                images[i] = trans(tiles[i])
            show_patches_and_transformations(X, images, tiles, self.scale_factor, self.tile_size)

        return X, label, time_list, last_batch, self.initial_num_patches, self.current_slide._filename
