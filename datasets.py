import numpy as np
import os
import pandas as pd
import pickle
from random import sample
import torch
from torchvision import transforms
import sys
import time
from torch.utils.data import Dataset
from typing import List
from utils import MyRotation, Cutout, _get_tiles, _choose_data, chunks
from utils import HEDColorJitter, define_transformations, define_data_root, assert_dataset_target
from utils import show_patches_and_transformations, get_breast_dir_dict
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
import openslide #RanS 9.2.21, preload slides
from tqdm import tqdm


MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
        'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
        'Ron': [0.8998, 0.8253, 0.9357]
        }

STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
       'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
       'Ron': [0.1125, 0.1751, 0.0787]
       }


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
                 c_param: float = 0.1,
                 n_patches: int = 50,
                 tta: bool = False,
                 mag: int = 20):

        print('Initializing {} DataSet....'.format('Train' if train else 'Test'))
        # Define data root:
        self.ROOT_PATH = define_data_root(DataSet)

        self.DataSet = DataSet
        #self.BASIC_MAGNIFICATION = 20
        self.basic_magnification = mag  #RanS 8.2.21
        if DataSet == 'RedSquares':
        #    self.BASIC_MAGNIFICATION = 10
            slides_data_file = 'slides_data_RedSquares.xlsx'
        else:
            slides_data_file = 'slides_data.xlsx'

        assert_dataset_target(DataSet, target_kind)

        meta_data_file = os.path.join(self.ROOT_PATH, slides_data_file)
        self.meta_data_DF = pd.read_excel(meta_data_file)

        if self.DataSet == 'Breast':
            dir_dict = get_breast_dir_dict()
        #if self.DataSet != 'ALL':
        else:
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['id'] == self.DataSet]
            self.meta_data_DF.reset_index(inplace=True)

        # RanS 12.1.21, this slide is buggy, avoid it
        self.meta_data_DF.loc[self.meta_data_DF['file'] == '18-1241_1_1_a.mrxs', 'ER status'] = 'Missing Data'
        self.meta_data_DF.loc[self.meta_data_DF['file'] == '18-1241_1_1_a.mrxs', 'PR status'] = 'Missing Data'
        self.meta_data_DF.loc[self.meta_data_DF['file'] == '18-1241_1_1_a.mrxs', 'Her2 status'] = 'Missing Data'
        self.meta_data_DF.reset_index(inplace=True)

        # for lung, take only origin:lung
        if self.DataSet == 'LUNG':
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            #self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Diagnosis'] == 'adenocarcinoma']
            #RanS 2.1.21, this slide is buggy, avoid it
            self.meta_data_DF.loc[self.meta_data_DF['file'] == '2019-27925.mrxs', 'PDL1 status'] = 'Missing Data'
            self.meta_data_DF.loc[self.meta_data_DF['file'] == '2019-27925.mrxs', 'EGFR status'] = 'Missing Data'
            self.meta_data_DF.reset_index(inplace=True)

        # self.meta_data_DF.set_index('id')
        self.tile_size = tile_size
        self.target_kind = target_kind
        self.test_fold = test_fold
        self.bag_size = bag_size
        self.train = train
        self.print_time = print_timing
        self.DX = DX
        self.get_images = get_images
        self.train_type = train_type
        self.c_param = c_param

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        # We'll use only the valid slides - the ones with a Negative or Positive label.
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(self.tile_size) + ' compatible @ X20'] == -1])
        if train_type == 'REG':
            n_minimal_patches = n_patches
        else:
            n_minimal_patches = bag_size
        slides_with_small_grid = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'] < n_minimal_patches])
        #valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid))
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid - slides_with_small_grid))

        # BUT...we want the train set to be a combination of all sets except the train set....Let's compute it:
        if DataSet == 'Breast':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        if self.train_type == 'REG' or train_type == 'MIL':
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
            raise ValueError('train_type is not defined')

        self.folds = folds

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])

        if DataSet == 'Breast':
            all_image_path_names = [os.path.join(dir_dict[ii], ii) for ii in self.meta_data_DF['id']]
        else:
            all_image_path_names = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X20'])
        if self.DataSet != 'TCGA':
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])
        #all_magnifications = list(self.meta_data_DF['Objective Power'])

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
        self.slides = [] #RanS 9.2.21, preload slides
        self.grid_lists = [] #RanS 9.2.21, preload slides

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                self.image_path_names.append(all_image_path_names[index])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])

                #RanS 9.2.21, preload slides
                try:
                    image_file = os.path.join(self.ROOT_PATH, self.image_path_names[-1], self.image_file_names[-1])
                    if sys.platform != 'win32':
                        self.slides.append(openslide.open_slide(image_file))
                    basic_file_name = '.'.join(self.image_file_names[-1].split('.')[:-1])
                    grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[-1], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    with open(grid_file, 'rb') as filehandle:
                        grid_list = pickle.load(filehandle)
                        self.grid_lists.append(grid_list)
                except:
                    print('aa')

        # Setting the transformation:
        self.transform, self.scale_factor = define_transformations(transform_type, self.train, MEAN, STD, self.tile_size, self.c_param)

        if train_type == 'REG':
            self.factor = n_patches
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train == False and tta:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

    def __len__(self):
        return len(self.target) * self.factor


    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length
        #cancelled RanS 9.2.21, preload slides
        '''basic_file_name = '.'.join(self.image_file_names[idx].split('.')[:-1])
        grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
        image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])'''
        if sys.platform == 'win32':
            image_file = os.path.join(self.ROOT_PATH, self.image_path_names[idx], self.image_file_names[idx])
            slide = openslide.open_slide(image_file)
        else:
            slide = self.slides[idx]

        tiles, time_list = _choose_data(grid_list=self.grid_lists[idx],
                                        slide=slide,
                                        how_many=self.bag_size,
                                        magnification=self.magnification[idx],
                                        tile_size=int(self.tile_size / (1 - self.scale_factor)),  # RanS 7.12.20, fix boundaries with scale
                                        print_timing=self.print_time,
                                        desired_mag=self.basic_magnification) #RanS 8.2.21


        #time1 = time.time()  # temp
        #print('time1:', str(time1 - start_getitem))  # temp

        label = [1] if self.target[idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        # X will hold the images after all the transformations
        X = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])

        #magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        #magnification_relation = self.magnification[idx] / self.BASIC_MAGNIFICATION
        #if magnification_relation != 1:
        '''
        if tile_sz != self.tile_size:
            transform = transforms.Compose([transforms.Resize(self.tile_size), self.transform])
        else:
            transform = self.transform
        '''
        start_aug = time.time()
        for i in range(self.bag_size):
            X[i] = self.transform(tiles[i])

        if self.get_images:
            images = torch.zeros([self.bag_size, 3, self.tile_size, self.tile_size])
            trans = transforms.Compose([transforms.CenterCrop(self.tile_size), transforms.ToTensor()])  # RanS 21.12.20
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

        debug_patches_and_transformations = False
        if debug_patches_and_transformations:
            show_patches_and_transformations(X, images, tiles, self.scale_factor, self.tile_size)

        return X, label, time_list, self.image_file_names[idx], images


class WSI_MILdataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 bag_size: int = 50,
                 target_kind: str = 'ER',
                 test_fold: int = 1,
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 DX: bool = False,
                 get_images: bool = False,
                 c_param: float = 0.1,
                 tta: bool = False,
                 mag: int = 20
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
                                             c_param=c_param,
                                             tta=tta,
                                             mag=mag)

        print(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
            .format(self.train_type,
                    'Train' if self.train else 'Test',
                    self.DataSet,
                    self.target_kind,
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
                 DX : bool = False,
                 get_images: bool = False,
                 c_param: float = 0.1,
                 n_patches: int = 50,
                 mag: int = 20
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
                                             c_param=c_param,
                                             n_patches=n_patches,
                                             mag=mag)

        print(
            'Initiation of WSI({}) {} {} DataSet for {} is Complete. Magnification is X{}, {} Slides, Tiles of size {}^2. {} tiles in a bag, {} Transform. TestSet is fold #{}. DX is {}'
                .format(self.train_type,
                        'Train' if self.train else 'Test',
                        self.DataSet,
                        self.target_kind,
                        self.basic_magnification,
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
                 mag: int = 20
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
                                            mag=mag)

        self.tiles_per_iter = tiles_per_iter
        self.folds = folds
        self.magnification = []
        self.num_patches = []
        self.slide_grids = []

        ind = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num]:
                    self.num_patches.append(num_tiles)
                else:
                    self.num_patches.append(self.all_tissue_tiles[slide_num])
                    print('{} Slide available tiles are less than {}'.format(self.all_image_file_names[slide_num], num_tiles))

                self.magnification.extend([self.all_magnifications[slide_num]] * self.num_patches[-1])

                basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                grid_file = os.path.join(self.ROOT_PATH, self.image_path_names[ind], 'Grids',
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')

                which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_patches[-1])

                with open(grid_file, 'rb') as filehandle:
                    grid_list = pickle.load(filehandle)
                chosen_locations = [grid_list[loc] for loc in which_patches]
                chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)

                ind += 1 #RanS 29.1.21

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
        return int(np.ceil(np.array(self.num_patches)/self.tiles_per_iter).sum())

    def __getitem__(self, idx):
        start_getitem = time.time()
        if self.tiles_to_go is None:
            self.tiles_to_go = self.num_patches[self.slide_num]
            self.current_file = os.path.join(self.ROOT_PATH, self.image_path_names[self.slide_num], self.image_file_names[self.slide_num])
            self.current_slide = openslide.open_slide(self.current_file)

            self.initial_num_patches = self.num_patches[self.slide_num]

        label = [1] if self.target[self.slide_num] == 'Positive' else [0]
        label = torch.LongTensor(label)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
            self.slide_num += 1
        else:
            self.tiles_to_go -= self.tiles_per_iter

        #adjusted_tile_size = self.tile_size * (self.magnification[idx] // self.BASIC_MAGNIFICATION)
        downsample = int(self.magnification[idx] / self.basic_magnification)
        adjusted_tile_size = int(self.tile_size * downsample) #RanS 30.12.20
        #tiles, time_list, tile_sz = _get_tiles(self.current_file,
        tiles, time_list, tile_sz = _get_tiles(self.current_slide, #RanS 9.2.21, preload slides
                                      self.slide_grids[idx],
                                      adjusted_tile_size,
                                      self.print_time,
                                      downsample)

        X = torch.zeros([len(tiles), 3, self.tile_size, self.tile_size])

        #magnification_relation = self.magnification[idx] // self.BASIC_MAGNIFICATION
        #if downsample != 1:
        if tile_sz != self.tile_size:
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

        debug_patches_and_transformations = False
        if debug_patches_and_transformations:
            images = torch.zeros_like(X)
            trans = transforms.Compose(
                [transforms.CenterCrop(self.tile_size), transforms.ToTensor()])  # RanS 21.12.20
            #if magnification_relation != 1:
            if tile_sz != self.tile_size:
                trans = transforms.Compose([transforms.Resize(self.tile_size), trans])
            for i in range(self.tiles_per_iter):
                images[i] = trans(tiles[i])
            show_patches_and_transformations(X, images, tiles, self.scale_factor, self.tile_size)

        return X, label, time_list, last_batch, self.initial_num_patches, self.current_file









