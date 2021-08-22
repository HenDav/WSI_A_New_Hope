import numpy as np
import os
import pandas as pd
import pickle
from random import sample, choices
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import time
from torch.utils.data import Dataset
from typing import List
from utils import MyRotation, Cutout, _get_tiles, _choose_data, chunks, map_original_grid_list_to_equiv_grid_list
from utils import HEDColorJitter, define_transformations, assert_dataset_target
from utils import show_patches_and_transformations, get_datasets_dir_dict
import openslide
from tqdm import tqdm
import sys
from math import isclose
from PIL import Image
from glob import glob


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
                 slide_repetitions: int = 1,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False
                 ):

        # Check if the target receptor is available for the requested train DataSet:
        assert_dataset_target(DataSet, target_kind)

        print('Initializing {} DataSet....'.format('Train' if train else 'Test'))
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
        self.loan = loan

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        print('Slide Data will be taken from these locations:')
        print(self.dir_dict)
        locations_list = []

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids', 'Grid_data.xlsx')

            slide_meta_data_DF = pd.read_excel(slide_meta_data_file)
            grid_meta_data_DF = pd.read_excel(grid_meta_data_file)
            meta_data_DF = pd.DataFrame({**slide_meta_data_DF.set_index('file').to_dict(),
                                         **grid_meta_data_DF.set_index('file').to_dict()})

            self.meta_data_DF = meta_data_DF if not hasattr(self, 'meta_data_DF') else self.meta_data_DF.append(
                meta_data_DF)
        self.meta_data_DF.reset_index(inplace=True)
        self.meta_data_DF.rename(columns={'index': 'file'}, inplace=True)

        if self.DataSet == 'PORTO_HE' or self.DataSet == 'PORTO_PDL1':
            # for lung, take only origin: lung
            self.meta_data_DF = self.meta_data_DF[self.meta_data_DF['Origin'] == 'lung']
            self.meta_data_DF.reset_index(inplace=True) #RanS 18.4.21

        if self.target_kind=='OR':
            PR_targets = list(self.meta_data_DF['PR status'])
            ER_targets = list(self.meta_data_DF['ER status'])
            all_targets = ['Missing Data']*len(ER_targets)
            for ii, (PR_target, ER_target) in enumerate(zip(PR_targets, ER_targets)):
                if (PR_target == 'Positive' or ER_target == 'Positive'):
                    all_targets[ii] = 'Positive'
                elif (PR_target == 'Negative' or ER_target == 'Negative'): #avoid 'Missing Data'
                    all_targets[ii] = 'Negative'
        else:
            all_targets = list(self.meta_data_DF[self.target_kind + ' status'])

        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        #RanS 17.8.21
        if slide_per_block:
            if DataSet == 'CARMEL':
                #all_patient_barcodes1 = ['17-5_1_1_a', '17-5_1_1_b', '18-81_1_2_e', '18-81_1_2_k', '17-10008_1_1_e'] ######temp

                all_blocks = []
                for barcode in all_patient_barcodes:
                    if barcode is not np.nan:
                        all_blocks.append(barcode[:-2])
                    else:
                        all_blocks.append(barcode)

                _, unique_inds = np.unique(all_blocks, return_index=True)
                all_inds = np.arange(0, len(all_blocks))
                excess_block_slides = set(all_inds) - set(unique_inds)
                print('slide_per_block: removing ' + str(len(excess_block_slides)) + ' slides')
            else:
                IOError('slide_per_block only implemented for CARMEL dataset')
        else:
            excess_block_slides = set()

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1])  #RanS 9.5.21
        else:
            slides_with_bad_seg = set()

        # train only on samples with ER=PR, RanS 27.6.21
        if er_eq_pr and self.train:
            slides_with_er_not_eq_pr = set(self.meta_data_DF.index[self.meta_data_DF['ER status'] != self.meta_data_DF['PR status']])
        else:
            slides_with_er_not_eq_pr = set()

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
            list(set(valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg - slides_with_er_not_eq_pr - excess_block_slides))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
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

        if type(folds) is int:
            folds = [folds]

        correct_folds = self.meta_data_DF[fold_column_name][valid_slide_indices].isin(folds)
        valid_slide_indices = np.array(correct_folds.index[correct_folds])

        all_image_file_names = list(self.meta_data_DF['file'])
        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])
        #if self.DataSet != 'TCGA':
        if 'TCGA' not in self.dir_dict:
            self.DX = False
        if self.DX:
            all_is_DX_cut = list(self.meta_data_DF['DX'])

        all_magnifications = list(self.meta_data_DF['Manipulated Objective Power'])

        if train_type == 'Infer':
            temp_select = False
            if temp_select: #RanS 10.8.21, hard coded selection of specific slides
                slidenames = ['19-14722_1_1_a.mrxs', '19-14722_1_1_b.mrxs', '19-14722_1_1_e.mrxs', '19-5229_2_1_a.mrxs', '19-5229_2_1_b.mrxs', '19-5229_2_1_e.mrxs']
                valid_slide_indices = []
                for slidename in slidenames:
                    valid_slide_index = self.meta_data_DF[self.meta_data_DF['file']==slidename].index.to_list()
                    valid_slide_indices.append(valid_slide_index[0])

            self.valid_slide_indices = valid_slide_indices
            self.all_magnifications = all_magnifications
            self.all_is_DX_cut = all_is_DX_cut if self.DX else [True] * len(self.all_magnifications)
            self.all_tissue_tiles = all_tissue_tiles
            self.all_image_file_names = all_image_file_names
            self.all_patient_barcodes = all_patient_barcodes

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                # self.image_path_names.append(all_image_path_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])
                #self.presaved_tiles.append(all_image_ids[index] == 'ABCTB') #cancelled RanS 19.7.21 - use tif instead of tiles
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')

                # Preload slides - improves speed during training.
                grid_file = []
                image_file = []
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles', '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        self.slides.append(openslide.open_slide(image_file))
                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])

                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]], 'Grids',
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or its Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        if self.presaved_tiles[idx]:  # load presaved patches
            time_tile_extraction = time.time()
            idxs = sample(range(self.tissue_tiles[idx]), self.bag_size)
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * self.bag_size
            for ii, tile_ind in enumerate(idxs):
                #tile_path = os.path.join(self.tiles_dir[idx], 'tile_' + str(tile_ind) + '.data')
                tile_path = os.path.join(self.slides[idx], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
                time_tile_extraction = (time.time() - time_tile_extraction) / len(idxs)
                time_list = [0, time_tile_extraction]
        else:
            slide = self.slides[idx]

            tiles, time_list, label = _choose_data(grid_list=self.grid_lists[idx],
                                            slide=slide,
                                            how_many=self.bag_size,
                                            magnification=self.magnification[idx],
                                            #tile_size=int(self.tile_size / (1 - self.scale_factor)),
                                            tile_size=self.tile_size, #RanS 28.4.21, scale out cancelled for simplicity
                                            # Fix boundaries with scale
                                            print_timing=self.print_time,
                                            desired_mag=self.desired_magnification,
                                            loan=self.loan)

        if not self.loan:
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
                 desired_slide_magnification: int = 10,
                 loan: bool = False,
                 er_eq_pr: bool = False,
                 slide_per_block: bool = False
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
                                             desired_slide_magnification=desired_slide_magnification,
                                             er_eq_pr=er_eq_pr,
                                             slide_per_block=slide_per_block)

        self.loan = loan
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
                 desired_slide_magnification: int = 10
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
        self.grid_lists = []
        self.patient_barcode = []
        #self.equivalent_grid = []
        #self.equivalent_grid_size = []

        ind = 0
        slide_with_not_enough_tiles = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:

                '''# Recreate the basic slide grids:
                height = int(self.meta_data_DF.loc[slide_num, 'Height'])
                width = int(self.meta_data_DF.loc[slide_num, 'Width'])
                objective_power = self.meta_data_DF.loc[slide_num, 'Manipulated Objective Power']

                adjusted_tile_size_at_level_0 = int(
                    self.tile_size * (int(objective_power) / self.desired_magnification))
                equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
                equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))
                basic_grid = [(row, col) for row in range(0, height, adjusted_tile_size_at_level_0) for col in
                              range(0, width, adjusted_tile_size_at_level_0)]
                equivalent_grid_dimensions = (equivalent_rows, equivalent_cols)
                self.equivalent_grid_size.append(equivalent_grid_dimensions)'''


                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    # self.num_patches.append(self.all_tissue_tiles[slide_num])
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))  # RanS 10.3.21
                    slide_with_not_enough_tiles += 1
                    '''print('{} Slide available tiles are less than {}'.format(self.all_image_file_names[slide_num],
                                                                             num_tiles))'''

                # self.magnification.extend([self.all_magnifications[slide_num]] * self.num_patches[-1])
                self.magnification.extend([self.all_magnifications[slide_num]])  # RanS 11.3.21
                self.patient_barcode.append(self.all_patient_barcodes[slide_num])
                which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                if self.presaved_tiles[ind]:
                    self.grid_lists.append(0)
                else:
                    basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                    grid_file = os.path.join(self.image_path_names[ind], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    with open(grid_file, 'rb') as filehandle:
                        grid_list = pickle.load(filehandle)
                        self.grid_lists.append(grid_list)
                    #chosen_locations = [grid_list[loc] for loc in which_patches] #moved to get_item, RanS 5.5.21

                #chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                patch_ind_chunks = chunks(which_patches, self.tiles_per_iter) #RanS 5.5.21
                self.slide_grids.extend(patch_ind_chunks)

                #equivalent_grid_list = map_original_grid_list_to_equiv_grid_list(adjusted_tile_size_at_level_0, grid_list)
                #self.equivalent_grid.append(equivalent_grid_list)

                ind += 1  # RanS 29.1.21

        print('There are {} slides with less than {} tiles'.format(slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
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
            self.slide_num += 1  #RanS 5.5.21
            self.tiles_to_go = self.num_tiles[self.slide_num]

            self.current_slide = self.slides[self.slide_num]
            #self.slide_size = self.current_slide.dimensions[::-1]
            self.initial_num_patches = self.num_tiles[self.slide_num]

            if not self.presaved_tiles[self.slide_num]: #RanS 5.5.21
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


        if self.presaved_tiles[self.slide_num]: #RanS 5.5.21
            idxs = self.slide_grids[idx]
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * len(idxs)
            for ii, tile_ind in enumerate(idxs):
                # tile_path = os.path.join(self.tiles_dir[idx], 'tile_' + str(tile_ind) + '.data')
                tile_path = os.path.join(self.slides[self.slide_num], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
        else:
            #locs_ind = [self.equivalent_grid[self.slide_num][loc] for loc in self.slide_grids[idx]]
            locs = [self.grid_lists[self.slide_num][loc] for loc in self.slide_grids[idx]]
            tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                          #locations=self.slide_grids[idx],
                                          locations=locs,
                                          tile_size_level_0=self.level_0_tile_size,
                                          adjusted_tile_sz=self.adjusted_tile_size,
                                          output_tile_sz=self.tile_size,
                                          best_slide_level=self.best_slide_level,
                                          random_shift=True)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
            #self.slide_num += 1 #moved RanS 5.5.21
        else:
            self.tiles_to_go -= self.tiles_per_iter

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

        #return X, label, time_list, last_batch, self.initial_num_patches, self.current_slide._filename
        #return X, label, time_list, last_batch, self.initial_num_patches, self.image_file_names[self.slide_num] #RanS 5.5.21
        #return X, label, time_list, last_batch, self.initial_num_patches, self.image_file_names[self.slide_num], self.patient_barcode[self.slide_num]  #Omer 24/5/21

        return {'Data': X,
                'Label': label,
                'Time List': time_list,
                'Is Last Batch': last_batch,
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.image_file_names[self.slide_num],
                #'Patch loc index': locs_ind,
                'Patient barcode': self.patient_barcode[self.slide_num],
                #'Slide Index Size': self.equivalent_grid_size[self.slide_num],
                #'Slide Size': self.slide_size,
                #'Patch Loc': locs,
                }



class Full_Slide_Inference_Dataset(WSI_Master_Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 dx: bool = False,
                 desired_slide_magnification: int = 10,
                 num_background_tiles: int = 200
                 ):
        super(Full_Slide_Inference_Dataset, self).__init__(DataSet=DataSet,
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
        self.equivalent_grid = []
        self.equivalent_grid_size = []
        self.is_tissue_tiles = []
        self.is_last_batch = []

        ind = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                # Recreate the basic slide grids:
                height = int(self.meta_data_DF.loc[slide_num, 'Height'])
                width = int(self.meta_data_DF.loc[slide_num, 'Width'])
                objective_power = self.meta_data_DF.loc[slide_num, 'Manipulated Objective Power']

                adjusted_tile_size_at_level_0 = int(self.tile_size * (int(objective_power) / self.desired_magnification))
                equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
                equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))
                basic_grid = [(row, col) for row in range(0, height, adjusted_tile_size_at_level_0) for col in
                              range(0, width, adjusted_tile_size_at_level_0)]
                equivalent_grid_dimensions = (equivalent_rows, equivalent_cols)
                self.equivalent_grid_size.append(equivalent_grid_dimensions)

                if len(basic_grid) != self.meta_data_DF.loc[slide_num, 'Total tiles - ' + str(self.tile_size) + ' compatible @ X' + str(self.desired_magnification)].item():
                    raise Exception('Total tile num do not fit')

                self.magnification.extend([self.all_magnifications[slide_num]])

                basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                grid_file = os.path.join(self.image_path_names[ind], 'Grids',
                                         basic_file_name + '--tlsz' + str(self.tile_size) + '.data')

                #which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                with open(grid_file, 'rb') as filehandle:
                    tissue_grid_list = pickle.load(filehandle)
                # Compute wich tiles are background tiles and pick "num_background_tiles" from them.
                non_tissue_grid_list = list(set(basic_grid) - set(tissue_grid_list))
                selected_non_tissue_grid_list = sample(non_tissue_grid_list, num_background_tiles)
                # Combine the list of selected non tissue tiles with the list of all tiles:
                combined_grid_list = selected_non_tissue_grid_list + tissue_grid_list
                combined_equivalent_grid_list = map_original_grid_list_to_equiv_grid_list(adjusted_tile_size_at_level_0, combined_grid_list)
                # We'll also create a list that says which tiles are tissue tiles:
                is_tissue_tile = [False] * len(selected_non_tissue_grid_list) + [True] * len(tissue_grid_list)

                self.num_tiles.append(len(combined_grid_list))

                chosen_locations_chunks = chunks(combined_grid_list, self.tiles_per_iter)
                self.slide_grids.extend(chosen_locations_chunks)

                chosen_locations_equivalent_grid_chunks = chunks(combined_equivalent_grid_list, self.tiles_per_iter)
                self.equivalent_grid.extend(chosen_locations_equivalent_grid_chunks)

                is_tissue_tile_chunks = chunks(is_tissue_tile, self.tiles_per_iter)
                self.is_tissue_tiles.extend(is_tissue_tile_chunks)
                self.is_last_batch.extend([False] * (len(chosen_locations_chunks) - 1))
                self.is_last_batch.append(True)

                ind += 1

        # The following properties will be used in the __getitem__ function
        #self.tiles_to_go = None
        #self.new_slide = True
        self.slide_num = -1
        self.current_file = None
        print(
            'Initiation of WSI INFERENCE for {} DataSet and {} of folds {} is Complete. {} Slides, Working on Tiles of size {}^2 with X{} magnification. {} tiles per iteration, {} iterations to complete full inference'
                .format(self.DataSet,
                        self.target_kind,
                        str(self.folds),
                        len(self.image_file_names),
                        self.tile_size,
                        self.desired_magnification,
                        self.tiles_per_iter,
                        self.__len__()))

    def __len__(self):
        return int(np.ceil(np.array(self.num_tiles) / self.tiles_per_iter).sum())

    def __getitem__(self, idx, location: List = None):
        start_getitem = time.time()
        if idx == 0 or (idx > 0 and self.is_last_batch[idx - 1]):
            #self.new_slide = True
            self.slide_num += 1

            #self.new_slide = False
            #self.tiles_to_go = self.num_tiles[self.slide_num]

            if sys.platform == 'win32':
                image_file = os.path.join(self.image_path_names[self.slide_num],
                                          self.image_file_names[self.slide_num])
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
                    level_downsample = int(
                        desired_downsample / self.current_slide.level_downsamples[best_next_level])

            self.tile_size = 2048  # FIXME: if you want 256 tiles or 1024
            self.adjusted_tile_size = self.tile_size * level_downsample

            self.best_slide_level = level if level > best_next_level else best_next_level
            self.level_0_tile_size = int(desired_downsample) * self.tile_size

        label = [1] if self.target[self.slide_num] == 'Positive' else [0]
        label = torch.LongTensor(label)

        tile_locations = self.slide_grids[idx] if location is None else location

        tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                         locations=tile_locations,
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

        debug_patches_and_transformations = False
        if debug_patches_and_transformations:
            images = torch.zeros_like(X)
            trans = transforms.Compose(
                [transforms.CenterCrop(self.tile_size), transforms.ToTensor()])  # RanS 21.12.20

            for i in range(self.tiles_per_iter):
                images[i] = trans(tiles[i])
            show_patches_and_transformations(X, images, tiles, self.scale_factor, self.tile_size)

        #return X, label, time_list, self.is_last_batch[idx], self.initial_num_patches, self.current_slide._filename, self.equivalent_grid[idx], self.is_tissue_tiles[idx], self.equivalent_grid_size[self.slide_num]
        return {'Data': X,
                'Label': label,
                'Time List': time_list,
                'Is Last Batch': self.is_last_batch[idx],
                'Initial Num Tiles': self.initial_num_patches,
                'Slide Filename': self.current_slide._filename,
                'Equivalent Grid': self.equivalent_grid[idx],
                'Is Tissue Tiles': self.is_tissue_tiles[idx],
                'Equivalent Grid Size': self.equivalent_grid_size[self.slide_num],
                'Level 0 Locations': self.slide_grids[idx],
                'Original Data': transforms.ToTensor()(tiles[0])
                }


class Infer_Dataset_Background(WSI_Master_Dataset):
    """
    This dataset was created on 21/7/2021 to support extraction of background tiles in order to check the features and
    scores that they get from the model
    """
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 tiles_per_iter: int = 500,
                 target_kind: str = 'ER',
                 folds: List = [1],
                 num_tiles: int = 500,
                 dx: bool = False,
                 desired_slide_magnification: int = 10
                 ):
        super(Infer_Dataset_Background, self).__init__(DataSet=DataSet,
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
        self.grid_lists = []
        self.patient_barcode = []

        ind = 0
        slide_with_not_enough_tiles = 0
        for _, slide_num in enumerate(self.valid_slide_indices):
            if (self.DX and self.all_is_DX_cut[slide_num]) or not self.DX:
                if num_tiles <= self.all_tissue_tiles[slide_num] and self.all_tissue_tiles[slide_num] > 0:
                    self.num_tiles.append(num_tiles)
                else:
                    # self.num_patches.append(self.all_tissue_tiles[slide_num])
                    self.num_tiles.append(int(self.all_tissue_tiles[slide_num]))  # RanS 10.3.21
                    slide_with_not_enough_tiles += 1
                    '''print('{} Slide available tiles are less than {}'.format(self.all_image_file_names[slide_num],
                                                                             num_tiles))'''

                # self.magnification.extend([self.all_magnifications[slide_num]] * self.num_patches[-1])
                self.magnification.extend([self.all_magnifications[slide_num]])  # RanS 11.3.21
                self.patient_barcode.append(self.all_patient_barcodes[slide_num])
                which_patches = sample(range(int(self.tissue_tiles[ind])), self.num_tiles[-1])

                if self.presaved_tiles[ind]:
                    self.grid_lists.append(0)
                else:
                    basic_file_name = '.'.join(self.image_file_names[ind].split('.')[:-1])
                    grid_file = os.path.join(self.image_path_names[ind], 'Grids', basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                    with open(grid_file, 'rb') as filehandle:
                        grid_list = pickle.load(filehandle)
                        self.grid_lists.append(grid_list)
                    #chosen_locations = [grid_list[loc] for loc in which_patches] #moved to get_item, RanS 5.5.21

                #chosen_locations_chunks = chunks(chosen_locations, self.tiles_per_iter)
                patch_ind_chunks = chunks(which_patches, self.tiles_per_iter) #RanS 5.5.21
                self.slide_grids.extend(patch_ind_chunks)

                ind += 1  # RanS 29.1.21

        print('There are {} slides with less than {} tiles'.format(slide_with_not_enough_tiles, num_tiles))

        # The following properties will be used in the __getitem__ function
        self.tiles_to_go = None
        self.slide_num = -1
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
            self.slide_num += 1  #RanS 5.5.21
            self.tiles_to_go = self.num_tiles[self.slide_num]

            self.current_slide = self.slides[self.slide_num]

            self.initial_num_patches = self.num_tiles[self.slide_num]

            if not self.presaved_tiles[self.slide_num]: #RanS 5.5.21
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


        if self.presaved_tiles[self.slide_num]: #RanS 5.5.21
            idxs = self.slide_grids[idx]
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * len(idxs)
            for ii, tile_ind in enumerate(idxs):
                # tile_path = os.path.join(self.tiles_dir[idx], 'tile_' + str(tile_ind) + '.data')
                tile_path = os.path.join(self.slides[self.slide_num], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
        else:
            locs = [self.grid_lists[self.slide_num][loc] for loc in self.slide_grids[idx]]
            tiles, time_list, _ = _get_tiles(slide=self.current_slide,
                                          #locations=self.slide_grids[idx],
                                          locations=locs,
                                          tile_size_level_0=self.level_0_tile_size,
                                          adjusted_tile_sz=self.adjusted_tile_size,
                                          output_tile_sz=self.tile_size,
                                          best_slide_level=self.best_slide_level,
                                          random_shift=True)

        if self.tiles_to_go <= self.tiles_per_iter:
            self.tiles_to_go = None
            #self.slide_num += 1 #moved RanS 5.5.21
        else:
            self.tiles_to_go -= self.tiles_per_iter

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

        #return X, label, time_list, last_batch, self.initial_num_patches, self.current_slide._filename
        #return X, label, time_list, last_batch, self.initial_num_patches, self.image_file_names[self.slide_num] #RanS 5.5.21
        return X, label, time_list, last_batch, self.initial_num_patches, self.image_file_names[self.slide_num], self.patient_barcode[self.slide_num]  #Omer 24/5/21


class WSI_Segmentation_Master_Dataset(Dataset):
    def __init__(self,
                 DataSet: str = 'TCGA',
                 tile_size: int = 256,
                 test_fold: int = 1,
                 infer_folds: List = [None],
                 train: bool = True,
                 print_timing: bool = False,
                 transform_type: str = 'flip',
                 get_images: bool = False,
                 train_type: str = 'MASTER',
                 color_param: float = 0.1,
                 n_tiles: int = 10,
                 desired_slide_magnification: int = 10,
                 slide_repetitions: int = 1):

        print('Initializing {} DataSet for Segmentation....'.format('Train' if train else 'Test'))

        self.DataSet = DataSet
        self.desired_magnification = desired_slide_magnification
        self.tile_size = tile_size
        self.test_fold = test_fold
        self.train = train
        self.print_time = print_timing
        self.get_images = get_images
        self.train_type = train_type
        self.color_param = color_param

        # Get DataSets location:
        self.dir_dict = get_datasets_dir_dict(Dataset=self.DataSet)
        print('Slide Data will be taken from these locations:')
        print(self.dir_dict)
        locations_list = []

        for _, key in enumerate(self.dir_dict):
            locations_list.append(self.dir_dict[key])

            slide_meta_data_file = os.path.join(self.dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_meta_data_file = os.path.join(self.dir_dict[key], 'Grids', 'Grid_data.xlsx')

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
            self.meta_data_DF.reset_index(inplace=True) #RanS 18.4.21

        all_targets = list(self.meta_data_DF[self.target_kind + ' status'])
        all_patient_barcodes = list(self.meta_data_DF['patient barcode'])

        # We'll use only the valid slides - the ones with a Negative or Positive label. (Some labels have other values)
        # Let's compute which slides are these:
        valid_slide_indices = np.where(np.isin(np.array(all_targets), ['Positive', 'Negative']) == True)[0]

        # Also remove slides without grid data:
        slides_without_grid = set(self.meta_data_DF.index[self.meta_data_DF['Total tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == -1])
        # Remove slides with 0 tiles:
        slides_with_0_tiles = set(self.meta_data_DF.index[self.meta_data_DF['Legitimate tiles - ' + str(
            self.tile_size) + ' compatible @ X' + str(self.desired_magnification)] == 0])

        if 'bad segmentation' in self.meta_data_DF.columns:
            slides_with_bad_seg = set(self.meta_data_DF.index[self.meta_data_DF['bad segmentation'] == 1]) #RanS 9.5.21
        else:
            slides_with_bad_seg = set()

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
            list(set(valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg))

        # The train set should be a combination of all sets except the test set and validation set:
        if self.DataSet == 'CAT' or self.DataSet == 'ABCTB_TCGA':
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

        all_image_ids = list(self.meta_data_DF['id'])

        all_in_fold = list(self.meta_data_DF[fold_column_name])
        all_tissue_tiles = list(self.meta_data_DF['Legitimate tiles - ' + str(self.tile_size) + ' compatible @ X' + str(
            self.desired_magnification)])

        if 'TCGA' not in self.dir_dict:
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
            self.all_patient_barcodes = all_patient_barcodes

        self.image_file_names = []
        self.image_path_names = []
        self.in_fold = []
        self.tissue_tiles = []
        self.target = []
        self.magnification = []
        self.slides = []
        self.grid_lists = []
        self.presaved_tiles = []

        for _, index in enumerate(tqdm(valid_slide_indices)):
            if (self.DX and all_is_DX_cut[index]) or not self.DX:
                self.image_file_names.append(all_image_file_names[index])
                # self.image_path_names.append(all_image_path_names[index])
                self.image_path_names.append(self.dir_dict[all_image_ids[index]])
                self.in_fold.append(all_in_fold[index])
                self.tissue_tiles.append(all_tissue_tiles[index])
                self.target.append(all_targets[index])
                self.magnification.append(all_magnifications[index])
                #self.presaved_tiles.append(all_image_ids[index] == 'ABCTB')
                self.presaved_tiles.append(all_image_ids[index] == 'ABCTB_TILES')

                # Preload slides - improves speed during training.
                try:
                    image_file = os.path.join(self.dir_dict[all_image_ids[index]], all_image_file_names[index])
                    if self.presaved_tiles[-1]:
                        tiles_dir = os.path.join(self.dir_dict[all_image_ids[index]], 'tiles', '.'.join((os.path.basename(image_file)).split('.')[:-1]))
                        self.slides.append(tiles_dir)
                        self.grid_lists.append(0)
                    else:
                        self.slides.append(openslide.open_slide(image_file))
                        basic_file_name = '.'.join(all_image_file_names[index].split('.')[:-1])
                        grid_file = os.path.join(self.dir_dict[all_image_ids[index]], 'Grids',
                                                 basic_file_name + '--tlsz' + str(self.tile_size) + '.data')
                        with open(grid_file, 'rb') as filehandle:
                            grid_list = pickle.load(filehandle)
                            self.grid_lists.append(grid_list)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        'Couldn\'t open slide {} or it\'s Grid file {}'.format(image_file, grid_file))

        # Setting the transformation:
        self.transform = define_transformations(transform_type, self.train, self.tile_size, self.color_param)
        if np.sum(self.presaved_tiles):
            self.rand_crop = transforms.RandomCrop(self.tile_size)

        if train_type == 'REG':
            self.factor = n_tiles
            self.real_length = int(self.__len__() / self.factor)
        elif train_type == 'MIL':
            self.factor = 1
            self.real_length = self.__len__()
        if train is False and test_time_augmentation:
            self.factor = 4
            self.real_length = int(self.__len__() / self.factor)

    def __len__(self):
        return len(self.target) * self.factor

    def __getitem__(self, idx):
        start_getitem = time.time()
        idx = idx % self.real_length

        if self.presaved_tiles[idx]:  # load presaved patches
            time_tile_extraction = time.time()
            idxs = sample(range(self.tissue_tiles[idx]), self.bag_size)
            empty_image = Image.fromarray(np.uint8(np.zeros((self.tile_size, self.tile_size, 3))))
            tiles = [empty_image] * self.bag_size
            for ii, tile_ind in enumerate(idxs):
                #tile_path = os.path.join(self.tiles_dir[idx], 'tile_' + str(tile_ind) + '.data')
                tile_path = os.path.join(self.slides[idx], 'tile_' + str(tile_ind) + '.data')
                with open(tile_path, 'rb') as fh:
                    header = fh.readline()
                    tile_bin = fh.read()
                dtype, w, h, c = header.decode('ascii').strip().split()
                tile = np.frombuffer(tile_bin, dtype=dtype).reshape((int(w), int(h), int(c)))
                tile1 = self.rand_crop(Image.fromarray(tile))
                tiles[ii] = tile1
                time_tile_extraction = (time.time() - time_tile_extraction) / len(idxs)
                time_list = [0, time_tile_extraction]
        else:
            slide = self.slides[idx]

            tiles, time_list, _ = _choose_data(grid_list=self.grid_lists[idx],
                                            slide=slide,
                                            how_many=self.bag_size,
                                            magnification=self.magnification[idx],
                                            #tile_size=int(self.tile_size / (1 - self.scale_factor)),
                                            tile_size=self.tile_size, #RanS 28.4.21, scale out cancelled for simplicity
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


class Features_MILdataset(Dataset):
    def __init__(self,
                 data_location: str = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/',
                 bag_size: int = 100,
                 minimum_tiles_in_slide: int = 50,
                 is_per_patient: bool = False,  # if True than data will be gathered and returned per patient (else per slide)
                 is_all_tiles: bool = False,  # if True than all slide tiles will be used (else only bag_tiles number of tiles)
                 fixed_tile_num: int = None,
                 is_repeating_tiles: bool = True,  # if True than the tile pick from each slide/patient is with repeating elements
                 target: str = 'ER',
                 is_train: bool = False,
                 data_limit: int = None,  # if 'None' than there will be no data limit. If a number is specified than it'll be the data limit
                 test_fold: int = None,
                 print_timing: bool = False,
                 slide_repetitions: int = 1
                 ):

        self.is_per_patient, self.is_all_tiles, self.is_repeating_tiles = is_per_patient, is_all_tiles, is_repeating_tiles
        self.bag_size = bag_size
        self.train_type = 'Features'

        self.slide_names = []
        self.labels = []
        self.targets = []
        self.features = []
        self.num_tiles = []
        self.scores = []
        self.tile_scores = []
        self.patient_data = {}
        self.bad_patient_list = []
        self.fixed_tile_num = fixed_tile_num  # This instance variable indicates what is the number of fixed tiles to be used. if "None" than all tiles will be used. This feature is used to check the necessity in using more than 500 feature tiles for training
        slides_from_same_patient_with_different_target_values, total_slides, bad_num_of_good_tiles = 0, 0, 0
        slides_with_not_enough_tiles, slides_with_bad_segmentation = 0, 0
        patient_list = []

        data_files = glob(os.path.join(data_location, '*.data'))
        if os.path.join(data_location, 'Model_Epoch_1000-Folds_[2, 3, 4, 5]_ER-Tiles_500.data') in data_files:
            data_files.remove(os.path.join(data_location, 'Model_Epoch_1000-Folds_[2, 3, 4, 5]_ER-Tiles_500.data'))
        if os.path.join(data_location, 'Model_Epoch_1000-Folds_[1]_ER-Tiles_500.data') in data_files:
            data_files.remove(os.path.join(data_location, 'Model_Epoch_1000-Folds_[1]_ER-Tiles_500.data'))

        if sys.platform == 'darwin':
            '''
            grid_location_dict = {
                'TCGA': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/TCGA_Grid_data.xlsx',
                'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_Grid_data.xlsx'}
            '''
            if target in ['ER', 'ER_Features'] or (target in ['PR', 'PR_Features', 'Her2', 'Her2_Features'] and test_fold == 1):
                grid_location_dict = {'TCGA': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/TCGA_Grid_data.xlsx',
                                      'ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Grids_data/ABCTB_Grid_data.xlsx'}
            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        elif sys.platform == 'linux':
            '''
            grid_location_dict = {'TCGA': r'/mnt/gipmed_new/Data/Breast/TCGA/Grids/Grid_data.xlsx',
                                  'ABCTB': r'/mnt/gipmed_new/Data/Breast/ABCTB/ABCTB/Grids/Grid_data.xlsx'}
            '''
            if target in ['ER', 'ER_Features'] or (target in ['PR', 'PR_Features', 'Her2', 'Her2_Features'] and test_fold == 1):
                grid_location_dict = {'TCGA': r'/mnt/gipmed_new/Data/Breast/TCGA/Grids/Grid_data.xlsx',
                                      'ABCTB': r'/mnt/gipmed_new/Data/Breast/ABCTB/ABCTB/Grids/Grid_data.xlsx'}
            else:
                raise Exception("Need to write which dictionaries to use in this receptor case")

        grid_DF = pd.DataFrame()
        for key in grid_location_dict.keys():
            new_grid_DF = pd.read_excel(grid_location_dict[key])
            grid_DF = pd.concat([grid_DF, new_grid_DF])

        grid_DF.set_index('file', inplace=True)

        for file_idx, file in enumerate(tqdm(data_files)):
            with open(os.path.join(data_location, file), 'rb') as filehandle:
                inference_data = pickle.load(filehandle)

            labels, targets, scores, patch_scores, slide_names, features = inference_data
            num_slides, max_tile_num = features.shape[0], features.shape[2]

            for slide_num in range(num_slides):
                # Skip slides that were extracted from ABCTB_TIF and have a "bad segmentation" marker in GridData.xlsx file
                try:
                    if grid_DF.loc[slide_names[slide_num], 'bad segmentation'] == 1:
                        slides_with_bad_segmentation += 1
                        continue
                except KeyError:
                    print()

                total_slides += 1
                feature_1 = features[slide_num, :, :, 0]
                nan_indices = np.argwhere(np.isnan(feature_1)).tolist()
                tiles_in_slide = nan_indices[0][1] if bool(nan_indices) else max_tile_num  # check if there are any nan values in feature_1
                tiles_in_slide_from_grid_data = grid_DF.loc[slide_names[slide_num], 'Legitimate tiles - 256 compatible @ X10']

                if tiles_in_slide_from_grid_data < tiles_in_slide:  # Checking that the number of tiles in Grid_data.xlsx is equall to the one found in the actual data
                    bad_num_of_good_tiles += 1
                    tiles_in_slide = tiles_in_slide_from_grid_data

                if data_limit is not None and is_train and tiles_in_slide > data_limit:  # Limit the number of feature tiles according to argument "data_limit
                    tiles_in_slide = data_limit

                if tiles_in_slide < minimum_tiles_in_slide:  # Checking that the slide has a minimum number of tiles to be useable
                    slides_with_not_enough_tiles += 1
                    continue

                if is_per_patient:
                    patient = slide_names[slide_num].split('.')[0]
                    if patient.split('-')[0] == 'TCGA':
                        patient = '-'.join(patient.split('-')[:3])
                    patient_list.append(patient)

                    if patient in self.patient_data.keys():
                        patient_dict = self.patient_data[patient]
                        patient_same_target = True if int(targets[slide_num]) == patient_dict['target'] else False  # Checking the the patient target is not changing between slides
                        if patient in self.bad_patient_list:
                            slides_from_same_patient_with_different_target_values += 1
                            continue
                        if not patient_same_target:
                            slides_from_same_patient_with_different_target_values = 1 + len(self.patient_data[patient]['slides'])  # we skip mofre than 1 slide since we need to count the current slide and the ones that are already inserted to the patient_dicr
                            self.patient_data.pop(patient)
                            self.bad_patient_list.append(patient)
                            continue

                        patient_dict['num tiles'].append(tiles_in_slide)
                        patient_dict['tile scores'] = np.concatenate((patient_dict['tile scores'], patch_scores[slide_num, :tiles_in_slide]), axis=0)
                        patient_dict['labels'].append(int(labels[slide_num]))
                        patient_dict['target'].append(int(targets[slide_num]))
                        patient_dict['slides'].append(slide_names[slide_num])
                        patient_dict['scores'].append(scores[slide_num])

                        # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                        if self.fixed_tile_num is not None:
                            tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                        features_old = patient_dict['features']
                        features_new = features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32)
                        patient_dict['features'] = np.concatenate((features_old, features_new), axis=0)

                    else:
                        patient_dict = {'num tiles': [tiles_in_slide],
                                        'features': features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32),
                                        'tile scores': patch_scores[slide_num, :tiles_in_slide],
                                        'labels': [int(labels[slide_num])],
                                        'target': int(targets[slide_num]),
                                        'slides': [slide_names[slide_num]],
                                        'scores': [scores[slide_num]]
                                        }

                        self.patient_data[patient] = patient_dict

                else:
                    # Now we decide how many feature tiles will be taken w.r.t self.fixed_tile_num parameter
                    if self.fixed_tile_num is not None:
                        tiles_in_slide = tiles_in_slide if tiles_in_slide <= self.fixed_tile_num else self.fixed_tile_num

                    self.num_tiles.append(tiles_in_slide)
                    self.features.append(features[slide_num, :, :tiles_in_slide, :].squeeze(0).astype(np.float32))
                    self.tile_scores.append(patch_scores[slide_num, :tiles_in_slide])
                    self.slide_names.append(slide_names[slide_num])
                    self.labels.append(int(labels[slide_num]))
                    self.targets.append(int(targets[slide_num]))
                    self.scores.append(scores[slide_num])

        print('There are {}/{} slides with \"bad number of good tile\" '.format(bad_num_of_good_tiles, total_slides))
        print('There are {}/{} slides with \"bad segmentation\" '.format(slides_with_bad_segmentation, total_slides))
        print('There are {}/{} slides with less than {} tiles '.format(slides_with_not_enough_tiles, total_slides, minimum_tiles_in_slide))
        if is_per_patient:
            self.patient_keys = list(self.patient_data.keys())
            print('Skipped {}/{} slides for {}/{} patients (Inconsistent target value for same patient)'.format(slides_from_same_patient_with_different_target_values, total_slides, len(self.bad_patient_list), len(set(patient_list))))
        if self.is_per_patient:
            print('Initialized Dataset with {} feature slides in {} patients'.format(total_slides - slides_from_same_patient_with_different_target_values - slides_with_not_enough_tiles,
                                                                                     self.__len__()))
        else:
            print('Initialized Dataset with {} feature slides'.format(self.__len__()))

    def __len__(self):
        if self.is_per_patient:
            return len(self.patient_keys)
        else:
            return len(self.slide_names)

    def __getitem__(self, item):
        if self.is_per_patient:
            patient_data = self.patient_data[self.patient_keys[item]]
            num_tiles = int(np.array(patient_data['num tiles']).sum())

            if self.is_repeating_tiles:
                tile_idx = list(range(num_tiles)) if self.is_all_tiles else choices(range(num_tiles), k=self.bag_size)
            else:
                tile_idx = list(range(num_tiles)) if self.is_all_tiles else sample(range(num_tiles), k=self.bag_size)

            return {'labels': np.array(patient_data['labels']).mean(),
                    'targets': patient_data['target'],
                    'scores': np.array(patient_data['scores']).mean(),
                    'tile scores': patient_data['tile scores'][tile_idx],
                    'features': patient_data['features'][tile_idx],
                    'num tiles': num_tiles
                    }

        else:
            tile_idx = list(range(self.num_tiles[item])) if self.is_all_tiles else choices(range(self.num_tiles[item]), k=self.bag_size)

            return {'labels': self.labels[item],
                    'targets': self.targets[item],
                    'scores': self.scores[item],
                    'tile scores': self.tile_scores[item][tile_idx],
                    'slide name': self.slide_names[item],
                    'features': self.features[item][tile_idx],
                    'num tiles': self.num_tiles[item]
                    }


class One_Full_Slide_Inference_Dataset(WSI_Master_Dataset):
    """
    This Class provides tile extraction for ONE specific slide WITHOUT using the legitimate tile grid
    """
    def __init__(self,
                 DataSet: str = 'TCGA',
                 slidename: str = '',
                 target_kind: str = 'ER',
                 folds: List = [1],
                 tile_size: int = 256,
                 desired_slide_magnification: int = 10
                 ):
        super(One_Full_Slide_Inference_Dataset, self).__init__(DataSet=DataSet,
                                                               tile_size=256,
                                                               bag_size=None,
                                                               target_kind=target_kind,
                                                               test_fold=1,
                                                               infer_folds=folds,
                                                               train=True,
                                                               print_timing=False,
                                                               transform_type='none',
                                                               get_images=False,
                                                               train_type='Infer',
                                                               desired_slide_magnification=desired_slide_magnification)

        self.tile_size = tile_size

        slide_idx = np.where(np.array(self.all_image_file_names) == slidename)[0][0]

        height = int(self.meta_data_DF.loc[slide_idx, 'Height'])
        width = int(self.meta_data_DF.loc[slide_idx, 'Width'])
        objective_power = self.meta_data_DF.loc[slide_idx, 'Manipulated Objective Power']

        adjusted_tile_size_at_level_0 = int(self.tile_size * (int(objective_power) / self.desired_magnification))
        equivalent_rows = int(np.ceil(height / adjusted_tile_size_at_level_0))
        equivalent_cols = int(np.ceil(width / adjusted_tile_size_at_level_0))

        self.delta_pixel = int(objective_power / self.desired_magnification)

        self.equivalent_grid_size = (equivalent_rows, equivalent_cols)
        self.magnification = self.all_magnifications[slide_idx]
        self.slide_name = self.all_image_file_names[slide_idx]
        self.slide = self.slides[slide_idx]

    def __len__(self):
        return 1

    def __getitem__(self, location: List = None):
        desired_downsample = self.magnification / self.desired_magnification

        level, best_next_level = -1, -1
        for index, downsample in enumerate(self.slide.level_downsamples):
            if isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = index
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                best_next_level = index
                level_downsample = int(
                    desired_downsample / self.slide.level_downsamples[best_next_level])

        self.adjusted_tile_size = self.tile_size * level_downsample

        self.best_slide_level = level if level > best_next_level else best_next_level
        self.level_0_tile_size = int(desired_downsample) * self.tile_size

        tiles, time_list, _ = _get_tiles(slide=self.slide,
                                         locations=location,
                                         tile_size_level_0=self.level_0_tile_size,
                                         adjusted_tile_sz=self.adjusted_tile_size,
                                         output_tile_sz=self.tile_size,
                                         best_slide_level=self.best_slide_level)


        # Converting the tiles to Tensor:
        X, original_data = [], []
        for tile in tiles:
            original_data.append(transforms.ToTensor()(tile))
            X.append(torch.reshape(self.transform(tile), (1, self.transform(tile).size(0), self.transform(tile).size(1), self.transform(tile).size(2))))

        return {'Data': X,
                'Slide Filename': self.slide._filename,
                'Equivalent Grid Size': self.equivalent_grid_size,
                'Original Data': original_data
                }
