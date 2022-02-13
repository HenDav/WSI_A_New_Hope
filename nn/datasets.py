# python core
import os
import pickle
import random
import time
import pathlib
import math
import queue
from multiprocessing import Process, Queue, SimpleQueue, cpu_count

# tqdm
from tqdm import tqdm

# data processing
import numpy as np
import pandas as pd

# pytorch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# openslide
import openslide

# PIL
from PIL import Image

# opencv
import cv2

# matplotlib
from matplotlib import pyplot as plt

# sklearn
from skimage.draw import line

# wsi
import utils


class WSIOnlineDataset(Dataset):
    def __init__(self, dataset_size, buffer_size, max_size, replace, num_workers):
        self._dataset_size = dataset_size
        self._replace = replace
        self._num_workers = num_workers
        self._buffer_size = buffer_size
        self._q = Queue(maxsize=max_size)
        self._args = [self._q]
        self._items = []

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, index):
        # item = {}
        mod_index = np.mod(index, self._buffer_size)
        tuplet = self._items[mod_index]
        # for key in tuplet.keys():
        #     item[key] = tuplet[key]

        if self._replace is True:
            try:
                new_tuplet = self._q.get_nowait()
                rand_index = int(np.random.randint(self._buffer_size, size=1))
                self._items[rand_index] = new_tuplet
            except queue.Empty:
                pass

        return tuplet

    def start(self):
        self._workers = [Process(target=self._map_func, args=self._args) for i in range(self._num_workers)]

        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {self._num_workers}', end='')

        print(f'\nItem {len(self._items)} / {self._buffer_size}', end='')
        while True:
            if self._q.empty() is False:
                self._items.append(self._q.get())
                print(f'\rItem {len(self._items)} / {self._buffer_size}', end='')
                if len(self._items) == self._buffer_size:
                    break
        print('\n')

    def stop(self):
        for i, worker in enumerate(self._workers):
            worker.terminate()

        self._q.close()


class WSIDistanceDataset(WSIOnlineDataset):
    @staticmethod
    def get_dir_dict(dataset_name, local_path_prefix=None):
        dir_dict = {}
        gipdeep_path_prefix = r'/mnt/gipmed_new/Data'
        path_suffixes = {
            'TCGA': 'Breast/TCGA',
            'ABCTB': 'Breast/ABCTB/ABCTB',
            'HEROHE': 'Breast/HEROHE',
            'SHEBA': 'Breast/Sheba/SHEBA',
            'Ipatimup': 'Breast/Ipatimup',
            'Carmel': 'Breast/Carmel',
            'Covilha': 'Breast/Covilha',
            'ABCTB_TIF': 'ABCTB_TIF',
            'TCGA_LUNG': 'Lung/TCGA_Lung/TCGA_LUNG',
            'LEUKEMIA': 'BoneMarrow/LEUKEMIA',
        }

        for k in path_suffixes.keys():
            if k == dataset_name:
                path_suffix = path_suffixes[k]
                if local_path_prefix is None:
                    dir_dict[k] = os.path.normpath(os.path.join(gipdeep_path_prefix, path_suffix))
                else:
                    dir_dict[k] = os.path.normpath(os.path.join(local_path_prefix, path_suffix))

        return dir_dict

    @staticmethod
    def load_datasets_metadata(datasets_dir_dict, desired_magnification):
        metadata_df = None
        for _, key in enumerate(datasets_dir_dict):
            slide_metadata_file = os.path.join(datasets_dir_dict[key], 'slides_data_' + key + '.xlsx')
            grid_metadata_file = os.path.join(datasets_dir_dict[key], 'Grids_' + str(desired_magnification), 'Grid_data.xlsx')
            slide_metadata_df = pd.read_excel(slide_metadata_file)
            grid_metadata_df = pd.read_excel(grid_metadata_file)
            current_metadata_df = pd.DataFrame({**slide_metadata_df.set_index('file').to_dict(), **grid_metadata_df.set_index('file').to_dict()})
            if metadata_df is None:
                metadata_df = current_metadata_df
            else:
                metadata_df.append(metadata_df)

        metadata_df.reset_index(inplace=True)
        metadata_df.rename(columns={'index': 'file'}, inplace=True)
        return metadata_df

    @staticmethod
    def get_valid_indices(metadata_df, tile_size, desired_magnification, minimal_tiles_count):
        slides_without_grid = set(metadata_df.index[metadata_df[f'Total tiles - {tile_size} compatible @ X{desired_magnification}'] == -1])
        slides_with_0_tiles = set(metadata_df.index[metadata_df[f'Legitimate tiles - {tile_size} compatible @ X{desired_magnification}'] == 0])

        if 'bad segmentation' in metadata_df.columns:
            slides_with_bad_seg = set(metadata_df.index[metadata_df['bad segmentation'] == 1])
        else:
            slides_with_bad_seg = set()

        slides_with_few_tiles = set(metadata_df.index[metadata_df[f'Legitimate tiles - {tile_size} compatible @ X{desired_magnification}'] < minimal_tiles_count])

        valid_slide_indices = np.array(range(metadata_df.shape[0]))
        valid_slide_indices = np.array(list(set(valid_slide_indices) - slides_without_grid - slides_with_few_tiles - slides_with_0_tiles - slides_with_bad_seg))
        return valid_slide_indices

    @staticmethod
    def get_fold_column_name(dataset_name):
        if dataset_name == 'CAT' or dataset_name == 'ABCTB_TCGA':
            fold_column_name = 'test fold idx breast'
        else:
            fold_column_name = 'test fold idx'

        return fold_column_name

    @staticmethod
    def get_indices_pool(metadata_df, fold_column_name, valid_indices, train, test_fold):
        if train:
            folds = list(metadata_df[fold_column_name].unique())
            folds.remove(test_fold)
        else:
            folds = [test_fold]

        correct_folds = metadata_df[fold_column_name][valid_indices].isin(folds)
        indices_pool = np.array(correct_folds.index[correct_folds])
        return indices_pool

    # @staticmethod
    # def get_optimal_slide_level(slide, magnification, desired_mag, tile_size):
    #     desired_downsample = magnification / desired_mag  # downsample needed for each dimension (reflected by level_downsamples property)
    #
    #     if desired_downsample < 1:  # upsample
    #         best_slide_level = 0
    #         tile_size_level_0 = int(desired_downsample * tile_size)
    #         adjusted_tile_size = tile_size_level_0
    #     else:
    #         level, best_next_level = -1, -1
    #         for index, downsample in enumerate(slide.level_downsamples):
    #             if math.isclose(desired_downsample, downsample, rel_tol=1e-3):
    #                 level = index
    #                 level_downsample = 1
    #                 break
    #
    #             elif downsample < desired_downsample:
    #                 best_next_level = index
    #                 level_downsample = int(desired_downsample / slide.level_downsamples[best_next_level])
    #
    #         adjusted_tile_size = tile_size * level_downsample
    #         best_slide_level = level if level > best_next_level else best_next_level
    #         tile_size_level_0 = int(desired_downsample) * tile_size
    #
    #     return best_slide_level, adjusted_tile_size, tile_size_level_0

    @staticmethod
    def get_best_level_for_downsample(slide, desired_downsample, tile_size):
        for i, downsample in enumerate(slide.level_downsamples):
            if math.isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = i
                level_downsample = 1
                break

            elif downsample < desired_downsample:
                level = i
                level_downsample = int(desired_downsample / slide.level_downsamples[level])

        adjusted_tile_size = tile_size * level_downsample

        return level, adjusted_tile_size

    @staticmethod
    def get_data(locations,
                 slide,
                 magnification,
                 tile_size,
                 desired_magnification):
        desired_downsample = magnification / desired_magnification
        level = slide.get_best_level_for_downsample(downsample=desired_downsample)
        location_index = np.random.randint(len(locations))
        location = locations[0]
        tile_size = 1024
        tile = slide.read_region((location[1], location[0]), level, (tile_size, tile_size)).convert('RGB')
        return tile

    @staticmethod
    def calculate_bitmap_indices(point, tile_size):
        bitmap_indices = (point / tile_size).astype(int)
        return bitmap_indices

    @staticmethod
    def validate_location(bitmap, indices):
        for i in range(3):
            for j in range(3):
                current_indices = tuple(indices + np.array([i, j]))
                if (not (0 <= current_indices[0] < bitmap.shape[0])) or (not (0 <= current_indices[1] < bitmap.shape[1])):
                    return False

                bit = bitmap[current_indices]
                if bit == 0:
                    return False
        return True

    @staticmethod
    def read_region_around_point(slide, point, tile_size, adjusted_tile_size, level):
        top_left_point = (point - adjusted_tile_size / 2).astype(int)
        tile = slide.read_region(top_left_point, level, (adjusted_tile_size, adjusted_tile_size)).convert('RGB')
        if adjusted_tile_size != tile_size:
            tile = tile.resize((tile_size, tile_size))
        return tile

    @staticmethod
    def create_tuple(slide_descriptor,
                     tile_size,
                     inner_radius,
                     outer_radius,
                     negative_examples_count,
                     transform):
        input = []
        image_file_path = slide_descriptor['image_file_path']
        desired_downsample = slide_descriptor['desired_downsample']
        original_tile_size = slide_descriptor['original_tile_size']
        image_file_suffix = slide_descriptor['image_file_suffix']
        components = slide_descriptor['components']

        slide = openslide.open_slide(image_file_path)
        level, adjusted_tile_size = WSIDistanceDataset.get_best_level_for_downsample(slide=slide, desired_downsample=desired_downsample, tile_size=tile_size)

        # level = slide.get_best_level_for_downsample(downsample=desired_downsample)
        mm_to_pixel = utils.get_mm_to_pixel(downsample=desired_downsample, image_file_suffix=image_file_suffix)
        inner_radius_pixels = inner_radius * mm_to_pixel
        min_outer_radius_pixels = outer_radius * mm_to_pixel
        max_outer_radius_pixels = 2 * min_outer_radius_pixels

        main_component = components[0]
        tile_indices = main_component['tile_indices']
        tiles_bitmap = main_component['bitmap']

        attempts = 0
        while True:
            if attempts == 10:
                break

            anchor_index = np.random.randint(tile_indices.shape[0])
            anchor_tile_indices = tile_indices[anchor_index, :]
            anchor_location = anchor_tile_indices * original_tile_size
            anchor_point_offset = (original_tile_size * np.random.uniform(size=2)).astype(int)
            anchor_point = anchor_point_offset + anchor_location
            anchor_bitmap_indices = WSIDistanceDataset.calculate_bitmap_indices(point=anchor_point, tile_size=original_tile_size)
            if WSIDistanceDataset.validate_location(bitmap=tiles_bitmap, indices=anchor_bitmap_indices) is False:
                attempts = attempts + 1
                continue

            anchor_tile = WSIDistanceDataset.read_region_around_point(slide=slide, point=anchor_point, tile_size=tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
            anchor_tile_grayscale = anchor_tile.convert('L')
            hist, _ = np.histogram(anchor_tile_grayscale, bins=tile_size)
            white_ratio = np.sum(hist[170:]) / (tile_size*tile_size)
            if white_ratio > 0.3:
                attempts = attempts + 1
                continue

            # anchor_tile = transform(WSIDistanceDataset.read_region_around_point(slide=slide, point=anchor_point, tile_size=adjusted_tile_size, level=level))

            input.append(np.array(anchor_tile))
            # input.append(transforms.ToTensor()(anchor_tile))


            #
            # # plt.imshow(anchor_tile_grayscale, cmap=plt.get_cmap('gray'))
            # # plt.show()
            break

        if len(input) == 0:
            return None

        attempts = 0
        while True:
            if attempts == 10:
                break

            positive_angle = 2 * np.pi * np.random.uniform(size=1)[0]
            positive_dir = np.array([np.cos(positive_angle), np.sin(positive_angle)])
            positive_radius = inner_radius_pixels * np.random.uniform(size=1)[0]
            positive_point = (anchor_point + positive_radius * positive_dir).astype(int)
            positive_bitmap_indices = WSIDistanceDataset.calculate_bitmap_indices(point=positive_point, tile_size=original_tile_size)
            if WSIDistanceDataset.validate_location(bitmap=tiles_bitmap, indices=positive_bitmap_indices) is False:
                attempts = attempts + 1
                continue

            # positive_tile = transform(WSIDistanceDataset.read_region_around_point(slide=slide, point=positive_point, tile_size=adjusted_tile_size, level=level))
            positive_tile = WSIDistanceDataset.read_region_around_point(slide=slide, point=positive_point, tile_size=tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
            input.append(np.array(positive_tile))
            # input.append(transforms.ToTensor()(positive_tile))
            break

        if len(input) == 1:
            return None

        negative_tiles = []
        attempts = 0
        for i in range(negative_examples_count):
            while True:
                if attempts == 10:
                    break

                negative_angle = 2 * np.pi * np.random.uniform(size=1)[0]
                negative_dir = np.array([np.cos(negative_angle), np.sin(negative_angle)])
                negative_radius = (max_outer_radius_pixels - min_outer_radius_pixels) * np.random.uniform(size=1)[0] + min_outer_radius_pixels
                negative_point1 = (anchor_point + min_outer_radius_pixels * negative_dir).astype(int)
                negative_point2 = (anchor_point + negative_radius * negative_dir).astype(int)
                negative_bitmap_indices1 = WSIDistanceDataset.calculate_bitmap_indices(point=negative_point1, tile_size=original_tile_size)
                negative_bitmap_indices2 = WSIDistanceDataset.calculate_bitmap_indices(point=negative_point2, tile_size=original_tile_size)

                if negative_bitmap_indices1[0] < 0 or negative_bitmap_indices1[1] < 0 or negative_bitmap_indices2[0] < 0 or negative_bitmap_indices2[1] < 0:
                    continue

                xx, yy = line(negative_bitmap_indices1[0], negative_bitmap_indices1[1], negative_bitmap_indices2[0], negative_bitmap_indices2[1])
                found = False
                for indices in zip(np.flip(xx), np.flip(yy)):
                    if WSIDistanceDataset.validate_location(bitmap=tiles_bitmap, indices=indices) is True:
                        found = True
                        break

                if found is False:
                    attempts = attempts + 1
                    continue

                negative_point_offset = (original_tile_size * np.random.uniform(size=2)).astype(int)
                negative_location = np.array(indices) * original_tile_size
                negative_point = negative_location + negative_point_offset
                # negative_tile = transform(WSIDistanceDataset.read_region_around_point(slide=slide, point=negative_point, tile_size=adjusted_tile_size, level=level))
                negative_tile = WSIDistanceDataset.read_region_around_point(slide=slide, point=negative_point, tile_size=tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
                negative_tiles.append(negative_tile)
                input.append(np.array(negative_tile))
                # input.append(transforms.ToTensor()(negative_tile))
                break

        if len(negative_tiles) < negative_examples_count:
            return None

        # return {
        #     'anchor_tile': anchor_tile,
        #     'positive_tile': positive_tile,
        #     'negative_tiles': negative_tiles
        # }

        # bla = transforms.ToTensor()(np.stack(input))

        # input_features = np.transpose(np.stack(input).astype(np.float32) / 255, (0, 3, 1, 2))
        input_features = np.transpose(np.stack(input), (0, 3, 1, 2))

        return {
            # 'input_features': np.stack(input)
            'input_features': input_features
        }

    @classmethod
    def _map_func(cls, q, slide_descriptors, tile_size, inner_radius, outer_radius, transform):
        slide_descriptors_count = len(slide_descriptors)
        while True:
            slide_descriptor_index = np.random.randint(slide_descriptors_count)
            slide_descriptor = slide_descriptors[slide_descriptor_index]
            tuple = WSIDistanceDataset.create_tuple(slide_descriptor=slide_descriptor,
                                                    tile_size=tile_size,
                                                    inner_radius=inner_radius,
                                                    outer_radius=outer_radius,
                                                    negative_examples_count=2,
                                                    transform=transform)
            if tuple is None:
                continue

            q.put(tuple)

    def __init__(
            self,
            dataset_size,
            buffer_size,
            max_size,
            replace,
            num_workers,
            dataset_name,
            tile_size,
            desired_magnification,
            minimal_tiles_count,
            test_fold,
            train,
            datasets_base_dir_path,
            inner_radius,
            outer_radius):

        WSIOnlineDataset.__init__(
            self,
            dataset_size=dataset_size,
            buffer_size=buffer_size,
            max_size=max_size,
            replace=replace,
            num_workers=num_workers)

        self._counter = 0
        self._dataset_name = dataset_name
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._minimal_tiles_count = minimal_tiles_count
        self._test_fold = test_fold
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._train = train
        self._local_path_prefix = datasets_base_dir_path
        self._slide_descriptors = []

        self._transform = torch.nn.Sequential(
            # transforms.ToPILImage(),
            transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=0.1, hue=(-0.1, 0.1)),
            transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(tile_size),
            # transforms.ToTensor()
        )

        self._args.append(self._slide_descriptors)
        self._args.append(self._tile_size)
        self._args.append(self._inner_radius)
        self._args.append(self._outer_radius)
        self._args.append(self._transform)

        self.dir_dict = WSIDistanceDataset.get_dir_dict(
            dataset_name=self._dataset_name,
            local_path_prefix=self._local_path_prefix)

        self.metadata_df = WSIDistanceDataset.load_datasets_metadata(
            datasets_dir_dict=self.dir_dict,
            desired_magnification=self._desired_magnification)

        self.valid_indices = WSIDistanceDataset.get_valid_indices(
            metadata_df=self.metadata_df,
            tile_size=self._tile_size,
            desired_magnification=self._desired_magnification,
            minimal_tiles_count=self._minimal_tiles_count)

        self.fold_column_name = WSIDistanceDataset.get_fold_column_name(
            dataset_name=self._dataset_name)

        self.indices_pool = WSIDistanceDataset.get_indices_pool(
            metadata_df=self.metadata_df,
            fold_column_name=self.fold_column_name,
            valid_indices=self.valid_indices,
            train=self._train,
            test_fold=self._test_fold)

        image_file_names = list(self.metadata_df['file'])
        image_ids = list(self.metadata_df['id'])
        image_folds = list(self.metadata_df[self.fold_column_name])
        image_compatible_tiles_counts = list(self.metadata_df[f'Legitimate tiles - {self._tile_size} compatible @ X{self._desired_magnification}'])
        image_magnifications = list(self.metadata_df['Manipulated Objective Power'])

        for _, index in enumerate(tqdm(self.indices_pool)):
            try:
                image_file_path = os.path.join(self.dir_dict[image_ids[index]], image_file_names[index])
                image_file_name = pathlib.Path(image_file_path).stem
                image_file_suffix = pathlib.Path(image_file_path).suffix
                magnification = image_magnifications[index]
                compatible_tiles_count = image_compatible_tiles_counts[index]
                image_fold = image_folds[index]
                # slide = openslide.open_slide(image_file_path)
                grid_file_path = os.path.normpath(os.path.join(self.dir_dict[image_ids[index]], f'Grids_{self._desired_magnification}', f'{image_file_name}--tlsz{self._tile_size}.data'))
                with open(grid_file_path, 'rb') as file_handle:
                    locations = np.array(pickle.load(file_handle))
                    locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()

                desired_downsample = magnification / self._desired_magnification
                original_tile_size = self._tile_size * desired_downsample
                indices = (np.array(locations) / original_tile_size).astype(int)
                dim1_size = indices[:, 0].max() + 1
                dim2_size = indices[:, 1].max() + 1
                bitmap = np.zeros([dim1_size, dim2_size]).astype(int)

                for (x, y) in indices:
                    bitmap[x, y] = 1

                bitmap_image = Image.fromarray(bitmap)
                bitmap_image_grayscale = np.uint8(bitmap_image)

                # plt.imshow(bitmap_image_grayscale, cmap='gray')
                # plt.show()

                components_count, components_labels, components_stats, _ = cv2.connectedComponentsWithStats(bitmap_image_grayscale)
                components = []

                if components_count == 1:
                    continue

                for component_id in range(1, components_count):
                    current_bitmap = (components_labels == component_id)
                    component_indices = np.where(current_bitmap)
                    component_size = np.count_nonzero(current_bitmap)
                    top_left = np.array([np.min(component_indices[0]), np.min(component_indices[1])])
                    bottom_right = np.array([np.max(component_indices[0]), np.max(component_indices[1])])
                    tile_indices = np.array([component_indices[0], component_indices[1]]).transpose()

                    components.append({
                        'bitmap': current_bitmap.astype(int),
                        'component_size': component_size,
                        'top_left': top_left,
                        'bottom_right': bottom_right,
                        'tile_indices': tile_indices
                    })

                components_sorted = sorted(components, key=lambda item: item['component_size'], reverse=True)
                largest_component = components_sorted[0]
                largest_component_aspect_ratio = utils.calculate_box_aspect_ratio(largest_component['top_left'], largest_component['bottom_right'])
                largest_component_size = largest_component['component_size']
                valid_components = [largest_component]
                for component in components_sorted[1:]:
                    current_aspect_ratio = utils.calculate_box_aspect_ratio(component['top_left'], component['bottom_right'])
                    current_component_size = component['component_size']
                    if np.abs(largest_component_aspect_ratio - current_aspect_ratio) < 0.02 and (current_component_size / largest_component_size) > 0.92:
                        valid_components.append(component)
                        # plt.imshow(bitmap_image_grayscale, cmap='gray')
                        # plt.show()

                self._slide_descriptors.append({
                    'image_file_path': image_file_path,
                    'image_file_name': image_file_name,
                    'image_file_suffix': image_file_suffix,
                    'magnification': magnification,
                    'compatible_tiles_count': compatible_tiles_count,
                    'fold': image_fold,
                    # 'slide': slide,
                    'locations': locations,
                    'indices': indices,
                    'bitmap': bitmap,
                    'desired_downsample': desired_downsample,
                    'original_tile_size': original_tile_size,
                    'components': valid_components
                })
            except Exception:
                continue


    # def __len__(self):
    #     return len(self._slide_descriptors) * 10

    # def __getitem__(self, idx):
    #     slide_descriptors_count = len(self._slide_descriptors)
    #     while True:
    #         slide_descriptor_index = np.random.randint(slide_descriptors_count)
    #         slide_descriptor = self._slide_descriptors[slide_descriptor_index]
    #         tuple = WSIDistanceDataset.create_tuple(slide_descriptor=slide_descriptor,
    #                                         tile_size=self._tile_size,
    #                                         inner_radius=self._inner_radius,
    #                                         outer_radius=self._outer_radius,
    #                                         negative_examples_count=2)
    #
    #         if tuple is not None:
    #             break
    #
    #     return tuple
