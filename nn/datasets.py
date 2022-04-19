# python core
import os
import pickle
import pathlib
import math
import io
import queue
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue

# tqdm
import numpy.random
import pandas

# data processing
import numpy as np
import pandas as pd

# pytorch
import torch
from torch.utils.data import Dataset

# openslide
import openslide

# PIL
from PIL import Image

# opencv
import cv2

# matplotlib
from matplotlib import pyplot as plt

# sklearn

# wsi
from utils import common_utils
from utils import slide_utils

class WSIOnlineDataset(Dataset):
    def __init__(self, dataset_size, buffer_size, max_size, replace, num_workers, train):
        self._dataset_size = dataset_size
        self._replace = replace
        self._num_workers = num_workers
        self._train = train
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

    def start(self, load_buffer=False, buffer_base_dir_path='./buffers'):
        self._workers = [Process(target=self._map_func, args=self._args) for i in range(self._num_workers)]
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {self._num_workers}', end='')

        buffer_type_dir_name = 'train' if self._train is True else 'validation'
        buffer_dir_path = os.path.normpath(os.path.join(buffer_base_dir_path, buffer_type_dir_name))
        buffer_file_name = 'train_buffer.pt' if self._train is True else 'validation_buffer.pt'
        if load_buffer is False:
            print(f'\nItem {len(self._items)} / {self._buffer_size}', end='')
            while True:
                if self._q.empty() is False:
                    print(f'Queue size = {self._q.qsize()}')
                    self._items.append(self._q.get())
                    print(f'\rItem {len(self._items)} / {self._buffer_size}', end='')
                    if len(self._items) == self._buffer_size:
                        buffers_dir_path = os.path.normpath(os.path.join(buffer_dir_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
                        buffer_file_path = os.path.join(buffers_dir_path, buffer_file_name)
                        Path(buffers_dir_path).mkdir(parents=True, exist_ok=True)
                        torch.save(self._items, buffer_file_path)
                        break
            print('\n')
        else:
            latest_dir_path = utils.get_latest_subdirectory(base_dir=buffer_dir_path)
            buffer_file_path = os.path.join(latest_dir_path, buffer_file_name)
            with open(buffer_file_path, 'rb') as f:
                buffer = io.BytesIO(f.read())
                self._items = torch.load(buffer)

    def stop(self):
        for i, worker in enumerate(self._workers):
            worker.terminate()

        self._q.close()


class WSITuplesGenerator:
    _file_column_name = 'file'
    _patient_barcode_column_name = 'patient barcode'
    _dataset_id_column_name = 'id'
    _mpp_column_name = 'mpp'
    _width_column_name = 'Width'
    _height_column_name = 'Height'
    _magnification_column_name = 'Manipulated Objective Power'
    _er_status_column_name = 'ER status'
    _pr_status_column_name = 'PR status'
    _her2_status_column_name = 'Her2 status'
    _invalid_fold_column_names = ['test fold idx breast', 'test fold idx', 'test fold idx breast - original for carmel']
    _fold_column_name = 'fold'
    _bad_segmentation_column_name = 'bad segmentation'
    _grids_data_prefix = 'slides_data_'
    _grid_data_file_name = 'Grid_data.xlsx'
    _invalid_values = ['Missing Data', 'Not performed', '[Not Evaluated]']
    _invalid_value = 'NA'
    _test_fold_id = 'test'
    _max_attempts = 10
    _white_ratio_threshold = 0.5
    _histogram_min_intensity_level = 170

    @staticmethod
    def get_total_tiles_column_name(tile_size, desired_magnification):
        return f'Total tiles - {tile_size} compatible @ X{desired_magnification}'

    @staticmethod
    def get_legitimate_tiles_column_name(tile_size, desired_magnification):
        return f'Legitimate tiles - {tile_size} compatible @ X{desired_magnification}'

    @staticmethod
    def get_slide_tile_usage_column_name(tile_size, desired_magnification):
        return f'Slide tile usage [%] (for {tile_size}^2 Pix/Tile) @ X{desired_magnification}'

    @staticmethod
    def get_slides_data_file_name(dataset_id):
        return f'slides_data_{dataset_id}.xlsx'

    @staticmethod
    def get_grids_folder_name(desired_magnification):
        return f'Grids_{str(desired_magnification)}'

    @staticmethod
    def _get_dataset_paths(dataset_ids, datasets_base_dir_path=None):
        dir_dict = {}
        gipdeep_path_prefix = r'/mnt/gipmed_new/Data'
        path_suffixes = {
            'TCGA': 'Breast/TCGA',
            'ABCTB': 'Breast/ABCTB/ABCTB',
            'HEROHE': 'Breast/HEROHE',
            'SHEBA': 'Breast/Sheba/SHEBA',
            'Ipatimup': 'Breast/Ipatimup',
            # 'Carmel': 'Breast/Carmel',
            'Covilha': 'Breast/Covilha',
            'ABCTB_TIF': 'ABCTB_TIF',
            'TCGA_LUNG': 'Lung/TCGA_Lung/TCGA_LUNG',
            'LEUKEMIA': 'BoneMarrow/LEUKEMIA',
        }

        for i in range(1, 12):
            path_suffixes[f'Carmel{i}'] = f'Breast/Batch_{i}/CARMEL{i}'

        for k in path_suffixes.keys():
            if k in dataset_ids:
                path_suffix = path_suffixes[k]
                if datasets_base_dir_path is None:
                    dir_dict[k] = os.path.normpath(os.path.join(gipdeep_path_prefix, path_suffix))
                else:
                    dir_dict[k] = os.path.normpath(os.path.join(datasets_base_dir_path, path_suffix))

        return dir_dict

    @staticmethod
    def _load_datasets_metadata(dataset_paths, desired_magnification):
        metadata_df = None
        for _, dataset_id in enumerate(dataset_paths):
            slide_metadata_file = os.path.join(dataset_paths[dataset_id], WSITuplesGenerator.get_slides_data_file_name(dataset_id=dataset_id))
            grid_metadata_file = os.path.join(dataset_paths[dataset_id], WSITuplesGenerator.get_grids_folder_name(desired_magnification=desired_magnification), WSITuplesGenerator._grid_data_file_name)
            slide_metadata_df = pd.read_excel(io=slide_metadata_file)
            grid_metadata_df = pd.read_excel(io=grid_metadata_file)
            current_metadata_df = pd.DataFrame({**slide_metadata_df.set_index(keys=WSITuplesGenerator._file_column_name).to_dict(), **grid_metadata_df.set_index(keys=WSITuplesGenerator._file_column_name).to_dict()})
            if metadata_df is None:
                metadata_df = current_metadata_df
            else:
                metadata_df.append(metadata_df)

        metadata_df.reset_index(inplace=True)
        metadata_df.rename(columns={'index': WSITuplesGenerator._file_column_name}, inplace=True)
        return metadata_df

    @staticmethod
    def _standardize_datasets_metadata(metadata_df):
        metadata_df = metadata_df.drop(WSITuplesGenerator._invalid_fold_column_names, axis=1)
        metadata_df = metadata_df.replace(WSITuplesGenerator._invalid_values, WSITuplesGenerator._invalid_value)
        metadata_df = metadata_df.dropna()
        return metadata_df

    @staticmethod
    def _validate_metadata(metadata_df, tile_size, desired_magnification, minimal_tiles_count):
        total_tiles_column_name = WSITuplesGenerator.get_total_tiles_column_name(tile_size=tile_size, desired_magnification=desired_magnification)
        legitimate_tiles_column_name = WSITuplesGenerator.get_legitimate_tiles_column_name(tile_size=tile_size, desired_magnification=desired_magnification)

        indices_of_slides_without_grid = set(metadata_df.index[metadata_df[total_tiles_column_name] == -1])
        indices_of_slides_with_0_tiles = set(metadata_df.index[metadata_df[legitimate_tiles_column_name] == 0])

        if WSITuplesGenerator._bad_segmentation_column_name in metadata_df.columns:
            indices_of_slides_with_bad_seg = set(metadata_df.index[metadata_df[WSITuplesGenerator._bad_segmentation_column_name] == 1])
        else:
            indices_of_slides_with_bad_seg = set()

        indices_of_slides_with_few_tiles = set(metadata_df.index[metadata_df[legitimate_tiles_column_name] < minimal_tiles_count])

        all_indices = set(np.array(range(metadata_df.shape[0])))
        valid_slide_indices = np.array(list(all_indices - indices_of_slides_without_grid - indices_of_slides_with_few_tiles - indices_of_slides_with_0_tiles - indices_of_slides_with_bad_seg))
        return metadata_df.iloc[valid_slide_indices]

    @staticmethod
    def _add_folds_to_metadata(metadata_df, folds_count):
        folds = np.random.randint(folds_count, size=metadata_df.shape[0])
        metadata_df[WSITuplesGenerator._fold_column_name] = folds
        return metadata_df

    @staticmethod
    def _select_folds_from_metadata(metadata_df, folds_count, test_fold, train):
        if train is True:
            folds = list(range(folds_count))
            folds.remove(test_fold)
        else:
            folds = [test_fold]

        return metadata_df[metadata_df[WSITuplesGenerator._fold_column_name].isin(folds)]

    @staticmethod
    def _calculate_bitmap_indices(point, tile_size):
        bitmap_indices = (point / tile_size).astype(int)
        return bitmap_indices

    @staticmethod
    def _validate_location(bitmap, indices):
        for i in range(3):
            for j in range(3):
                current_indices = tuple(indices + np.array([i, j]))
                if (not (0 <= current_indices[0] < bitmap.shape[0])) or (not (0 <= current_indices[1] < bitmap.shape[1])):
                    return False

                bit = bitmap[current_indices]
                if bit == 0:
                    return False
        return True

    def _open_slide(self, image_file_path, desired_downsample):
        slide = openslide.open_slide(image_file_path)
        level, adjusted_tile_size = slide_utils.get_best_level_for_downsample(slide=slide, desired_downsample=desired_downsample, tile_size=self._tile_size)

        return {
            'slide': slide,
            'level': level,
            'adjusted_tile_size': adjusted_tile_size
        }

    def _calculate_inner_radius_pixels(self, desired_downsample, image_file_name_suffix):
        mm_to_pixel = slide_utils.get_mm_to_pixel(downsample=desired_downsample, image_file_suffix=image_file_name_suffix)
        inner_radius_pixels = self._inner_radius * mm_to_pixel
        return inner_radius_pixels

    def _create_random_example(self, slide_descriptor, component_index):
        image_file_path = slide_descriptor['image_file_path']
        desired_downsample = slide_descriptor['desired_downsample']
        original_tile_size = slide_descriptor['original_tile_size']
        components = slide_descriptor['components']
        slide_data = self._open_slide(image_file_path=image_file_path, desired_downsample=desired_downsample)
        component = components[component_index]

        tile_indices = component['tile_indices']
        tiles_bitmap = component['bitmap']
        slide = slide_data['slide']
        adjusted_tile_size = slide_data['adjusted_tile_size']
        level = slide_data['level']

        attempts = 0
        while True:
            if attempts == WSITuplesGenerator._max_attempts:
                break

            tile_index = np.random.randint(tile_indices.shape[0])
            tile_indices = tile_indices[tile_index, :]
            location = tile_indices * original_tile_size
            point_offset = (original_tile_size * np.random.uniform(size=2)).astype(int)
            point = point_offset + location
            bitmap_indices = WSITuplesGenerator._calculate_bitmap_indices(point=point, tile_size=original_tile_size)
            if WSITuplesGenerator._validate_location(bitmap=tiles_bitmap, indices=bitmap_indices) is False:
                attempts = attempts + 1
                continue

            tile = slide_utils.read_region_around_point(slide=slide, point=point, tile_size=self._tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
            tile_grayscale = tile.convert('L')
            hist, _ = np.histogram(tile_grayscale, bins=self._tile_size)
            white_ratio = np.sum(hist[WSITuplesGenerator._histogram_min_intensity_level:]) / (self._tile_size * self._tile_size)
            if white_ratio > WSITuplesGenerator._white_ratio_threshold:
                attempts = attempts + 1
                continue

            return {
                'tile': np.array(tile),
                'point': point
            }

        return None

    def _create_anchor_example(self, slide_descriptor, component_index):
        return self._create_random_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index)

    def _create_positive_example(self, slide_descriptor, component_index, anchor_point):
        image_file_path = slide_descriptor['image_file_path']
        desired_downsample = slide_descriptor['desired_downsample']
        original_tile_size = slide_descriptor['original_tile_size']
        image_file_name_suffix = slide_descriptor['image_file_name_suffix']
        components = slide_descriptor['components']
        slide_data = self._open_slide(image_file_path=image_file_path, desired_downsample=desired_downsample)
        component = components[component_index]

        inner_radius_pixels = self._calculate_inner_radius_pixels(desired_downsample=desired_downsample, image_file_name_suffix=image_file_name_suffix)
        slide = slide_data['slide']
        adjusted_tile_size = slide_data['adjusted_tile_size']
        level = slide_data['level']
        tiles_bitmap = component['bitmap']

        attempts = 0
        while True:
            if attempts == WSITuplesGenerator._max_attempts:
                break

            positive_angle = 2 * np.pi * np.random.uniform(size=1)[0]
            positive_dir = np.array([np.cos(positive_angle), np.sin(positive_angle)])
            positive_radius = inner_radius_pixels * np.random.uniform(size=1)[0]
            positive_point = (anchor_point + positive_radius * positive_dir).astype(int)
            positive_bitmap_indices = WSITuplesGenerator._calculate_bitmap_indices(point=positive_point, tile_size=original_tile_size)
            if WSITuplesGenerator._validate_location(bitmap=tiles_bitmap, indices=positive_bitmap_indices) is False:
                attempts = attempts + 1
                continue

            positive_tile = slide_utils.read_region_around_point(
                slide=slide,
                point=positive_point,
                tile_size=self._tile_size,
                adjusted_tile_size=adjusted_tile_size,
                level=level)

            return np.array(positive_tile)

        return None

    def _create_negative_example(self, slide_descriptor):
        row_anchor_patient = self._metadata_df.loc[self._metadata_df[WSITuplesGenerator._file_column_name] == slide_descriptor['image_file_name']].iloc[0]
        patient_barcode = row_anchor_patient[WSITuplesGenerator._patient_barcode_column_name]
        er_status = row_anchor_patient[WSITuplesGenerator._er_status_column_name]
        pr_status = row_anchor_patient[WSITuplesGenerator._pr_status_column_name]
        her2_status = row_anchor_patient[WSITuplesGenerator._her2_status_column_name]
        df = self._metadata_df[(self._metadata_df[WSITuplesGenerator._patient_barcode_column_name] != patient_barcode) &
                               (self._metadata_df[WSITuplesGenerator._pr_status_column_name] != pr_status) &
                               (self._metadata_df[WSITuplesGenerator._er_status_column_name] != er_status) &
                               (self._metadata_df[WSITuplesGenerator._her2_status_column_name] != her2_status)]

        index = int(np.random.randint(df.shape[0], size=1))
        row_negative_patient = df.iloc[index]
        image_file_name_negative_patient = row_negative_patient[WSITuplesGenerator._file_column_name]
        slide_descriptor_negative_patient = self._image_file_name_to_slide_descriptor[image_file_name_negative_patient]
        component_index = WSITuplesGenerator._get_random_component_index(slide_descriptor=slide_descriptor_negative_patient)
        return self._create_random_example(slide_descriptor=slide_descriptor_negative_patient, component_index=component_index)


    # @classmethod
    # def _map_func(
    #         cls,
    #         q,
    #         slide_descriptors,
    #         tile_size,
    #         inner_radius,
    #         outer_radius,
    #         transform,
    #         patient_to_slide_descriptors,
    #         pr_positive_patient_to_slide_descriptors,
    #         pr_negative_patient_to_slide_descriptors,
    #         er_positive_patient_to_slide_descriptors,
    #         er_negative_patient_to_slide_descriptors):
    #
    #     slide_descriptors_count = len(slide_descriptors)
    #     while True:
    #         slide_descriptor_index = np.random.randint(slide_descriptors_count)
    #         slide_descriptor = slide_descriptors[slide_descriptor_index]
    #         patient_barcode = slide_descriptor['patient_barcode']
    #         tuple = WSITuplesGenerator.create_tuple(slide_descriptor=slide_descriptor,
    #                                                 tile_size=tile_size,
    #                                                 inner_radius=inner_radius,
    #                                                 outer_radius=outer_radius,
    #                                                 negative_examples_count=2,
    #                                                 transform=transform)
    #         if tuple is None:
    #             continue
    #
    #         q.put(tuple)

    @staticmethod
    def _create_metadata(
            dataset_paths,
            tile_size,
            desired_magnification,
            minimal_tiles_count,
            folds_count,
            test_fold,
            train):

        metadata_df = WSITuplesGenerator._load_datasets_metadata(
            dataset_paths=dataset_paths,
            desired_magnification=desired_magnification)

        metadata_df = WSITuplesGenerator._standardize_datasets_metadata(
            metadata_df=metadata_df)

        metadata_df = WSITuplesGenerator._validate_metadata(
            metadata_df=metadata_df,
            tile_size=tile_size,
            desired_magnification=desired_magnification,
            minimal_tiles_count=minimal_tiles_count)

        metadata_df = WSITuplesGenerator._add_folds_to_metadata(
            metadata_df=metadata_df,
            folds_count=folds_count)

        metadata_df = WSITuplesGenerator._select_folds_from_metadata(
            metadata_df=metadata_df,
            folds_count=folds_count,
            test_fold=test_fold,
            train=train)

        return metadata_df

    @staticmethod
    def _create_tile_bitmap(original_tile_size, tile_locations, plot_bitmap=False):
        indices = (np.array(tile_locations) / original_tile_size).astype(int)
        dim1_size = indices[:, 0].max() + 1
        dim2_size = indices[:, 1].max() + 1
        bitmap = np.zeros([dim1_size, dim2_size]).astype(int)

        for (x, y) in indices:
            bitmap[x, y] = 1

        tile_bitmap = np.uint8(Image.fromarray(bitmap))

        if plot_bitmap is True:
            plt.imshow(tile_bitmap, cmap='gray')
            plt.show()

        return tile_bitmap

    @staticmethod
    def _create_connected_components(tile_bitmap):
        components_count, components_labels, components_stats, _ = cv2.connectedComponentsWithStats(tile_bitmap)
        components = []

        # if components_count == 1:
        #     continue

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
        largest_component_aspect_ratio = common_utils.calculate_box_aspect_ratio(largest_component['top_left'], largest_component['bottom_right'])
        largest_component_size = largest_component['component_size']
        valid_components = [largest_component]
        for component in components_sorted[1:]:
            current_aspect_ratio = common_utils.calculate_box_aspect_ratio(component['top_left'], component['bottom_right'])
            current_component_size = component['component_size']
            if np.abs(largest_component_aspect_ratio - current_aspect_ratio) < 0.02 and (current_component_size / largest_component_size) > 0.92:
                valid_components.append(component)

        return valid_components

    @staticmethod
    def _get_random_component_index(slide_descriptor):
        components = slide_descriptor['components']
        component_index = int(numpy.random.randint(len(components), size=1))
        return component_index

    def _create_tile_locations(self, dataset_id, image_file_name_stem):
        grid_file_path = os.path.normpath(os.path.join(self._dataset_paths[dataset_id], f'Grids_{self._desired_magnification}', f'{image_file_name_stem}--tlsz{self._tile_size}.data'))
        with open(grid_file_path, 'rb') as file_handle:
            locations = np.array(pickle.load(file_handle))
            locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()

        return locations

    def _create_slide_descriptor(self, row):
        image_file_name = row[WSITuplesGenerator._file_column_name]
        dataset_id = row[WSITuplesGenerator._dataset_id_column_name]
        image_file_path = os.path.join(self._dataset_paths[dataset_id], image_file_name)
        image_file_name_stem = pathlib.Path(image_file_path).stem
        image_file_name_suffix = pathlib.Path(image_file_path).suffix
        magnification = row[WSITuplesGenerator._magnification_column_name]
        legitimate_tiles_count = row[WSITuplesGenerator.get_legitimate_tiles_column_name(tile_size=self._tile_size, desired_magnification=self._desired_magnification)]
        fold = row[WSITuplesGenerator._fold_column_name]
        desired_downsample = magnification / self._desired_magnification
        original_tile_size = self._tile_size * desired_downsample
        tile_locations = self._create_tile_locations(dataset_id=dataset_id, image_file_name_stem=image_file_name_stem)
        tile_bitmap = WSITuplesGenerator._create_tile_bitmap(original_tile_size=original_tile_size, tile_locations=tile_locations, plot_bitmap=False)
        components = WSITuplesGenerator._create_connected_components(tile_bitmap=tile_bitmap)

        slide_descriptor = {
            'image_file_path': image_file_path,
            'image_file_name': image_file_name,
            'image_file_name_stem': image_file_name_stem,
            'image_file_name_suffix': image_file_name_suffix,
            'magnification': magnification,
            'legitimate_tiles_count': legitimate_tiles_count,
            'fold': fold,
            'tile_locations': tile_locations,
            'tile_bitmap': tile_bitmap,
            'desired_downsample': desired_downsample,
            'original_tile_size': original_tile_size,
            'components': components
        }

        return slide_descriptor

    def _create_slide_descriptors(self):
        slide_descriptors = []
        for (index, row) in self._metadata_df.iterrows():
            slide_descriptor = self._create_slide_descriptor(row=row)
            slide_descriptors.append(slide_descriptor)
        return slide_descriptors

    def _create_tuple(self, slide_descriptor, negative_examples_count):
        examples = []

        # Anchor Example
        component_index = WSITuplesGenerator._get_random_component_index(slide_descriptor=slide_descriptor)
        anchor_example = self._create_anchor_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index)

        if anchor_example is None:
            return None

        examples.append(anchor_example)

        # Positive Example
        anchor_point = anchor_example['point']
        positive_example = self._create_positive_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index,
            anchor_point=anchor_point)

        if positive_example is None:
            return None

        examples.append(positive_example)

        # Negative Examples
        for i in range(negative_examples_count):
            negative_example = self._create_negative_example(slide_descriptor=slide_descriptor)
            if negative_example is None:
                return None

            examples.append(negative_example)

        input_tuple = np.transpose(np.stack(examples), (0, 3, 1, 2))

        return input_tuple

    def save_metadata(self, output_file_path):
        self._metadata_df.to_excel(output_file_path)

    def create_tuples(self, negative_examples_count):

        def start(self, load_buffer=False, buffer_base_dir_path='./buffers'):
            self._workers = [Process(target=self._map_func, args=self._args) for i in range(self._num_workers)]
            for i, worker in enumerate(self._workers):
                worker.start()
                print(f'\rWorker Started {i + 1} / {self._num_workers}', end='')

            buffer_type_dir_name = 'train' if self._train is True else 'validation'
            buffer_dir_path = os.path.normpath(os.path.join(buffer_base_dir_path, buffer_type_dir_name))
            buffer_file_name = 'train_buffer.pt' if self._train is True else 'validation_buffer.pt'
            if load_buffer is False:
                print(f'\nItem {len(self._items)} / {self._buffer_size}', end='')
                while True:
                    if self._q.empty() is False:
                        print(f'Queue size = {self._q.qsize()}')
                        self._items.append(self._q.get())
                        print(f'\rItem {len(self._items)} / {self._buffer_size}', end='')
                        if len(self._items) == self._buffer_size:
                            buffers_dir_path = os.path.normpath(os.path.join(buffer_dir_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
                            buffer_file_path = os.path.join(buffers_dir_path, buffer_file_name)
                            Path(buffers_dir_path).mkdir(parents=True, exist_ok=True)
                            torch.save(self._items, buffer_file_path)
                            break
                print('\n')
            else:
                latest_dir_path = utils.get_latest_subdirectory(base_dir=buffer_dir_path)
                buffer_file_path = os.path.join(latest_dir_path, buffer_file_name)
                with open(buffer_file_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    self._items = torch.load(buffer)

        def stop(self):
            for i, worker in enumerate(self._workers):
                worker.terminate()

            self._q.close()









        self._slide_descriptors = self._create_slide_descriptors()
        self._image_file_name_to_slide_descriptor = dict((desc['image_file_name'], desc) for desc in self._slide_descriptors)
        slide_descriptors_count = len(self._slide_descriptors)
        while True:
            slide_descriptor_index = np.random.randint(slide_descriptors_count)
            slide_descriptor = self._slide_descriptors[slide_descriptor_index]
            tuple = self._create_tuple(
                slide_descriptor=slide_descriptor,
                negative_examples_count=negative_examples_count)
            if tuple is None:
                continue

    def __init__(
            self,
            inner_radius,
            outer_radius,
            test_fold,
            train,
            tile_size,
            desired_magnification,
            metadata_file_path=None,
            datasets_base_dir_path=None,
            dataset_ids=None,
            minimal_tiles_count=None,
            folds_count=None):

        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._dataset_paths = WSITuplesGenerator._get_dataset_paths(
            dataset_ids=dataset_ids,
            datasets_base_dir_path=datasets_base_dir_path)

        if metadata_file_path is not None:
            self._metadata_df = WSITuplesGenerator._create_metadata(
                dataset_paths=self._dataset_paths,
                tile_size=self._tile_size,
                desired_magnification=self._desired_magnification,
                minimal_tiles_count=minimal_tiles_count,
                folds_count=folds_count,
                test_fold=test_fold,
                train=train)
        else:
            self._metadata_df = pandas.read_excel(metadata_file_path)

        self._slide_descriptors = None
        self._image_file_name_to_slide_descriptor = None
