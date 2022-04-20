# python core
import os
import pickle
import pathlib
import math
import io
import queue
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Queue, connection, current_process
import queue
import glob

# pandas
import pandas

# numpy
import numpy

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


class WSITuplesDataset(Dataset):
    def __init__(self, dir_path):
        self._tuple_files = [f for f in glob.glob(os.path.normpath(os.path.join(dir_path, '*.npy')))]

    def __len__(self):
        return len(self._tuple_files)

    def __getitem__(self, index):
        return numpy.load(self._tuple_files[index])


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
    _min_component_ratio = 0.92
    _max_aspect_ratio_diff = 0.02

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
            slide_metadata_df = pandas.read_excel(io=slide_metadata_file)
            grid_metadata_df = pandas.read_excel(io=grid_metadata_file)
            current_metadata_df = pandas.DataFrame({**slide_metadata_df.set_index(keys=WSITuplesGenerator._file_column_name).to_dict(), **grid_metadata_df.set_index(keys=WSITuplesGenerator._file_column_name).to_dict()})
            if metadata_df is None:
                metadata_df = current_metadata_df
            else:
                metadata_df.append(metadata_df)

        metadata_df.reset_index(inplace=True)
        metadata_df.rename(columns={'index': WSITuplesGenerator._file_column_name}, inplace=True)
        return metadata_df

    @staticmethod
    def _enhance_datasets_metadata(enhancement_dir_path, metadata_df):
        brca_tcga_pan_can_atlas_2018_df = pandas.read_excel(io=os.path.join(enhancement_dir_path, 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv'))

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

        all_indices = set(numpy.array(range(metadata_df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_without_grid - indices_of_slides_with_few_tiles - indices_of_slides_with_0_tiles - indices_of_slides_with_bad_seg))
        return metadata_df.iloc[valid_slide_indices]

    @staticmethod
    def _add_folds_to_metadata(metadata_df, folds_count):
        folds = numpy.random.randint(folds_count, size=metadata_df.shape[0])
        metadata_df[WSITuplesGenerator._fold_column_name] = folds
        return metadata_df

    @staticmethod
    def _calculate_bitmap_indices(point, tile_size):
        bitmap_indices = (point / tile_size).astype(int)
        return bitmap_indices

    @staticmethod
    def _validate_location(bitmap, indices):
        for i in range(3):
            for j in range(3):
                current_indices = tuple(indices + numpy.array([i, j]))
                if (not (0 <= current_indices[0] < bitmap.shape[0])) or (not (0 <= current_indices[1] < bitmap.shape[1])):
                    return False

                bit = bitmap[current_indices]
                if bit == 0:
                    return False
        return True

    @staticmethod
    def _create_metadata(
            dataset_paths,
            tile_size,
            desired_magnification,
            minimal_tiles_count,
            folds_count,
            enhancement_dir_path):

        metadata_df = WSITuplesGenerator._load_datasets_metadata(
            dataset_paths=dataset_paths,
            desired_magnification=desired_magnification)

        metadata_df = WSITuplesGenerator._enhance_datasets_metadata(
            metadata_df=metadata_df,
            enhancement_dir_path=enhancement_dir_path)

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

        return metadata_df

    @staticmethod
    def _create_tile_bitmap(original_tile_size, tile_locations, plot_bitmap=False):
        indices = (numpy.array(tile_locations) / original_tile_size).astype(int)
        dim1_size = indices[:, 0].max() + 1
        dim2_size = indices[:, 1].max() + 1
        bitmap = numpy.zeros([dim1_size, dim2_size]).astype(int)

        for (x, y) in indices:
            bitmap[x, y] = 1

        tile_bitmap = numpy.uint8(Image.fromarray(bitmap))

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
            component_indices = numpy.where(current_bitmap)
            component_size = numpy.count_nonzero(current_bitmap)
            top_left = numpy.array([numpy.min(component_indices[0]), numpy.min(component_indices[1])])
            bottom_right = numpy.array([numpy.max(component_indices[0]), numpy.max(component_indices[1])])
            tile_indices = numpy.array([component_indices[0], component_indices[1]]).transpose()

            components.append({
                'tiles_bitmap': current_bitmap.astype(int),
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
            if numpy.abs(largest_component_aspect_ratio - current_aspect_ratio) < WSITuplesGenerator._max_aspect_ratio_diff and (current_component_size / largest_component_size) > WSITuplesGenerator._min_component_ratio:
                valid_components.append(component)

        return valid_components

    @staticmethod
    def _get_random_component_index(slide_descriptor):
        components = slide_descriptor['components']
        component_index = int(numpy.random.randint(len(components), size=1))
        return component_index

    @staticmethod
    def _start_workers(args, f, num_workers):
        workers = [Process(target=f, args=args[i]) for i in range(num_workers)]
        print('')
        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i + 1} / {num_workers}', end='', flush=True)
        print('')

        return workers

    @staticmethod
    def _join_workers(workers):
        for i, worker in enumerate(workers):
            worker.join()

    @staticmethod
    def _stop_workers(workers):
        for i, worker in enumerate(workers):
            worker.terminate()

    @staticmethod
    def _drain_queue(q, count):
        items = []
        while True:
            try:
                item = q.get_nowait()
                items.append(item)
                if len(items) == count:
                    break
            except queue.Empty:
                pass
        return items

    @staticmethod
    def _drain_queue_to_disk(q, count, dir_path, file_name_stem):
        i = 0
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        while True:
            try:
                item = q.get_nowait()
                item_file_path = os.path.normpath(os.path.join(dir_path, f'{file_name_stem}_{i}.npy'))
                numpy.save(item_file_path, item)
                i = i + 1
                if i == count:
                    break
            except queue.Empty:
                pass

    @staticmethod
    def _append_tiles(tiles, example):
        tiles.append(example['tile'])

    def _select_folds_from_metadata(self, folds):
        return self._metadata_df[self._metadata_df[WSITuplesGenerator._fold_column_name].isin(folds)]

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
        tiles_bitmap = component['tiles_bitmap']
        slide = slide_data['slide']
        adjusted_tile_size = slide_data['adjusted_tile_size']
        level = slide_data['level']

        attempts = 0
        while True:
            if attempts == WSITuplesGenerator._max_attempts:
                break

            index = int(numpy.random.randint(tile_indices.shape[0], size=1))
            random_tile_indices = tile_indices[index, :]
            location = random_tile_indices * original_tile_size
            point_offset = (original_tile_size * numpy.random.uniform(size=2)).astype(int)
            point = point_offset + location
            bitmap_indices = WSITuplesGenerator._calculate_bitmap_indices(point=point, tile_size=original_tile_size)
            if WSITuplesGenerator._validate_location(bitmap=tiles_bitmap, indices=bitmap_indices) is False:
                attempts = attempts + 1
                continue

            tile = slide_utils.read_region_around_point(slide=slide, point=point, tile_size=self._tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
            tile_grayscale = tile.convert('L')
            hist, _ = numpy.histogram(tile_grayscale, bins=self._tile_size)
            white_ratio = numpy.sum(hist[WSITuplesGenerator._histogram_min_intensity_level:]) / (self._tile_size * self._tile_size)
            if white_ratio > WSITuplesGenerator._white_ratio_threshold:
                attempts = attempts + 1
                continue

            return {
                'tile': numpy.array(tile),
                'point': point
            }

        return None

    def _create_anchor_example(self, slide_descriptor, component_index):
        return self._create_random_example(slide_descriptor=slide_descriptor, component_index=component_index)

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
        tiles_bitmap = component['tiles_bitmap']

        attempts = 0
        while True:
            if attempts == WSITuplesGenerator._max_attempts:
                break

            positive_angle = 2 * numpy.pi * numpy.random.uniform(size=1)[0]
            positive_dir = numpy.array([numpy.cos(positive_angle), numpy.sin(positive_angle)])
            positive_radius = inner_radius_pixels * numpy.random.uniform(size=1)[0]
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

            return {
                'tile': numpy.array(positive_tile),
                'point': positive_point
            }

        return None

    def _create_negative_example(self, df, slide_descriptor):
        row_anchor_patient = self._metadata_df.loc[self._metadata_df[WSITuplesGenerator._file_column_name] == slide_descriptor['image_file_name']].iloc[0]
        patient_barcode = row_anchor_patient[WSITuplesGenerator._patient_barcode_column_name]
        er_status = row_anchor_patient[WSITuplesGenerator._er_status_column_name]
        pr_status = row_anchor_patient[WSITuplesGenerator._pr_status_column_name]
        her2_status = row_anchor_patient[WSITuplesGenerator._her2_status_column_name]
        filtered_df = df[(df[WSITuplesGenerator._patient_barcode_column_name] != patient_barcode) &
                         (df[WSITuplesGenerator._pr_status_column_name] != pr_status) &
                         (df[WSITuplesGenerator._er_status_column_name] != er_status) &
                         (df[WSITuplesGenerator._her2_status_column_name] != her2_status)]

        index = int(numpy.random.randint(filtered_df.shape[0], size=1))
        row_negative_patient = filtered_df.iloc[index]
        image_file_name_negative_patient = row_negative_patient[WSITuplesGenerator._file_column_name]
        slide_descriptor_negative_patient = self._image_file_name_to_slide_descriptor[image_file_name_negative_patient]
        component_index = WSITuplesGenerator._get_random_component_index(slide_descriptor=slide_descriptor_negative_patient)
        return self._create_random_example(slide_descriptor=slide_descriptor_negative_patient, component_index=component_index)

    def _create_tile_locations(self, dataset_id, image_file_name_stem):
        grid_file_path = os.path.normpath(os.path.join(self._dataset_paths[dataset_id], f'Grids_{self._desired_magnification}', f'{image_file_name_stem}--tlsz{self._tile_size}.data'))
        with open(grid_file_path, 'rb') as file_handle:
            locations = numpy.array(pickle.load(file_handle))
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

    def _slide_descriptors_creation_worker(self, df, indices, q):
        df = df.iloc[indices]
        for (index, row) in df.iterrows():
            slide_descriptor = self._create_slide_descriptor(row=row)
            q.put(slide_descriptor)

    def _tuples_creation_worker(self, df, indices, negative_examples_count, q):
        slide_descriptors_count = len(self._slide_descriptors)
        for _ in indices:
            while True:
                slide_descriptor_index = numpy.random.randint(slide_descriptors_count)
                slide_descriptor = self._slide_descriptors[slide_descriptor_index]
                tiles_tuple = self._create_tuple(df=df, slide_descriptor=slide_descriptor, negative_examples_count=negative_examples_count)
                if tiles_tuple is not None:
                    break
            q.put(tiles_tuple)

    def _create_tuple(self, df, slide_descriptor, negative_examples_count):
        tiles = []

        # Anchor Example
        component_index = WSITuplesGenerator._get_random_component_index(slide_descriptor=slide_descriptor)
        anchor_example = self._create_anchor_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index)

        if anchor_example is None:
            return None

        WSITuplesGenerator._append_tiles(tiles=tiles, example=anchor_example)

        # Positive Example
        anchor_point = anchor_example['point']
        positive_example = self._create_positive_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index,
            anchor_point=anchor_point)

        if positive_example is None:
            return None

        WSITuplesGenerator._append_tiles(tiles=tiles, example=positive_example)

        # Negative Examples
        for i in range(negative_examples_count):
            negative_example = self._create_negative_example(df=df, slide_descriptor=slide_descriptor)
            if negative_example is None:
                return None

            WSITuplesGenerator._append_tiles(tiles=tiles, example=negative_example)

        tiles_tuple = numpy.transpose(numpy.stack(tiles), (0, 3, 1, 2))

        return tiles_tuple

    def save_metadata(self, output_file_path):
        self._metadata_df.to_excel(output_file_path)

    def _create_slide_descriptors(self, df, num_workers):
        q = Queue()
        rows_count = df.shape[0]
        indices = list(range(rows_count))
        indices_groups = common_utils.split(items=indices, n=num_workers)
        args = [(df, indices_group, q) for indices_group in indices_groups]
        workers = WSITuplesGenerator._start_workers(args=args, f=self._slide_descriptors_creation_worker, num_workers=num_workers)
        slide_descriptors = WSITuplesGenerator._drain_queue(q=q, count=rows_count)
        WSITuplesGenerator._join_workers(workers=workers)
        WSITuplesGenerator._stop_workers(workers=workers)
        q.close()
        return slide_descriptors

    def create_tuples(self, tuples_count, negative_examples_count, folds, dir_path, num_workers):
        q = Queue()
        df = self._select_folds_from_metadata(folds=folds)
        self._slide_descriptors = self._create_slide_descriptors(df=df, num_workers=num_workers)
        self._image_file_name_to_slide_descriptor = dict((desc['image_file_name'], desc) for desc in self._slide_descriptors)
        indices = list(range(tuples_count))
        indices_groups = common_utils.split(items=indices, n=num_workers)
        args = [(df, indices_group, negative_examples_count, q) for indices_group in indices_groups]
        workers = WSITuplesGenerator._start_workers(args=args, f=self._tuples_creation_worker, num_workers=num_workers)
        # tuples = WSITuplesGenerator._drain_queue(q=q, count=tuples_count)
        WSITuplesGenerator._drain_queue_to_disk(q=q, count=tuples_count, dir_path=dir_path, file_name_stem='tuple')
        WSITuplesGenerator._join_workers(workers=workers)
        WSITuplesGenerator._stop_workers(workers=workers)
        q.close()
        # return tuples

    def get_folds_count(self):
        return self._metadata_df[WSITuplesGenerator._fold_column_name].nunique() + 1

    def __init__(
            self,
            inner_radius,
            outer_radius,
            tile_size,
            desired_magnification,
            metadata_file_path=None,
            datasets_base_dir_path=None,
            dataset_ids=None,
            minimal_tiles_count=None,
            folds_count=None,
            enhancement_dir_path=None):

        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._dataset_paths = WSITuplesGenerator._get_dataset_paths(
            dataset_ids=dataset_ids,
            datasets_base_dir_path=datasets_base_dir_path)
        self._enhancement_dir_path = enhancement_dir_path

        if metadata_file_path is not None:
            self._metadata_df = WSITuplesGenerator._create_metadata(
                dataset_paths=self._dataset_paths,
                tile_size=self._tile_size,
                desired_magnification=self._desired_magnification,
                minimal_tiles_count=minimal_tiles_count,
                folds_count=folds_count)
        else:
            self._metadata_df = pandas.read_excel(metadata_file_path)

        self._slide_descriptors = []
        self._image_file_name_to_slide_descriptor = {}
