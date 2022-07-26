# python core
import os
import pickle
import pathlib
import math
import io
import queue
from datetime import datetime
from pathlib import Path
from torch.multiprocessing import Process, Queue
import queue
import glob
import re
import itertools

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


class WSITuplesOnlineDataset(Dataset):
    def __init__(self, tuplets_generator, replace):
        self._tuplets_generator = tuplets_generator
        self._replace = replace

    def __len__(self):
        return self._tuplets_generator.get_dataset_size()

    def __getitem__(self, index):
        return self._tuplets_generator.get_tuplet(index=index, replace=self._replace)


class WSITupletsGenerator:
    # General parameters
    _test_fold_id = 'test'
    _max_attempts = 10
    _white_ratio_threshold = 0.5
    _histogram_min_intensity_level = 170
    _min_component_ratio = 0.92
    _max_aspect_ratio_diff = 0.02

    # Invalid values
    _invalid_values = ['Missing Data', 'Not performed', '[Not Evaluated]', '[Not Available]']
    _invalid_value = 'NA'
    _invalid_fold_column_names = ['test fold idx breast', 'test fold idx', 'test fold idx breast - original for carmel']

    # Dataset ids
    _dataset_id_abctb = 'ABCTB'
    _dataset_id_sheba = 'SHEBA'
    _dataset_id_carmel = 'CARMEL'
    _dataset_id_tcga = 'TCGA'
    _dataset_id_prefixes = [_dataset_id_abctb, _dataset_id_sheba, _dataset_id_carmel, _dataset_id_tcga]

    # Grid data
    _bad_segmentation_column_name = 'bad segmentation'
    _grids_data_prefix = 'slides_data_'
    _grid_data_file_name = 'Grid_data.xlsx'

    # Unified
    _file_column_name = 'file'
    _patient_barcode_column_name = 'patient_barcode'
    _dataset_id_column_name = 'id'
    _mpp_column_name = 'mpp'
    _scan_date_column_name = 'scan_date'
    _width_column_name = 'width'
    _height_column_name = 'height'
    _magnification_column_name = 'magnification'
    _er_status_column_name = 'er_status'
    _pr_status_column_name = 'pr_status'
    _her2_status_column_name = 'her2_status'
    _fold_column_name = 'fold'
    _grade_column_name = 'grade'
    _tumor_type_column_name = 'tumor_type'
    _slide_barcode_column_name = 'slide_barcode'
    _slide_barcode_prefix_column_name = 'slide_barcode_prefix'
    _legitimate_tiles_column_name = 'legitimate_tiles'
    _total_tiles_column_name = 'total_tiles'
    _tile_usage_column_name = 'tile_usage'

    # Carmel
    _slide_barcode_column_name_carmel = 'slide barcode'
    _slide_barcode_column_name_enhancement_carmel = 'TissueID'
    _patient_barcode_column_name_enhancement_carmel = 'PatientIndex'
    _block_id_column_name_enhancement_carmel = 'BlockID'

    # TCGA
    _patient_barcode_column_name_enhancement_tcga = 'Sample CLID'
    _slide_barcode_prefix_column_name_enhancement_tcga = 'Sample CLID'

    # ABCTB
    _file_column_name_enhancement_abctb = 'Image File'
    _patient_barcode_column_name_enhancement_abctb = 'Identifier'

    # SHEBA
    _er_status_column_name_sheba = 'ER '
    _pr_status_column_name_sheba = 'PR '
    _her2_status_column_name_sheba = 'HER-2 IHC '
    _grade_column_name_sheba = 'Grade'
    _tumor_type_column_name_sheba = 'Histology'

    # Shared
    _file_column_name_shared = 'file'
    _patient_barcode_column_name_shared = 'patient barcode'
    _dataset_id_column_name_shared = 'id'
    _mpp_column_name_shared = 'MPP'
    _scan_date_column_name_shared = 'Scan Date'
    _width_column_name_shared = 'Width'
    _height_column_name_shared = 'Height'
    _magnification_column_name_shared = 'Manipulated Objective Power'
    _er_status_column_name_shared = 'ER status'
    _pr_status_column_name_shared = 'PR status'
    _her2_status_column_name_shared = 'Her2 status'
    _fold_column_name_shared = 'test fold idx'

    @staticmethod
    def _build_path_suffixes():
        path_suffixes = {
            WSITupletsGenerator._dataset_id_tcga: f'Breast/{WSITupletsGenerator._dataset_id_tcga}',
            WSITupletsGenerator._dataset_id_abctb: f'Breast/{WSITupletsGenerator._dataset_id_abctb}_TIF',
        }

        for i in range(1, 12):
            path_suffixes[f'{WSITupletsGenerator._dataset_id_carmel}{i}'] = f'Breast/{WSITupletsGenerator._dataset_id_carmel.capitalize()}/Batch_{i}/{WSITupletsGenerator._dataset_id_carmel}{i}'

        for i in range(2, 7):
            path_suffixes[f'{WSITupletsGenerator._dataset_id_sheba}{i}'] = f'Breast/{WSITupletsGenerator._dataset_id_sheba.capitalize()}/Batch_{i}/{WSITupletsGenerator._dataset_id_sheba}{i}'

        return path_suffixes

    @staticmethod
    def _get_slides_data_file_name(dataset_id):
        return f'slides_data_{dataset_id}.xlsx'

    @staticmethod
    def _get_grids_folder_name(desired_magnification):
        return f'Grids_{str(desired_magnification)}'

    @staticmethod
    def _get_dataset_paths(dataset_ids, datasets_base_dir_path):
        dir_dict = {}
        path_suffixes = WSITupletsGenerator._build_path_suffixes()

        for k in path_suffixes.keys():
            if k in dataset_ids:
                path_suffix = path_suffixes[k]
                dir_dict[k] = os.path.normpath(os.path.join(datasets_base_dir_path, path_suffix))

        return dir_dict

    @staticmethod
    def _get_dataset_id_prefix(dataset_id):
        return ''.join(i for i in dataset_id if not i.isdigit())

    ############
    ### TCGA ###
    ############
    @staticmethod
    def _calculate_grade_tcga(row):
        try:
            column_names = ['Epithelial tubule formation', 'Nuclear pleomorphism', 'Mitosis']
            grade_score = 0
            for column_name in column_names:
                column_score = re.findall(r'\d+', str(row[column_name]))
                if len(column_score) == 0:
                    return 'NA'
                grade_score = grade_score + int(column_score[0])

            if 3 <= grade_score <= 5:
                return 1
            elif 6 <= grade_score <= 7:
                return 2
            else:
                return 3
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_tcga(row):
        try:
            column_name = '2016 Histology Annotations'
            tumor_type = row[column_name]

            if tumor_type == 'Invasive ductal carcinoma':
                return 'IDC'
            elif tumor_type == 'Invasive lobular carcinoma':
                return 'ILC'
            else:
                return 'OTHER'
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_slide_barcode_prefix_tcga(row):
        try:
            return row[WSITupletsGenerator._patient_barcode_column_name_enhancement_tcga]
        except Exception:
            return 'NA'

    #############
    ### ABCTB ###
    #############
    @staticmethod
    def _calculate_grade_abctb(row):
        try:
            column_name = 'Histopathological Grade'
            column_score = re.findall(r'\d+', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(column_score[0])
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_abctb(row):
        try:
            column_name = 'Primary Histologic Diagnosis'
            tumor_type = row[column_name]

            if tumor_type == 'IDC':
                return 'IDC'
            elif tumor_type == 'ILC':
                return 'ILC'
            else:
                return 'OTHER'
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_slide_barcode_prefix_abctb(row):
        try:
            return row[WSITupletsGenerator._file_column_name_enhancement_abctb]
        except Exception:
            return 'NA'

    ##############
    ### CARMEL ###
    ##############
    @staticmethod
    def _calculate_grade_carmel(row):
        try:
            column_name = 'Grade'
            column_score = re.findall(r'\d+(?:\.\d+)?', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(float(column_score[0]))
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_carmel(row):
        try:
            column_name = 'TumorType'
            tumor_type = row[column_name]

            if tumor_type == 'IDC':
                return 'IDC'
            elif tumor_type == 'ILC':
                return 'ILC'
            else:
                return 'OTHER'
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_slide_barcode_prefix_carmel(row):
        try:
            slide_barcode = row[WSITupletsGenerator._slide_barcode_column_name_enhancement_carmel]
            block_id = row[WSITupletsGenerator._block_id_column_name_enhancement_carmel]
            if math.isnan(block_id):
                block_id = 1

            slide_barcode = f"{slide_barcode.replace('/', '_')}_{int(block_id)}"
            return slide_barcode
        except Exception:
            return 'NA'

    #############
    ### SHEBA ###
    #############
    @staticmethod
    def _calculate_grade_sheba(row):
        try:
            column_name = 'Grade'
            column_score = re.findall(r'\d+(?:\.\d+)?', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(float(column_score[0]))
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_sheba(row):
        try:
            column_name = 'Histology'
            tumor_type = row[column_name]

            if tumor_type.startswith('IDC'):
                return 'IDC'
            elif tumor_type.startswith('ILC'):
                return 'ILC'
            else:
                return 'OTHER'
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_slide_barcode_prefix_sheba(row):
        try:
            return row[WSITupletsGenerator._patient_barcode_column_name]
        except Exception:
            return 'NA'

    ###########
    ### ALL ###
    ###########
    @staticmethod
    def _calculate_slide_barcode_prefix(row):
        try:
            dataset_id = row[WSITupletsGenerator._dataset_id_column_name]
            if dataset_id == 'TCGA':
                return row[WSITupletsGenerator._patient_barcode_column_name]
            elif dataset_id == 'ABCTB':
                return row[WSITupletsGenerator._file_column_name].replace('tif', 'ndpi')
            elif dataset_id.startswith('CARMEL'):
                return row[WSITupletsGenerator._slide_barcode_column_name_carmel][:-2]
            elif dataset_id == 'SHEBA':
                return row[WSITupletsGenerator._patient_barcode_column_name]
        except Exception:
            return 'NA'

    # @staticmethod
    # def _tcga_calculate_tumor_type(row):
    #     try:
    #         column_name = '2016 Histology Annotations'
    #         tumor_type = row[column_name]
    #
    #         if tumor_type == 'Invasive ductal carcinoma':
    #             return 'IDC'
    #         elif tumor_type == 'Invasive lobular carcinoma':
    #             return 'ILC'
    #         elif tumor_type == 'Cribriform carcinoma':
    #             return 'ICC'
    #         elif tumor_type == 'Invasive carcinoma with medullary features':
    #             return 'MBC'
    #         elif tumor_type == 'Invasive micropapillary carcinoma':
    #             return 'IMC'
    #         elif tumor_type == 'Metaplastic carcinoma':
    #             return 'MPC'
    #         elif tumor_type == 'Mixed':
    #             return 'MX'
    #         elif tumor_type == 'Mucinous carcinoma':
    #             return 'MC'
    #         elif tumor_type == 'Papillary neoplasm':
    #             return 'PN'
    #         else:
    #             return 'OTHER'
    #     except Exception:
    #         return 'NA'

    @staticmethod
    def _extract_annotations(df, patient_barcode_column_name, calculate_slide_barcode_prefix, calculate_tumor_type, calculate_grade):
        df[WSITupletsGenerator._slide_barcode_prefix_column_name] = df.apply(lambda row: calculate_slide_barcode_prefix(row), axis=1)
        df[WSITupletsGenerator._tumor_type_column_name] = df.apply(lambda row: calculate_tumor_type(row), axis=1)
        df[WSITupletsGenerator._grade_column_name] = df.apply(lambda row: calculate_grade(row), axis=1)

        annotations = df[[
            patient_barcode_column_name,
            WSITupletsGenerator._slide_barcode_prefix_column_name,
            WSITupletsGenerator._grade_column_name,
            WSITupletsGenerator._tumor_type_column_name]]
        annotations = annotations.rename(columns={patient_barcode_column_name: WSITupletsGenerator._patient_barcode_column_name})

        # print('')
        # print('TCGA Enhanced Data:')
        # print('-------------------')
        # print(tcga_data['grade'].value_counts() / tcga_data.shape[0])
        # print('')
        # print(tcga_data['tumor type'].value_counts() / tcga_data.shape[0])

        return annotations

    # @staticmethod
    # def _enhance_metadata_tcga(cell_genomics_tcga_file1_df, cell_genomics_tcga_file2_df):
    #     cell_genomics_tcga_file2_df[WSITuplesGenerator._slide_barcode_prefix_column_name] = cell_genomics_tcga_file2_df.apply(lambda row: WSITuplesGenerator._calculate_slide_barcode_prefix_tcga(row), axis=1)
    #     cell_genomics_tcga_file2_df[WSITuplesGenerator._tumor_type_column_name] = cell_genomics_tcga_file2_df.apply(lambda row: WSITuplesGenerator._calculate_tumor_type_tcga(row), axis=1)
    #     cell_genomics_tcga_file2_df[WSITuplesGenerator._grade_column_name] = cell_genomics_tcga_file2_df.apply(lambda row: WSITuplesGenerator._calculate_grade_tcga(row), axis=1)
    #
    #     tcga_data = cell_genomics_tcga_file2_df[[
    #         WSITuplesGenerator._patient_barcode_column_name_tcga,
    #         WSITuplesGenerator._slide_barcode_prefix_column_name,
    #         WSITuplesGenerator._grade_column_name,
    #         WSITuplesGenerator._tumor_type_column_name]]
    #     tcga_data = tcga_data.rename(columns={WSITuplesGenerator._patient_barcode_column_name_tcga: WSITuplesGenerator._patient_barcode_column_name})
    #
    #     print('')
    #     print('TCGA Enhanced Data:')
    #     print('-------------------')
    #     print(tcga_data['grade'].value_counts() / tcga_data.shape[0])
    #     print('')
    #     print(tcga_data['tumor type'].value_counts() / tcga_data.shape[0])
    #
    #     return tcga_data
    #
    # @staticmethod
    # def _enhance_metadata_carmel(carmel_annotations_26_10_2021_df, carmel_annotations_Batch11_26_10_21_df):
    #     carmel_annotations_26_10_2021_df[WSITuplesGenerator._slide_barcode_prefix_column_name] = carmel_annotations_26_10_2021_df.apply(lambda row: WSITuplesGenerator._calculate_slide_barcode_prefix_carmel(row), axis=1)
    #     carmel_annotations_26_10_2021_df[WSITuplesGenerator._tumor_type_column_name] = carmel_annotations_26_10_2021_df.apply(lambda row: WSITuplesGenerator._calculate_tumor_type_carmel(row), axis=1)
    #     carmel_annotations_26_10_2021_df[WSITuplesGenerator._grade_column_name] = carmel_annotations_26_10_2021_df.apply(lambda row: WSITuplesGenerator._calculate_grade_carmel(row), axis=1)
    #
    #     carmel_annotations_Batch11_26_10_21_df[WSITuplesGenerator._slide_barcode_prefix_column_name] = carmel_annotations_Batch11_26_10_21_df.apply(lambda row: WSITuplesGenerator._calculate_slide_barcode_prefix_carmel(row), axis=1)
    #     carmel_annotations_Batch11_26_10_21_df[WSITuplesGenerator._tumor_type_column_name] = carmel_annotations_Batch11_26_10_21_df.apply(lambda row: WSITuplesGenerator._calculate_tumor_type_carmel(row), axis=1)
    #     carmel_annotations_Batch11_26_10_21_df[WSITuplesGenerator._grade_column_name] = carmel_annotations_Batch11_26_10_21_df.apply(lambda row: WSITuplesGenerator._calculate_grade_carmel(row), axis=1)
    #
    #     carmel_annotations = pandas.concat([carmel_annotations_26_10_2021_df, carmel_annotations_Batch11_26_10_21_df])
    #
    #     carmel_data = carmel_annotations[[
    #         WSITuplesGenerator._patient_barcode_column_name_carmel,
    #         WSITuplesGenerator._slide_barcode_prefix_column_name,
    #         WSITuplesGenerator._grade_column_name,
    #         WSITuplesGenerator._tumor_type_column_name]]
    #
    #     carmel_data = carmel_data.rename(columns={
    #         WSITuplesGenerator._patient_barcode_column_name_carmel: WSITuplesGenerator._patient_barcode_column_name,
    #     })
    #
    #     print('')
    #     print('TCGA Enhanced Data:')
    #     print('-------------------')
    #     print(carmel_data['grade'].value_counts() / carmel_data.shape[0])
    #     print('')
    #     print(carmel_data['tumor type'].value_counts() / carmel_data.shape[0])
    #
    #     return carmel_data
    #
    # @staticmethod
    # def _enhance_metadata_abctb(abctb_path_data_df):
    #     abctb_path_data_df[WSITuplesGenerator._slide_barcode_prefix_column_name] = abctb_path_data_df.apply(lambda row: WSITuplesGenerator._calculate_slide_barcode_prefix_abctb(row), axis=1)
    #     abctb_path_data_df[WSITuplesGenerator._tumor_type_column_name] = abctb_path_data_df.apply(lambda row: WSITuplesGenerator._calculate_tumor_type_abctb(row), axis=1)
    #     abctb_path_data_df[WSITuplesGenerator._grade_column_name] = abctb_path_data_df.apply(lambda row: WSITuplesGenerator._calculate_grade_abctb(row), axis=1)
    #
    #     abctb_data = abctb_path_data_df[[
    #         WSITuplesGenerator._patient_barcode_column_name_abctb,
    #         WSITuplesGenerator._slide_barcode_prefix_column_name,
    #         WSITuplesGenerator._grade_column_name,
    #         WSITuplesGenerator._tumor_type_column_name]]
    #
    #     abctb_data = abctb_data.rename(columns={WSITuplesGenerator._patient_barcode_column_name_abctb: WSITuplesGenerator._patient_barcode_column_name})
    #
    #     print('')
    #     print('TCGA Enhanced Data:')
    #     print('-------------------')
    #     print(abctb_data['grade'].value_counts() / abctb_data.shape[0])
    #     print('')
    #     print(abctb_data['tumor type'].value_counts() / abctb_data.shape[0])
    #
    #     return abctb_data

    @staticmethod
    def _add_slide_barcode_prefix(df):
        df[WSITupletsGenerator._slide_barcode_prefix_column_name] = df.apply(lambda row: WSITupletsGenerator._calculate_slide_barcode_prefix(row), axis=1)
        return df

    @staticmethod
    def _select_metadata(df):
        df = df[[
            WSITupletsGenerator._file_column_name,
            WSITupletsGenerator._patient_barcode_column_name,
            WSITupletsGenerator._dataset_id_column_name,
            WSITupletsGenerator._mpp_column_name,
            WSITupletsGenerator._total_tiles_column_name,
            WSITupletsGenerator._legitimate_tiles_column_name,
            # WSITupletsGenerator._tile_usage_column_name,
            WSITupletsGenerator._width_column_name,
            WSITupletsGenerator._height_column_name,
            WSITupletsGenerator._magnification_column_name,
            WSITupletsGenerator._er_status_column_name,
            WSITupletsGenerator._pr_status_column_name,
            WSITupletsGenerator._her2_status_column_name,
            WSITupletsGenerator._grade_column_name,
            WSITupletsGenerator._tumor_type_column_name,
            WSITupletsGenerator._fold_column_name
        ]]
        return df

    @staticmethod
    def _standardize_metadata(df):
        # fix folds
        pandas.options.mode.chained_assignment = None
        df = df[~df[WSITupletsGenerator._fold_column_name].isin(WSITupletsGenerator._invalid_values)]
        folds = list(df[WSITupletsGenerator._fold_column_name].unique())
        numeric_folds = [common_utils.to_int(fold) for fold in folds]
        # try:
        max_val = numpy.max(numeric_folds) + 1
        df.loc[df[WSITupletsGenerator._fold_column_name] == 'test', WSITupletsGenerator._fold_column_name] = max_val
        df[WSITupletsGenerator._fold_column_name] = df[WSITupletsGenerator._fold_column_name].astype(int)
        # except Exception:
        #     print(folds)
        #     print(numeric_folds)

        # remove invalid values
        df = df.replace(WSITupletsGenerator._invalid_values, WSITupletsGenerator._invalid_value)
        df = df.dropna()
        return df

    @staticmethod
    def _add_folds_to_metadata(metadata_df, folds_count):
        folds = numpy.random.randint(folds_count, size=metadata_df.shape[0])
        metadata_df[WSITupletsGenerator._fold_column_name] = folds
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
    def _create_tile_bitmap(original_tile_size, tile_locations, plot_bitmap=False):
        indices = (numpy.array(tile_locations) / original_tile_size).astype(int)
        dim1_size = indices[:, 0].max() + 1
        dim2_size = indices[:, 1].max() + 1
        bitmap = numpy.zeros([dim1_size, dim2_size]).astype(int)

        for (x, y) in indices:
            bitmap[x, y] = 1

        tile_bitmap = numpy.uint8(Image.fromarray((bitmap * 255).astype(numpy.uint8)))

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
            if numpy.abs(largest_component_aspect_ratio - current_aspect_ratio) < WSITupletsGenerator._max_aspect_ratio_diff and (current_component_size / largest_component_size) > WSITupletsGenerator._min_component_ratio:
                valid_components.append(component)

        return valid_components

    @staticmethod
    def _get_random_component_index(slide_descriptor):
        components = slide_descriptor['components']
        component_index = int(numpy.random.randint(len(components), size=1))
        return component_index

    @staticmethod
    def _start_workers(args, f, workers_count):
        workers = [Process(target=f, args=args[i]) for i in range(workers_count)]
        print('')
        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i + 1} / {workers_count}', end='', flush=True)
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
            # try:
            #     item = q.get_nowait()
            #     items.append(item)
            #     items_count = len(items)
            #     print(f'\rQueue item #{items_count} added', end='')
            #     if items_count == count:
            #         break
            # except queue.Empty:
            #     pass

            item = q.get()
            items.append(item)
            items_count = len(items)
            print(f'\rQueue item #{items_count} added', end='')
            if items_count == count:
                break

        print('')
        return items

    @staticmethod
    def _drain_queue_to_disk(q, count, dir_path, file_name_stem):
        i = 0
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        while True:
            try:
                # item = q.get_nowait()
                # item_file_path = os.path.normpath(os.path.join(dir_path, f'{file_name_stem}_{i}.npy'))
                # numpy.save(item_file_path, item)
                # print(f'\rQueue item #{i} saved', end='')
                # i = i + 1
                # if i == count:
                #     break

                item = q.get()
                item_file_path = os.path.normpath(os.path.join(dir_path, f'{file_name_stem}_{i}.npy'))
                numpy.save(item_file_path, item)
                print(f'\rQueue item #{i} saved', end='')
                i = i + 1
                if i == count:
                    break

            except queue.Empty:
                pass
        print('')

    @staticmethod
    def _append_tiles(tiles, example):
        tiles.append(example['tile'])

    def _build_column_names(self):
        column_names = {}
        for dataset_id_prefix in WSITupletsGenerator._dataset_id_prefixes:
            column_names[dataset_id_prefix] = {}
            column_names[dataset_id_prefix][WSITupletsGenerator._file_column_name_shared] = WSITupletsGenerator._file_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._patient_barcode_column_name_shared] = WSITupletsGenerator._patient_barcode_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._dataset_id_column_name_shared] = WSITupletsGenerator._dataset_id_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._mpp_column_name_shared] = WSITupletsGenerator._mpp_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._scan_date_column_name_shared] = WSITupletsGenerator._scan_date_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._width_column_name_shared] = WSITupletsGenerator._width_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._height_column_name_shared] = WSITupletsGenerator._height_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._magnification_column_name_shared] = WSITupletsGenerator._magnification_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._er_status_column_name_shared] = WSITupletsGenerator._er_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._pr_status_column_name_shared] = WSITupletsGenerator._pr_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._her2_status_column_name_shared] = WSITupletsGenerator._her2_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._fold_column_name_shared] = WSITupletsGenerator._fold_column_name
            column_names[dataset_id_prefix][self._get_total_tiles_column_name()] = WSITupletsGenerator._total_tiles_column_name
            column_names[dataset_id_prefix][self._get_legitimate_tiles_column_name()] = WSITupletsGenerator._legitimate_tiles_column_name
            column_names[dataset_id_prefix][self._get_slide_tile_usage_column_name(dataset_id_prefix=dataset_id_prefix)] = WSITupletsGenerator._tile_usage_column_name

            if dataset_id_prefix.startswith(WSITupletsGenerator._dataset_id_sheba):
                column_names[dataset_id_prefix][WSITupletsGenerator._er_status_column_name_sheba] = WSITupletsGenerator._er_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._pr_status_column_name_sheba] = WSITupletsGenerator._pr_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._her2_status_column_name_sheba] = WSITupletsGenerator._her2_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._grade_column_name_sheba] = WSITupletsGenerator._grade_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._tumor_type_column_name_sheba] = WSITupletsGenerator._tumor_type_column_name

        return column_names

    def _get_total_tiles_column_name(self):
        return f'Total tiles - {self._tile_size} compatible @ X{self._desired_magnification}'

    def _get_legitimate_tiles_column_name(self):
        return f'Legitimate tiles - {self._tile_size} compatible @ X{self._desired_magnification}'

    def _get_slide_tile_usage_column_name(self, dataset_id_prefix):
        if dataset_id_prefix == 'ABCTB':
            return f'Slide tile usage [%] (for {self._tile_size}^2 Pix/Tile)'
        else:
            return f'Slide tile usage [%] (for {self._tile_size}^2 Pix/Tile) @ X{self._desired_magnification}'

    def _load_metadata(self):
        df = None
        for _, dataset_id in enumerate(self._dataset_paths):
            print(f'Processing metadata for {dataset_id}')
            slide_metadata_file = os.path.join(self._dataset_paths[dataset_id], WSITupletsGenerator._get_slides_data_file_name(dataset_id=dataset_id))
            grid_metadata_file = os.path.join(self._dataset_paths[dataset_id], WSITupletsGenerator._get_grids_folder_name(desired_magnification=self._desired_magnification), WSITupletsGenerator._grid_data_file_name)
            slide_df = pandas.read_excel(io=slide_metadata_file)
            grid_df = pandas.read_excel(io=grid_metadata_file)
            current_df = pandas.DataFrame({**slide_df.set_index(keys=WSITupletsGenerator._file_column_name).to_dict(), **grid_df.set_index(keys=WSITupletsGenerator._file_column_name).to_dict()})
            current_df.reset_index(inplace=True)
            current_df.rename(columns={'index': WSITupletsGenerator._file_column_name}, inplace=True)

            current_df = self._prevalidate_metadata(
                df=current_df)

            current_df = self._rename_metadata(
                df=current_df,
                dataset_id_prefix=WSITupletsGenerator._get_dataset_id_prefix(dataset_id=dataset_id))

            current_df = self._enhance_metadata(
                df=current_df,
                dataset_id=dataset_id)

            current_df = WSITupletsGenerator._select_metadata(
                df=current_df)

            current_df = WSITupletsGenerator._standardize_metadata(
                df=current_df)

            current_df = self._postvalidate_metadata(
                df=current_df)

            current_df = self._select_folds_from_metadata(
                df=current_df)

            if df is None:
                df = current_df
            else:
                # df = df.append(current_df)
                df = pandas.concat((df, current_df))
        return df

    def _enhance_metadata_tcga(self, df):
        df = WSITupletsGenerator._add_slide_barcode_prefix(df=df)

        brca_tcga_pan_can_atlas_2018_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', 'brca_tcga_pan_can_atlas_2018_clinical_data.tsv')),
            sep='\t')

        brca_tcga_pub_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', 'brca_tcga_pub_clinical_data.tsv')),
            sep='\t')

        brca_tcga_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', 'brca_tcga_clinical_data.tsv')),
            sep='\t')

        brca_tcga_pub2015_clinical_data_df = pandas.read_csv(
            filepath_or_buffer=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', 'brca_tcga_pub2015_clinical_data.tsv')),
            sep='\t')

        cell_genomics_tcga_file1_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', '1-s2.0-S2666979X21000835-mmc2.xlsx')))

        cell_genomics_tcga_file2_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'TCGA', '1-s2.0-S2666979X21000835-mmc3.xlsx')))

        annotations_tcga = WSITupletsGenerator._extract_annotations(
            df=cell_genomics_tcga_file2_df,
            patient_barcode_column_name=WSITupletsGenerator._patient_barcode_column_name_enhancement_tcga,
            calculate_slide_barcode_prefix=WSITupletsGenerator._calculate_slide_barcode_prefix_tcga,
            calculate_tumor_type=WSITupletsGenerator._calculate_tumor_type_tcga,
            calculate_grade=WSITupletsGenerator._calculate_grade_tcga)

        enhanced_metadata = pandas.concat([annotations_tcga])
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator._patient_barcode_column_name, WSITupletsGenerator._slide_barcode_prefix_column_name])
        return df

    def _enhance_metadata_carmel(self, df):
        df = WSITupletsGenerator._add_slide_barcode_prefix(df=df)

        carmel_annotations_Batch11_26_10_21_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'Carmel', 'Carmel_annotations_Batch11_26-10-21.xlsx')))

        carmel_annotations_26_10_2021_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'Carmel', 'Carmel_annotations_26-10-2021.xlsx')))

        annotations1_carmel = WSITupletsGenerator._extract_annotations(
            df=carmel_annotations_Batch11_26_10_21_df,
            patient_barcode_column_name=WSITupletsGenerator._patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=WSITupletsGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=WSITupletsGenerator._calculate_tumor_type_carmel,
            calculate_grade=WSITupletsGenerator._calculate_grade_carmel)

        annotations2_carmel = WSITupletsGenerator._extract_annotations(
            df=carmel_annotations_26_10_2021_df,
            patient_barcode_column_name=WSITupletsGenerator._patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=WSITupletsGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=WSITupletsGenerator._calculate_tumor_type_carmel,
            calculate_grade=WSITupletsGenerator._calculate_grade_carmel)

        enhanced_metadata = pandas.concat([annotations1_carmel, annotations2_carmel])
        # try:
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator._patient_barcode_column_name, WSITupletsGenerator._slide_barcode_prefix_column_name])
        # except Exception:
        #     h = 5
        return df

    def _enhance_metadata_abctb(self, df):
        df = WSITupletsGenerator._add_slide_barcode_prefix(df=df)

        abctb_path_data_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'ABCTB', 'ABCTB_Path_Data.xlsx')))

        annotations_abctb = WSITupletsGenerator._extract_annotations(
            df=abctb_path_data_df,
            patient_barcode_column_name=WSITupletsGenerator._patient_barcode_column_name_enhancement_abctb,
            calculate_slide_barcode_prefix=WSITupletsGenerator._calculate_slide_barcode_prefix_abctb,
            calculate_tumor_type=WSITupletsGenerator._calculate_tumor_type_abctb,
            calculate_grade=WSITupletsGenerator._calculate_grade_abctb)

        enhanced_metadata = pandas.concat([annotations_abctb])
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator._patient_barcode_column_name, WSITupletsGenerator._slide_barcode_prefix_column_name])
        return df

    def _enhance_metadata(self, df, dataset_id):
        if dataset_id == 'TCGA':
            return self._enhance_metadata_tcga(df=df)
        elif dataset_id.startswith('CARMEL'):
            return self._enhance_metadata_carmel(df=df)
        elif dataset_id == 'ABCTB':
            return self._enhance_metadata_abctb(df=df)

        return df

    def _rename_metadata(self, df, dataset_id_prefix):
        column_names = self._build_column_names()[dataset_id_prefix]
        df = df.rename(columns=column_names)
        return df

    def _prevalidate_metadata(self, df):
        if WSITupletsGenerator._bad_segmentation_column_name in df.columns:
            indices_of_slides_with_bad_seg = set(df.index[df[WSITupletsGenerator._bad_segmentation_column_name] == 1])
        else:
            indices_of_slides_with_bad_seg = set()

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_with_bad_seg))
        return df.iloc[valid_slide_indices]

    def _postvalidate_metadata(self, df):
        indices_of_slides_without_grid = set(df.index[df[WSITupletsGenerator._total_tiles_column_name] == -1])
        indices_of_slides_with_few_tiles = set(df.index[df[WSITupletsGenerator._legitimate_tiles_column_name] < self._minimal_tiles_count])

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_without_grid - indices_of_slides_with_few_tiles))
        return df.iloc[valid_slide_indices]

    def _select_folds_from_metadata(self, df):
        return df[df[WSITupletsGenerator._fold_column_name].isin(self._folds)]

    def _open_slide(self, image_file_path, desired_downsample):
        slide = openslide.open_slide(image_file_path)
        level, adjusted_tile_size = slide_utils.get_best_level_for_downsample(slide=slide, desired_downsample=desired_downsample, tile_size=self._tile_size)

        return {
            'slide': slide,
            'level': level,
            'adjusted_tile_size': adjusted_tile_size
        }

    def _calculate_inner_radius_pixels(self, desired_downsample, image_file_name_suffix, mpp):
        # mm_to_pixel = slide_utils.get_mm_to_pixel(downsample=desired_downsample, image_file_suffix=image_file_name_suffix)
        # inner_radius_pixels = self._inner_radius * mm_to_pixel
        inner_radius_pixels = int(self._inner_radius / (mpp / 1000))
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
            if attempts == WSITupletsGenerator._max_attempts:
                break

            index = int(numpy.random.randint(tile_indices.shape[0], size=1))
            random_tile_indices = tile_indices[index, :]
            location = random_tile_indices * original_tile_size
            point_offset = (original_tile_size * numpy.random.uniform(size=2)).astype(int)
            point = point_offset + location
            bitmap_indices = WSITupletsGenerator._calculate_bitmap_indices(point=point, tile_size=original_tile_size)
            if WSITupletsGenerator._validate_location(bitmap=tiles_bitmap, indices=bitmap_indices) is False:
                attempts = attempts + 1
                continue

            tile = slide_utils.read_region_around_point(slide=slide, point=point, tile_size=self._tile_size, adjusted_tile_size=adjusted_tile_size, level=level)
            tile_grayscale = tile.convert('L')
            hist, _ = numpy.histogram(tile_grayscale, bins=self._tile_size)
            white_ratio = numpy.sum(hist[WSITupletsGenerator._histogram_min_intensity_level:]) / (self._tile_size * self._tile_size)
            if white_ratio > WSITupletsGenerator._white_ratio_threshold:
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
        mpp = slide_descriptor['mpp']
        slide_data = self._open_slide(image_file_path=image_file_path, desired_downsample=desired_downsample)
        component = components[component_index]

        inner_radius_pixels = self._calculate_inner_radius_pixels(desired_downsample=desired_downsample, image_file_name_suffix=image_file_name_suffix, mpp=mpp)
        slide = slide_data['slide']
        adjusted_tile_size = slide_data['adjusted_tile_size']
        level = slide_data['level']
        tiles_bitmap = component['tiles_bitmap']

        attempts = 0
        while True:
            if attempts == WSITupletsGenerator._max_attempts:
                break

            positive_angle = 2 * numpy.pi * numpy.random.uniform(size=1)[0]
            positive_dir = numpy.array([numpy.cos(positive_angle), numpy.sin(positive_angle)])
            # positive_radius = inner_radius_pixels * numpy.random.uniform(size=1)[0]
            positive_radius = inner_radius_pixels
            positive_point = (anchor_point + positive_radius * positive_dir).astype(int)
            positive_bitmap_indices = WSITupletsGenerator._calculate_bitmap_indices(point=positive_point, tile_size=original_tile_size)
            if WSITupletsGenerator._validate_location(bitmap=tiles_bitmap, indices=positive_bitmap_indices) is False:
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
        row_anchor_patient = self._df.loc[self._df[WSITupletsGenerator._file_column_name] == slide_descriptor['image_file_name']].iloc[0]
        patient_barcode = row_anchor_patient[WSITupletsGenerator._patient_barcode_column_name]
        er_status = row_anchor_patient[WSITupletsGenerator._er_status_column_name]
        pr_status = row_anchor_patient[WSITupletsGenerator._pr_status_column_name]
        her2_status = row_anchor_patient[WSITupletsGenerator._her2_status_column_name]
        tumor_type = row_anchor_patient[WSITupletsGenerator._her2_status_column_name]
        grade = row_anchor_patient[WSITupletsGenerator._grade_column_name]
        filtered_df = df[(df[WSITupletsGenerator._patient_barcode_column_name] != patient_barcode) &
                         ((df[WSITupletsGenerator._pr_status_column_name] != pr_status) |
                          (df[WSITupletsGenerator._er_status_column_name] != er_status) |
                          (df[WSITupletsGenerator._her2_status_column_name] != her2_status) |
                          (df[WSITupletsGenerator._tumor_type_column_name] != tumor_type) |
                          (df[WSITupletsGenerator._grade_column_name] != grade))]

        index = int(numpy.random.randint(filtered_df.shape[0], size=1))
        row_negative_patient = filtered_df.iloc[index]
        image_file_name_negative_patient = row_negative_patient[WSITupletsGenerator._file_column_name]
        slide_descriptor_negative_patient = self._image_file_name_to_slide_descriptor[image_file_name_negative_patient]
        component_index = WSITupletsGenerator._get_random_component_index(slide_descriptor=slide_descriptor_negative_patient)
        return self._create_random_example(slide_descriptor=slide_descriptor_negative_patient, component_index=component_index)

    def _create_tile_locations(self, dataset_id, image_file_name_stem):
        grid_file_path = os.path.normpath(os.path.join(self._dataset_paths[dataset_id], f'Grids_{self._desired_magnification}', f'{image_file_name_stem}--tlsz{self._tile_size}.data'))
        with open(grid_file_path, 'rb') as file_handle:
            locations = numpy.array(pickle.load(file_handle))
            locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()

        return locations

    def _create_slide_descriptor(self, row):
        image_file_name = row[WSITupletsGenerator._file_column_name]
        dataset_id = row[WSITupletsGenerator._dataset_id_column_name]
        image_file_path = os.path.join(self._dataset_paths[dataset_id], image_file_name)
        image_file_name_stem = pathlib.Path(image_file_path).stem
        image_file_name_suffix = pathlib.Path(image_file_path).suffix
        magnification = row[WSITupletsGenerator._magnification_column_name]
        # mpp = row[WSITupletsGenerator._mpp_column_name]
        mpp = common_utils.magnification_to_mpp(magnification=magnification)
        legitimate_tiles_count = row[WSITupletsGenerator._legitimate_tiles_column_name]
        fold = row[WSITupletsGenerator._fold_column_name]
        desired_downsample = magnification / self._desired_magnification
        original_tile_size = self._tile_size * desired_downsample
        tile_locations = self._create_tile_locations(dataset_id=dataset_id, image_file_name_stem=image_file_name_stem)
        tile_bitmap = WSITupletsGenerator._create_tile_bitmap(original_tile_size=original_tile_size, tile_locations=tile_locations, plot_bitmap=False)
        components = WSITupletsGenerator._create_connected_components(tile_bitmap=tile_bitmap)

        slide_descriptor = {
            'image_file_path': image_file_path,
            'image_file_name': image_file_name,
            'image_file_name_stem': image_file_name_stem,
            'image_file_name_suffix': image_file_name_suffix,
            'mpp': mpp,
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
        for i in itertools.count(0):
            if indices is not None and i == len(indices):
                break

            tuplet = self._create_tuplet(df=df, negative_examples_count=negative_examples_count)
            q.put(tuplet)

    def _create_tuplet(self, df, negative_examples_count):
        slide_descriptors_count = len(self._slide_descriptors)
        while True:
            slide_descriptor_index = numpy.random.randint(slide_descriptors_count)
            slide_descriptor = self._slide_descriptors[slide_descriptor_index]

            # try:
            tuplet = self._try_create_tuplet(df=df, slide_descriptor=slide_descriptor, negative_examples_count=negative_examples_count)
            # except Exception as e:
            #     tuplet = None
            #     Path(self._dump_dir_path).mkdir(parents=True, exist_ok=True)
            #     slide_descriptor['exception'] = str(e)
            #     with open(os.path.join(self._dump_dir_path, f'slide_descriptor{slide_descriptor_index}.pkl'), 'wb') as f:
            #         pickle.dump(slide_descriptor, f)
            #     print(e)

            if tuplet is not None:
                return tuplet

    def _try_create_tuplet(self, df, slide_descriptor, negative_examples_count):
        tiles = []

        # Anchor Example
        component_index = WSITupletsGenerator._get_random_component_index(slide_descriptor=slide_descriptor)
        anchor_example = self._create_anchor_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index)

        if anchor_example is None:
            return None

        WSITupletsGenerator._append_tiles(tiles=tiles, example=anchor_example)

        # Positive Example
        anchor_point = anchor_example['point']
        positive_example = self._create_positive_example(
            slide_descriptor=slide_descriptor,
            component_index=component_index,
            anchor_point=anchor_point)

        if positive_example is None:
            return None

        WSITupletsGenerator._append_tiles(tiles=tiles, example=positive_example)

        # Negative Examples
        for i in range(negative_examples_count):
            negative_example = self._create_negative_example(df=df, slide_descriptor=slide_descriptor)
            if negative_example is None:
                return None

            WSITupletsGenerator._append_tiles(tiles=tiles, example=negative_example)

        tiles_tuplet = numpy.transpose(numpy.stack(tiles), (0, 3, 1, 2))

        return tiles_tuplet

    def save_metadata(self, output_file_path):
        self._df.to_excel(output_file_path)

    def _create_slide_descriptors(self, df, num_workers):
        q = Queue()
        rows_count = df.shape[0]
        indices = list(range(rows_count))
        indices_groups = common_utils.split(items=indices, n=num_workers)
        args = [(df, indices_group, q) for indices_group in indices_groups]
        workers = WSITupletsGenerator._start_workers(args=args, f=self._slide_descriptors_creation_worker, workers_count=num_workers)
        slide_descriptors = WSITupletsGenerator._drain_queue(q=q, count=rows_count)
        WSITupletsGenerator._join_workers(workers=workers)
        WSITupletsGenerator._stop_workers(workers=workers)
        q.close()
        return slide_descriptors

    def _queue_tuplets(self, tuplets_count, queue_size, negative_examples_count, workers_count):
        self._tuplets_queue = Queue(maxsize=queue_size)
        self._slide_descriptors = self._create_slide_descriptors(df=self._df, num_workers=workers_count)
        self._image_file_name_to_slide_descriptor = dict((desc['image_file_name'], desc) for desc in self._slide_descriptors)

        indices_groups = None
        if tuplets_count < numpy.inf:
            indices = list(range(tuplets_count))
            indices_groups = common_utils.split(items=indices, n=workers_count)

        args = [(self._df, indices_groups, negative_examples_count, self._tuplets_queue) for _ in range(workers_count)]
        self._tuplets_workers = WSITupletsGenerator._start_workers(args=args, f=self._tuples_creation_worker, workers_count=workers_count)

        # WSITuplesGenerator._drain_queue_to_disk(q=q, count=tuples_count, dir_path=dir_path, file_name_stem='tuple')
        # WSITuplesGenerator._join_workers(workers=workers)
        # WSITuplesGenerator._stop_workers(workers=workers)
        # q.close()

    def start_tuplets_creation(self, negative_examples_count, queue_size, workers_count):
        self._queue_tuplets(
            tuplets_count=numpy.inf,
            queue_size=queue_size,
            negative_examples_count=negative_examples_count,
            workers_count=workers_count)

        self._tuplets = WSITupletsGenerator._drain_queue(q=self._tuplets_queue, count=self._dataset_size)

    def stop_tuplets_creation(self):
        # WSITupletsGenerator._join_workers(workers=self._tuplets_workers)
        WSITupletsGenerator._stop_workers(workers=self._tuplets_workers)
        self._tuplets_queue.close()

    def save_tuplets(self, tuplets_count, negative_examples_count, workers_count, dir_path):
        self._queue_tuplets(
            tuplets_count=tuplets_count,
            queue_size=tuplets_count,
            negative_examples_count=negative_examples_count,
            workers_count=workers_count)

        WSITupletsGenerator._drain_queue_to_disk(q=self._tuplets_queue, count=tuplets_count, dir_path=dir_path, file_name_stem='tuple')

    def get_tuplet(self, index, replace=True):
        mod_index = numpy.mod(index, self._dataset_size)
        tuplet = self._tuplets[mod_index]

        if replace is True:
            try:
                new_tuplet = self._tuplets_queue.get_nowait()
                # new_tuplet = self._tuplets_queue.get()
                rand_index = int(numpy.random.randint(self._dataset_size, size=1))
                self._tuplets[rand_index] = new_tuplet
                # print('=== NEW TUPLET ADDED ===')
            except queue.Empty:
                # print('=== QUEUE IS EMPTY ===')
                pass

        return numpy.array(tuplet)

    def get_dataset_size(self):
        return self._dataset_size

    def __init__(
            self,
            folds,
            inner_radius,
            outer_radius,
            tile_size,
            desired_magnification,
            dataset_size,
            datasets_base_dir_path,
            dump_dir_path,
            dataset_ids,
            minimal_tiles_count,
            metadata_enhancement_dir_path):

        self._folds = folds
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._dataset_paths = WSITupletsGenerator._get_dataset_paths(
            dataset_ids=dataset_ids,
            datasets_base_dir_path=datasets_base_dir_path)

        self._minimal_tiles_count = minimal_tiles_count
        self._metadata_enhancement_dir_path = metadata_enhancement_dir_path
        self._dump_dir_path = dump_dir_path

        # if metadata_file_path is not None:
        #     self._minimal_tiles_count = minimal_tiles_count
        #     self._metadata_enhancement_dir_path = metadata_enhancement_dir_path
        #     self._metadata_df = self._load_metadata()
        # else:
        #     self._metadata_df = pandas.read_excel(metadata_file_path)

        self._dataset_size = dataset_size
        self._slide_descriptors = []
        self._image_file_name_to_slide_descriptor = {}
        self._tuplets_queue = None
        self._tuplets_workers = None
        self._tuplets = None

        self._df = self._load_metadata()
        # self._df = self._df.iloc[[10]]

