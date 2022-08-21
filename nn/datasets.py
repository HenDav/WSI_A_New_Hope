# python core
import os
import pickle
import pathlib
import math
import io
import queue
import glob
import re
import itertools
import logging
from datetime import datetime
from pathlib import Path
from torch.multiprocessing import Process, Queue
from typing import List, Set, Tuple, Dict, Union, Callable

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


class WSITupletsOnlineDataset(Dataset):
    def __init__(self, tuplets_generator):
        self._tuplets_generator = tuplets_generator
        # self._replace = replace

    def __len__(self):
        return self._tuplets_generator.get_dataset_size()

    def __getitem__(self, index):
        return self._tuplets_generator.get_tuplet(index=index)
        # return self._tuplets_generator.get_tuplet(index=index, replace=self._replace)


class SlideContext:
    def __init__(self, row: pandas.Series, dataset_path: str, desired_magnification: int, tile_size: int):
        self._row = row
        self._dataset_path = dataset_path
        self._desired_magnification = desired_magnification
        self._tile_size = tile_size
        self._image_file_name = self._row[WSITupletsGenerator.file_column_name]
        self._image_file_path = os.path.join(dataset_path, self._image_file_name)
        self._dataset_id = self._row[WSITupletsGenerator.dataset_id_column_name]
        self._image_file_name_stem = pathlib.Path(self._image_file_path).stem
        self._image_file_name_suffix = pathlib.Path(self._image_file_path).suffix
        self._magnification = row[WSITupletsGenerator.magnification_column_name]
        self._mpp = common_utils.magnification_to_mpp(magnification=self._magnification)
        self._legitimate_tiles_count = row[WSITupletsGenerator.legitimate_tiles_column_name]
        self._fold = row[WSITupletsGenerator.fold_column_name]
        self._desired_downsample = self._magnification / self._desired_magnification
        self._slide = openslide.open_slide(self._image_file_path)
        self._level, self._level_downsample = self._get_best_level_for_downsample()
        self._selected_level_tile_size = self._tile_size * self._level_downsample
        self._zero_level_tile_size = self._tile_size * self._desired_downsample

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def desired_magnification(self) -> int:
        return self._desired_magnification

    @property
    def image_file_name(self) -> str:
        return self._image_file_name

    @property
    def image_file_path(self) -> str:
        return self._image_file_path

    @property
    def image_file_name_stem(self) -> str:
        return self._image_file_name_stem

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def slide(self) -> openslide.OpenSlide:
        return self._slide

    @property
    def level(self) -> int:
        return self._level

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def selected_level_tile_size(self) -> int:
        return self._selected_level_tile_size

    @property
    def selected_level_half_tile_size(self) -> int:
        return self._selected_level_tile_size // 2

    @property
    def zero_level_tile_size(self) -> int:
        return self._zero_level_tile_size

    @property
    def zero_level_half_tile_size(self) -> int:
        return self._zero_level_tile_size // 2

    @property
    def mpp(self) -> float:
        return self._mpp

    def mm_to_pixels(self, mm: float) -> int:
        pixels = int(mm / (self._mpp / 1000))
        return pixels

    def read_region_around_pixel(self, pixel: numpy.ndarray) -> Image:
        top_left_pixel = (pixel - self.selected_level_tile_size / 2).astype(int)
        region = self.slide.read_region(top_left_pixel, self.level, (self.selected_level_tile_size, self.selected_level_tile_size)).convert('RGB')
        if self.selected_level_tile_size != self.tile_size:
            region = region.resize((self.tile_size, self.tile_size))
        return region

    def _get_best_level_for_downsample(self):
        level = 0
        level_downsample = self._desired_downsample
        if self._desired_downsample > 1:
            for i, downsample in enumerate(self._slide.level_downsamples):
                if math.isclose(self._desired_downsample, downsample, rel_tol=1e-3):
                    level = i
                    level_downsample = 1
                    break
                elif downsample < self._desired_downsample:
                    level = i
                    level_downsample = int(self._desired_downsample / self._slide.level_downsamples[level])

        # A tile of size (tile_size, tile_size) in an image downsampled by 'level_downsample', will cover the same image portion of a tile of size (adjusted_tile_size, adjusted_tile_size) in the original image
        return level, level_downsample


class SlideElement:
    def __init__(self, slide_context: SlideContext):
        self._slide_context = slide_context

    @property
    def slide_context(self) -> SlideContext:
        return self._slide_context


class Tile(SlideElement):
    def __init__(self, slide_context: SlideContext, tile_location: numpy.ndarray):
        super().__init__(slide_context=slide_context)
        self._slide_context = slide_context
        self._tile_location = tile_location

    @property
    def tile_location(self) -> numpy.ndarray:
        return self._tile_location

    def get_random_pixel(self) -> numpy.ndarray:
        pixel = self._tile_location * self._slide_context.zero_level_tile_size + self._slide_context.zero_level_half_tile_size
        offset = (self._slide_context.zero_level_half_tile_size * numpy.random.uniform(size=2)).astype(int)
        pixel = pixel + offset
        return pixel


class ConnectedComponent(SlideElement):
    def __init__(self, slide_context: SlideContext, bitmap: numpy.ndarray):
        super().__init__(slide_context=slide_context)
        self._bitmap = bitmap
        self._valid_tiles_count = numpy.count_nonzero(self._bitmap)
        self._valid_tile_indices = numpy.where(self._bitmap)
        self._tile_locations = numpy.array([self._valid_tile_indices[0], self._valid_tile_indices[1]]).transpose()
        self._top_left_tile_location = numpy.array([numpy.min(self._valid_tile_indices[0]), numpy.min(self._valid_tile_indices[1])])
        self._bottom_right_tile_location = numpy.array([numpy.max(self._valid_tile_indices[0]), numpy.max(self._valid_tile_indices[1])])
        self._tiles_list = self._create_tiles_list()
        self._tiles_dict = self._create_tiles_dict()

    @property
    def bitmap(self) -> numpy.ndarray:
        return self._bitmap

    @property
    def valid_tiles_count(self) -> int:
        return self._valid_tiles_count

    @property
    def top_left_tile_location(self) -> numpy.ndarray:
        return self._top_left_tile_location

    @property
    def bottom_right_tile_location(self) -> numpy.ndarray:
        return self._bottom_right_tile_location

    @property
    def tile_locations(self) -> numpy.ndarray:
        return self._tile_locations

    def _create_tiles_list(self) -> List[Tile]:
        tiles = []
        for i in range(self._valid_tiles_count):
            tiles.append(Tile(slide_context=self._slide_context, tile_location=self._tile_locations[i, :]))
        return tiles

    def _create_tiles_dict(self) -> Dict[bytes, Tile]:
        tiles_dict = {}
        for tile in self._tiles_list:
            tiles_dict[tile.tile_location.tobytes()] = tile

        return tiles_dict

    def get_random_tile(self) -> Tile:
        tile_index = int(numpy.random.randint(self._valid_tiles_count, size=1))
        return self._tiles_list[tile_index]

    def get_random_pixel(self) -> numpy.ndarray:
        tile = self.get_random_tile()
        return tile.get_random_pixel()

    def is_interior_tile(self, tile: Tile) -> bool:
        for i in range(3):
            for j in range(3):
                current_tile_location = tile.tile_location + numpy.array([i, j])
                if not self.is_valid_tile_location(tile_location=current_tile_location):
                    return False

                bit = self._bitmap[current_tile_location]
                if bit == 0:
                    return False
        return True

    def get_tile_at_pixel(self, pixel: numpy.ndarray) -> Union[Tile, None]:
        tile_location = (pixel / self._slide_context.zero_level_tile_size).astype(int)
        if self.is_valid_tile_location(tile_location=tile_location):
            return self._tiles_dict[tile_location.tobytes()]
        return None

    def is_valid_tile_location(self, tile_location: numpy.ndarray) -> bool:
        if tile_location.tobytes() in self._tiles_dict:
            return True
        return False


class Patch(SlideElement):
    def __init__(self, slide_context: SlideContext, component: ConnectedComponent, center_pixel: numpy.ndarray):
        super().__init__(slide_context=slide_context)
        self._component = component
        self._center_pixel = center_pixel
        self._image = self._slide_context.read_region_around_pixel(pixel=center_pixel)

    @property
    def image(self) -> Image:
        return self._image

    @property
    def component(self) -> ConnectedComponent:
        return self._component

    @property
    def center_pixel(self) -> numpy.ndarray:
        return self._center_pixel

    # def get_white_ratio(self, white_intensity_threshold: int) -> float:
    #     patch_grayscale = self._image.convert('L')
    #     hist, _ = numpy.histogram(a=patch_grayscale, bins=self._slide_context.tile_size)
    #     white_ratio = numpy.sum(hist[white_intensity_threshold:]) / (self._slide_context.tile_size * self._slide_context.tile_size)
    #     return white_ratio

    def get_containing_tile(self) -> Union[Tile, None]:
        return self._component.get_tile_at_pixel(self._center_pixel)


class Slide(SlideElement):
    _min_component_ratio = 0.92
    _max_aspect_ratio_diff = 0.02

    def __init__(self, slide_context: SlideContext):
        super().__init__(slide_context=slide_context)
        self._tile_locations = self._create_tile_locations()
        self._tile_bitmap = self._create_tile_bitmap(plot_bitmap=False)
        self._tile_components = self._create_connected_components()

    @property
    def components(self):
        return self._tile_components

    def get_component(self, component_index: int) -> ConnectedComponent:
        return self._tile_components[component_index]

    def get_random_component(self) -> ConnectedComponent:
        component_index = int(numpy.random.randint(len(self._tile_components), size=1))
        return self.get_component(component_index=component_index)

    def _create_tile_bitmap(self, plot_bitmap: bool = False) -> Image:
        indices = (numpy.array(self._tile_locations) / self._slide_context.zero_level_tile_size).astype(int)
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

    def _create_tile_locations(self) -> numpy.ndarray:
        grid_file_path = os.path.normpath(os.path.join(self._slide_context.dataset_path, f'Grids_{self._slide_context.desired_magnification}', f'{self._slide_context.image_file_name_stem}--tlsz{self._slide_context.tile_size}.data'))
        with open(grid_file_path, 'rb') as file_handle:
            locations = numpy.array(pickle.load(file_handle))
            locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()

        return locations

    def _create_connected_components(self) -> List[ConnectedComponent]:
        components_count, components_labels, components_stats, _ = cv2.connectedComponentsWithStats(self._tile_bitmap)
        components = []

        for component_id in range(1, components_count):
            component_bitmap = (components_labels == component_id).astype(int)
            components.append(ConnectedComponent(bitmap=component_bitmap))

        components_sorted = sorted(components, key=lambda item: item.component_size, reverse=True)
        largest_component = components_sorted[0]
        largest_component_aspect_ratio = common_utils.calculate_box_aspect_ratio(largest_component.top_left, largest_component.bottom_right)
        largest_component_size = largest_component.valid_tiles_count
        valid_components = [largest_component]
        for component in components_sorted[1:]:
            current_aspect_ratio = common_utils.calculate_box_aspect_ratio(component.top_left, component.bottom_right)
            current_component_size = component.valid_tiles_count
            if (numpy.abs(largest_component_aspect_ratio - current_aspect_ratio) < Slide._max_aspect_ratio_diff) and ((current_component_size / largest_component_size) > Slide._min_component_ratio):
                valid_components.append(component)

        return valid_components


class PatchExtractor:
    _max_attempts = 10
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

    def __init__(self, slide: Slide, inner_radius_mm: float):
        self._slide = slide
        self._inner_radius_mm = inner_radius_mm
        self._inner_radius_pixels = self._slide.slide_context.mm_to_pixels(mm=inner_radius_mm)

    def extract_patch(self, patch_validators: List[Callable[[Patch], bool]], reference_patch: Patch = None) -> Patch:
        attempts = 0
        while True:
            if attempts == PatchExtractor._max_attempts:
                break

            if reference_patch is None:
                component = self._slide.get_random_component()
                pixel = component.get_random_pixel()
            else:
                component = reference_patch.component
                pixel = reference_patch.center_pixel

            angle = 2 * numpy.pi * numpy.random.uniform(size=1)[0]
            direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
            proximate_pixel = (pixel + self._inner_radius_pixels * direction).astype(int)
            tile = component.get_tile_at_pixel(pixel=proximate_pixel)
            if (tile is None) or (not component.is_interior_tile(tile=tile)):
                attempts = attempts + 1
                continue

            patch = Patch(slide_context=self._slide.slide_context, component=component, center_pixel=proximate_pixel)
            patch_validation_failed = False
            for patch_validator in patch_validators:
                if not patch_validator(patch):
                    attempts = attempts + 1
                    patch_validation_failed = True
                    break

            if patch_validation_failed is True:
                attempts = attempts + 1
                continue

            return patch


class WSITupletsGenerator:
    # General parameters
    _test_fold_id = 'test'
    _max_attempts = 10
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

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

    # Curated
    file_column_name = 'file'
    patient_barcode_column_name = 'patient_barcode'
    dataset_id_column_name = 'id'
    mpp_column_name = 'mpp'
    scan_date_column_name = 'scan_date'
    width_column_name = 'width'
    height_column_name = 'height'
    magnification_column_name = 'magnification'
    er_status_column_name = 'er_status'
    pr_status_column_name = 'pr_status'
    her2_status_column_name = 'her2_status'
    fold_column_name = 'fold'
    grade_column_name = 'grade'
    tumor_type_column_name = 'tumor_type'
    slide_barcode_column_name = 'slide_barcode'
    slide_barcode_prefix_column_name = 'slide_barcode_prefix'
    legitimate_tiles_column_name = 'legitimate_tiles'
    total_tiles_column_name = 'total_tiles'
    tile_usage_column_name = 'tile_usage'

    @staticmethod
    def _build_path_suffixes() -> Dict:
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
    def _get_slides_data_file_name(dataset_id: str) -> str:
        return f'slides_data_{dataset_id}.xlsx'

    @staticmethod
    def _get_grids_folder_name(desired_magnification: int) -> str:
        return f'Grids_{desired_magnification}'

    @staticmethod
    def _get_dataset_paths(dataset_ids: List[str], datasets_base_dir_path: str) -> Dict:
        dir_dict = {}
        path_suffixes = WSITupletsGenerator._build_path_suffixes()

        for k in path_suffixes.keys():
            if k in dataset_ids:
                path_suffix = path_suffixes[k]
                dir_dict[k] = os.path.normpath(os.path.join(datasets_base_dir_path, path_suffix))

        return dir_dict

    @staticmethod
    def _get_dataset_id_prefix(dataset_id: str) -> str:
        return ''.join(i for i in dataset_id if not i.isdigit())

    ############
    ### TCGA ###
    ############
    @staticmethod
    def _calculate_grade_tcga(row: pandas.Series) -> Union[str, int]:
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
    def _calculate_tumor_type_tcga(row: pandas.Series) -> str:
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
    def _calculate_slide_barcode_prefix_tcga(row: pandas.Series) -> str:
        try:
            return row[WSITupletsGenerator._patient_barcode_column_name_enhancement_tcga]
        except Exception:
            return 'NA'

    #############
    ### ABCTB ###
    #############
    @staticmethod
    def _calculate_grade_abctb(row: pandas.Series) -> Union[str, int]:
        try:
            column_name = 'Histopathological Grade'
            column_score = re.findall(r'\d+', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(column_score[0])
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_abctb(row: pandas.Series) -> str:
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
    def _calculate_slide_barcode_prefix_abctb(row: pandas.Series) -> str:
        try:
            return row[WSITupletsGenerator._file_column_name_enhancement_abctb]
        except Exception:
            return 'NA'

    ##############
    ### CARMEL ###
    ##############
    @staticmethod
    def _calculate_grade_carmel(row: pandas.Series) -> Union[str, int]:
        try:
            column_name = 'Grade'
            column_score = re.findall(r'\d+(?:\.\d+)?', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(float(column_score[0]))
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_carmel(row: pandas.Series) -> str:
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
    def _calculate_slide_barcode_prefix_carmel(row: pandas.Series) -> str:
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
    def _calculate_grade_sheba(row: pandas.Series) -> Union[str, int]:
        try:
            column_name = 'Grade'
            column_score = re.findall(r'\d+(?:\.\d+)?', str(row[column_name]))
            if len(column_score) == 0:
                return 'NA'

            return int(float(column_score[0]))
        except Exception:
            return 'NA'

    @staticmethod
    def _calculate_tumor_type_sheba(row: pandas.Series) -> str:
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
    def _calculate_slide_barcode_prefix_sheba(row: pandas.Series) -> str:
        try:
            return row[WSITupletsGenerator.patient_barcode_column_name]
        except Exception:
            return 'NA'

    ###########
    ### ALL ###
    ###########
    @staticmethod
    def _calculate_slide_barcode_prefix(row: pandas.Series) -> str:
        try:
            dataset_id = row[WSITupletsGenerator.dataset_id_column_name]
            if dataset_id == 'TCGA':
                return row[WSITupletsGenerator.patient_barcode_column_name]
            elif dataset_id == 'ABCTB':
                return row[WSITupletsGenerator.file_column_name].replace('tif', 'ndpi')
            elif dataset_id.startswith('CARMEL'):
                return row[WSITupletsGenerator._slide_barcode_column_name_carmel][:-2]
            elif dataset_id == 'SHEBA':
                return row[WSITupletsGenerator.patient_barcode_column_name]
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
    def _extract_annotations(df: pandas.DataFrame, patient_barcode_column_name: str, calculate_slide_barcode_prefix: Callable, calculate_tumor_type: Callable, calculate_grade: Callable) -> pandas.DataFrame:
        df[WSITupletsGenerator.slide_barcode_prefix_column_name] = df.apply(lambda row: calculate_slide_barcode_prefix(row), axis=1)
        df[WSITupletsGenerator.tumor_type_column_name] = df.apply(lambda row: calculate_tumor_type(row), axis=1)
        df[WSITupletsGenerator.grade_column_name] = df.apply(lambda row: calculate_grade(row), axis=1)

        annotations = df[[
            patient_barcode_column_name,
            WSITupletsGenerator.slide_barcode_prefix_column_name,
            WSITupletsGenerator.grade_column_name,
            WSITupletsGenerator.tumor_type_column_name]]
        annotations = annotations.rename(columns={patient_barcode_column_name: WSITupletsGenerator.patient_barcode_column_name})

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
    def _add_slide_barcode_prefix(df: pandas.DataFrame) -> pandas.DataFrame:
        df[WSITupletsGenerator.slide_barcode_prefix_column_name] = df.apply(lambda row: WSITupletsGenerator._calculate_slide_barcode_prefix(row), axis=1)
        return df

    @staticmethod
    def _select_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        df = df[[
            WSITupletsGenerator.file_column_name,
            WSITupletsGenerator.patient_barcode_column_name,
            WSITupletsGenerator.dataset_id_column_name,
            WSITupletsGenerator.mpp_column_name,
            WSITupletsGenerator.total_tiles_column_name,
            WSITupletsGenerator.legitimate_tiles_column_name,
            # WSITupletsGenerator._tile_usage_column_name,
            WSITupletsGenerator.width_column_name,
            WSITupletsGenerator.height_column_name,
            WSITupletsGenerator.magnification_column_name,
            WSITupletsGenerator.er_status_column_name,
            WSITupletsGenerator.pr_status_column_name,
            WSITupletsGenerator.her2_status_column_name,
            WSITupletsGenerator.grade_column_name,
            WSITupletsGenerator.tumor_type_column_name,
            WSITupletsGenerator.fold_column_name
        ]]
        return df

    @staticmethod
    def _standardize_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        # fix folds
        pandas.options.mode.chained_assignment = None
        df = df[~df[WSITupletsGenerator.fold_column_name].isin(WSITupletsGenerator._invalid_values)]
        folds = list(df[WSITupletsGenerator.fold_column_name].unique())
        numeric_folds = [common_utils.to_int(fold) for fold in folds]
        # try:
        max_val = numpy.max(numeric_folds) + 1
        df.loc[df[WSITupletsGenerator.fold_column_name] == 'test', WSITupletsGenerator.fold_column_name] = max_val
        df[WSITupletsGenerator.fold_column_name] = df[WSITupletsGenerator.fold_column_name].astype(int)
        # except Exception:
        #     print(folds)
        #     print(numeric_folds)

        # remove invalid values
        df = df.replace(WSITupletsGenerator._invalid_values, WSITupletsGenerator._invalid_value)
        df = df.dropna()
        return df

    # @staticmethod
    # def _add_folds_to_metadata(metadata_df, folds_count):
    #     folds = numpy.random.randint(folds_count, size=metadata_df.shape[0])
    #     metadata_df[WSITupletsGenerator.fold_column_name] = folds
    #     return metadata_df

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

    # @staticmethod
    # def _create_tile_bitmap(original_tile_size, tile_locations, plot_bitmap=False):
    #     indices = (numpy.array(tile_locations) / original_tile_size).astype(int)
    #     dim1_size = indices[:, 0].max() + 1
    #     dim2_size = indices[:, 1].max() + 1
    #     bitmap = numpy.zeros([dim1_size, dim2_size]).astype(int)
    #
    #     for (x, y) in indices:
    #         bitmap[x, y] = 1
    #
    #     tile_bitmap = numpy.uint8(Image.fromarray((bitmap * 255).astype(numpy.uint8)))
    #
    #     if plot_bitmap is True:
    #         plt.imshow(tile_bitmap, cmap='gray')
    #         plt.show()
    #
    #     return tile_bitmap
    #
    # @staticmethod
    # def _create_connected_components(tile_bitmap):
    #     components_count, components_labels, components_stats, _ = cv2.connectedComponentsWithStats(tile_bitmap)
    #     components = []
    #
    #     # if components_count == 1:
    #     #     continue
    #
    #     for component_id in range(1, components_count):
    #         current_bitmap = (components_labels == component_id)
    #         component_indices = numpy.where(current_bitmap)
    #         component_size = numpy.count_nonzero(current_bitmap)
    #         top_left = numpy.array([numpy.min(component_indices[0]), numpy.min(component_indices[1])])
    #         bottom_right = numpy.array([numpy.max(component_indices[0]), numpy.max(component_indices[1])])
    #         tile_indices = numpy.array([component_indices[0], component_indices[1]]).transpose()
    #
    #         components.append({
    #             'tiles_bitmap': current_bitmap.astype(int),
    #             'component_size': component_size,
    #             'top_left': top_left,
    #             'bottom_right': bottom_right,
    #             'tile_indices': tile_indices
    #         })
    #
    #     components_sorted = sorted(components, key=lambda item: item['component_size'], reverse=True)
    #     largest_component = components_sorted[0]
    #     largest_component_aspect_ratio = common_utils.calculate_box_aspect_ratio(largest_component['top_left'], largest_component['bottom_right'])
    #     largest_component_size = largest_component['component_size']
    #     valid_components = [largest_component]
    #     for component in components_sorted[1:]:
    #         current_aspect_ratio = common_utils.calculate_box_aspect_ratio(component['top_left'], component['bottom_right'])
    #         current_component_size = component['component_size']
    #         if numpy.abs(largest_component_aspect_ratio - current_aspect_ratio) < WSITupletsGenerator._max_aspect_ratio_diff and (current_component_size / largest_component_size) > WSITupletsGenerator._min_component_ratio:
    #             valid_components.append(component)
    #
    #     return valid_components

    # @staticmethod
    # def _get_random_connected_component(slide_descriptor: SlideDescriptor) -> int:
    #     component_index = int(numpy.random.randint(len(slide_descriptor.components), size=1))
    #     return slide
    #
    # @staticmethod
    # def _get_random_component_index(slide_descriptor: SlideDescriptor) -> int:
    #     component_index = int(numpy.random.randint(len(slide_descriptor.components), size=1))
    #     return component_index

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
    def _validate_histogram(patch: Patch) -> bool:
        patch_grayscale = patch.image.convert('L')
        hist, _ = numpy.histogram(a=patch_grayscale, bins=patch.slide_context.tile_size)
        white_ratio = numpy.sum(hist[WSITupletsGenerator._white_intensity_threshold:]) / (patch.slide_context.tile_size * patch.slide_context.tile_size)
        if white_ratio > WSITupletsGenerator._white_ratio_threshold:
            return False
        return True

    def _build_column_names(self):
        column_names = {}
        for dataset_id_prefix in WSITupletsGenerator._dataset_id_prefixes:
            column_names[dataset_id_prefix] = {}
            column_names[dataset_id_prefix][WSITupletsGenerator._file_column_name_shared] = WSITupletsGenerator.file_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._patient_barcode_column_name_shared] = WSITupletsGenerator.patient_barcode_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._dataset_id_column_name_shared] = WSITupletsGenerator.dataset_id_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._mpp_column_name_shared] = WSITupletsGenerator.mpp_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._scan_date_column_name_shared] = WSITupletsGenerator.scan_date_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._width_column_name_shared] = WSITupletsGenerator.width_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._height_column_name_shared] = WSITupletsGenerator.height_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._magnification_column_name_shared] = WSITupletsGenerator.magnification_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._er_status_column_name_shared] = WSITupletsGenerator.er_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._pr_status_column_name_shared] = WSITupletsGenerator.pr_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._her2_status_column_name_shared] = WSITupletsGenerator.her2_status_column_name
            column_names[dataset_id_prefix][WSITupletsGenerator._fold_column_name_shared] = WSITupletsGenerator.fold_column_name
            column_names[dataset_id_prefix][self._get_total_tiles_column_name()] = WSITupletsGenerator.total_tiles_column_name
            column_names[dataset_id_prefix][self._get_legitimate_tiles_column_name()] = WSITupletsGenerator.legitimate_tiles_column_name
            column_names[dataset_id_prefix][self._get_slide_tile_usage_column_name(dataset_id_prefix=dataset_id_prefix)] = WSITupletsGenerator.tile_usage_column_name

            if dataset_id_prefix.startswith(WSITupletsGenerator._dataset_id_sheba):
                column_names[dataset_id_prefix][WSITupletsGenerator._er_status_column_name_sheba] = WSITupletsGenerator.er_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._pr_status_column_name_sheba] = WSITupletsGenerator.pr_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._her2_status_column_name_sheba] = WSITupletsGenerator.her2_status_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._grade_column_name_sheba] = WSITupletsGenerator.grade_column_name
                column_names[dataset_id_prefix][WSITupletsGenerator._tumor_type_column_name_sheba] = WSITupletsGenerator.tumor_type_column_name

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
            self._logger.info(f'Processing metadata for {dataset_id}...')
            slide_metadata_file = os.path.join(self._dataset_paths[dataset_id], WSITupletsGenerator._get_slides_data_file_name(dataset_id=dataset_id))
            grid_metadata_file = os.path.join(self._dataset_paths[dataset_id], WSITupletsGenerator._get_grids_folder_name(desired_magnification=self._desired_magnification), WSITupletsGenerator._grid_data_file_name)
            slide_df = pandas.read_excel(io=slide_metadata_file)
            grid_df = pandas.read_excel(io=grid_metadata_file)
            current_df = pandas.DataFrame({**slide_df.set_index(keys=WSITupletsGenerator.file_column_name).to_dict(), **grid_df.set_index(keys=WSITupletsGenerator.file_column_name).to_dict()})
            current_df.reset_index(inplace=True)
            current_df.rename(columns={'index': WSITupletsGenerator.file_column_name}, inplace=True)

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

            if df is None:
                df = current_df
            else:
                df = pandas.concat((df, current_df))

            self._logger.info(f'Processing metadata for {dataset_id}... Done.')
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
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator.patient_barcode_column_name, WSITupletsGenerator.slide_barcode_prefix_column_name])
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
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator.patient_barcode_column_name, WSITupletsGenerator.slide_barcode_prefix_column_name])
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
        df = pandas.merge(left=df, right=enhanced_metadata, on=[WSITupletsGenerator.patient_barcode_column_name, WSITupletsGenerator.slide_barcode_prefix_column_name])
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
        indices_of_slides_without_grid = set(df.index[df[WSITupletsGenerator.total_tiles_column_name] == -1])
        indices_of_slides_with_few_tiles = set(df.index[df[WSITupletsGenerator.legitimate_tiles_column_name] < self._minimal_tiles_count])

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_without_grid - indices_of_slides_with_few_tiles))
        return df.iloc[valid_slide_indices]

    def _select_folds_from_metadata(self, df, folds):
        return df[df[WSITupletsGenerator.fold_column_name].isin(folds)]

    def _open_slide(self, image_file_path, desired_downsample):
        slide = openslide.open_slide(image_file_path)
        level, adjusted_tile_size = slide_utils.get_best_level_for_downsample(slide=slide, desired_downsample=desired_downsample, tile_size=self._tile_size)

        return {
            'slide': slide,
            'level': level,
            'adjusted_tile_size': adjusted_tile_size
        }

    # def _calculate_inner_radius_pixels(self, mpp: float):
    #     inner_radius_pixels = int(self._inner_radius_mm / (mpp / 1000))
    #     return inner_radius_pixels

    # def _create_random_example(self, slide_descriptor: Slide, component_index: int):
    #     # image_file_path = slide_descriptor['image_file_path']
    #     # desired_downsample = slide_descriptor['desired_downsample']
    #     # original_tile_size = slide_descriptor['original_tile_size']
    #     # components = slide_descriptor['components']
    #     # slide_data = self._open_slide(image_file_path=image_file_path, desired_downsample=desired_downsample)
    #
    #
    #     # tile_indices = component['tile_indices']
    #     # tiles_bitmap = component['tiles_bitmap']
    #     # slide = slide_data['slide']
    #     # adjusted_tile_size = slide_data['adjusted_tile_size']
    #     # level = slide_data['level']
    #
    #     attempts = 0
    #     component = slide_descriptor.get_component(component_index=component_index)
    #     while True:
    #         if attempts == WSITupletsGenerator._max_attempts:
    #             break
    #
    #         index = int(numpy.random.randint(tile_indices.shape[0], size=1))
    #         random_tile_indices = tile_indices[index, :]
    #         location = random_tile_indices * original_tile_size
    #         point_offset = (original_tile_size * numpy.random.uniform(size=2)).astype(int)
    #         point = point_offset + location
    #         bitmap_indices = WSITupletsGenerator._calculate_bitmap_indices(point=point, tile_size=original_tile_size)
    #         if WSITupletsGenerator._validate_location(bitmap=tiles_bitmap, indices=bitmap_indices) is False:
    #             attempts = attempts + 1
    #             continue
    #
    #         tile = slide_utils.read_region_around_point(slide=slide, point=point, tile_size=self._tile_size, selected_level_tile_size=adjusted_tile_size, level=level)
    #         tile_grayscale = tile.convert('L')
    #         hist, _ = numpy.histogram(tile_grayscale, bins=self._tile_size)
    #         white_ratio = numpy.sum(hist[WSITupletsGenerator._histogram_min_intensity_level:]) / (self._tile_size * self._tile_size)
    #         if white_ratio > WSITupletsGenerator._white_ratio_threshold:
    #             attempts = attempts + 1
    #             continue
    #
    #         return {
    #             'tile': numpy.array(tile),
    #             'point': point
    #         }
    #
    #     return None

    # def _create_anchor_example(self, slide_descriptor, component_index):
    #     return self._create_random_example(slide_descriptor=slide_descriptor, component_index=component_index)
    #
    # def _create_positive_example(self, slide_descriptor: Slide, component_index: int, anchor_point: numpy.ndarray):
    #     # image_file_path = slide_descriptor['image_file_path']
    #     # desired_downsample = slide_descriptor['desired_downsample']
    #     # original_tile_size = slide_descriptor['original_tile_size']
    #     # image_file_name_suffix = slide_descriptor['image_file_name_suffix']
    #     # components = slide_descriptor['components']
    #     # mpp = slide_descriptor['mpp']
    #     # slide_data = self._open_slide(image_file_path=image_file_path, desired_downsample=desired_downsample)
    #     component = slide_descriptor.get_component(component_index=component_index)
    #     inner_radius_pixels = self._calculate_inner_radius_pixels(mpp=slide_descriptor.mpp)
    #     # slide = slide_data['slide']
    #     # adjusted_tile_size = slide_data['adjusted_tile_size']
    #     # level = slide_data['level']
    #     # tiles_bitmap = component['tiles_bitmap']
    #
    #     attempts = 0
    #     while True:
    #         if attempts == WSITupletsGenerator._max_attempts:
    #             break
    #
    #         positive_angle = 2 * numpy.pi * numpy.random.uniform(size=1)[0]
    #         positive_dir = numpy.array([numpy.cos(positive_angle), numpy.sin(positive_angle)])
    #         # positive_radius = inner_radius_pixels * numpy.random.uniform(size=1)[0]
    #         positive_radius = inner_radius_pixels
    #         positive_point = (anchor_point + positive_radius * positive_dir).astype(int)
    #         positive_bitmap_indices = WSITupletsGenerator._calculate_bitmap_indices(point=positive_point, tile_size=original_tile_size)
    #         if WSITupletsGenerator._validate_location(bitmap=component.tiles_bitmap, indices=positive_bitmap_indices) is False:
    #             attempts = attempts + 1
    #             continue
    #
    #         positive_tile = slide_utils.read_region_around_point(
    #             slide=slide,
    #             point=positive_point,
    #             tile_size=self._tile_size,
    #             selected_level_tile_size=adjusted_tile_size,
    #             level=level)
    #
    #         return {
    #             'tile': numpy.array(positive_tile),
    #             'point': positive_point
    #         }
    #
    #     return None

    def _create_negative_example(self, df, slide_descriptor):
        row_anchor_patient = self._df.loc[self._df[WSITupletsGenerator.file_column_name] == slide_descriptor['image_file_name']].iloc[0]
        patient_barcode = row_anchor_patient[WSITupletsGenerator.patient_barcode_column_name]
        er_status = row_anchor_patient[WSITupletsGenerator.er_status_column_name]
        pr_status = row_anchor_patient[WSITupletsGenerator.pr_status_column_name]
        her2_status = row_anchor_patient[WSITupletsGenerator.her2_status_column_name]
        tumor_type = row_anchor_patient[WSITupletsGenerator.her2_status_column_name]
        grade = row_anchor_patient[WSITupletsGenerator.grade_column_name]
        filtered_df = df[(df[WSITupletsGenerator.patient_barcode_column_name] != patient_barcode) &
                         ((df[WSITupletsGenerator.pr_status_column_name] != pr_status) |
                          (df[WSITupletsGenerator.er_status_column_name] != er_status) |
                          (df[WSITupletsGenerator.her2_status_column_name] != her2_status) |
                          (df[WSITupletsGenerator.tumor_type_column_name] != tumor_type) |
                          (df[WSITupletsGenerator.grade_column_name] != grade))]

        index = int(numpy.random.randint(filtered_df.shape[0], size=1))
        row_negative_patient = filtered_df.iloc[index]
        image_file_name_negative_patient = row_negative_patient[WSITupletsGenerator.file_column_name]
        slide_descriptor_negative_patient = self._image_file_name_to_slide_descriptor[image_file_name_negative_patient]
        component_index = WSITupletsGenerator._get_random_component_index(slide_descriptor=slide_descriptor_negative_patient)
        return self._create_random_example(slide_descriptor=slide_descriptor_negative_patient, component_index=component_index)

    # def _create_tile_locations(self, dataset_id, image_file_name_stem):
    #     grid_file_path = os.path.normpath(os.path.join(self._dataset_paths[dataset_id], f'Grids_{self._desired_magnification}', f'{image_file_name_stem}--tlsz{self._tile_size}.data'))
    #     with open(grid_file_path, 'rb') as file_handle:
    #         locations = numpy.array(pickle.load(file_handle))
    #         locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()
    #
    #     return locations

    def _create_slide_descriptor(self, row):
        dataset_id = row[WSITupletsGenerator.dataset_id_column_name]
        dataset_path = self._dataset_paths[dataset_id]
        slide_descriptor = Slide(row=row, dataset_path=dataset_path, desired_magnification=self._desired_magnification, tile_size=self._tile_size)
        return slide_descriptor

    def _slide_descriptors_creation_worker(self, df, indices, q):
        df = df.iloc[indices]
        for (index, row) in df.iterrows():
            slide_descriptor = self._create_slide_descriptor(row=row)
            q.put(slide_descriptor)

    def _tuples_creation_worker(self, df, tuplet_indices, slide_indices, negative_examples_count, q):
        tuplets_created = 0
        for i in itertools.count():
            if tuplet_indices is not None and tuplets_created == len(tuplet_indices):
                break

            slide_descriptor_index = slide_indices[i % len(slide_indices)]
            slide_descriptor = self._slide_descriptors[slide_descriptor_index]
            tuplet = self._try_create_tuplet(df=df, slide=slide_descriptor, negative_examples_count=negative_examples_count)
            if tuplet is not None:
                q.put(tuplet)
                tuplets_created = tuplets_created + 1

    def _try_create_tuplet(self, df: pandas.DataFrame, slide: Slide, negative_examples_count: int):
        patches = []
        patch_extractor = PatchExtractor(slide=slide, inner_radius_mm=self._inner_radius_mm)

        anchor_patch = patch_extractor.extract_patch(patch_validators=[WSITupletsGenerator._validate_histogram])
        patches.append(numpy.array(anchor_patch.image))

        positive_patch = patch_extractor.extract_patch(patch_validators=[], reference_patch=anchor_patch)
        patches.append(numpy.array(positive_patch.image))

        # for i in range(negative_examples_count):
        #     pass

        patches_tuplet = numpy.transpose(numpy.stack(patches), (0, 3, 1, 2))
        return patches_tuplet

    def _create_slides(self, df, num_workers):
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

    def _queue_tuplets(self, tuplets_count, queue_size, negative_examples_count, folds, workers_count):
        self._tuplets_queue = Queue(maxsize=queue_size)
        self._slide_descriptors = self._create_slides(df=self._df, num_workers=workers_count)
        self._image_file_name_to_slide_descriptor = dict((slide_descriptor.image_file_name, slide_descriptor) for slide_descriptor in self._slide_descriptors)
        df = self._select_folds_from_metadata(df=self._df, folds=folds)

        tuplet_indices_groups = None
        if tuplets_count < numpy.inf:
            tuplet_indices = list(range(tuplets_count))
            tuplet_indices_groups = common_utils.split(items=tuplet_indices, n=workers_count)

        slide_indices = list(range(len(self._slide_descriptors)))
        slide_indices_groups = common_utils.split(items=slide_indices, n=workers_count)

        args = [(df, tuplet_indices_groups[i] if tuplet_indices_groups is not None else None, slide_indices_groups[i], negative_examples_count, self._tuplets_queue) for i in range(workers_count)]
        self._tuplets_workers = WSITupletsGenerator._start_workers(args=args, f=self._tuples_creation_worker, workers_count=workers_count)

        # WSITuplesGenerator._drain_queue_to_disk(q=q, count=tuples_count, dir_path=dir_path, file_name_stem='tuple')
        # WSITuplesGenerator._join_workers(workers=workers)
        # WSITuplesGenerator._stop_workers(workers=workers)
        # q.close()

    def start_tuplets_creation(self, queue_size, negative_examples_count, folds, workers_count):
        self._queue_tuplets(
            tuplets_count=numpy.inf,
            queue_size=queue_size,
            negative_examples_count=negative_examples_count,
            folds=folds,
            workers_count=workers_count)

        self._tuplets = WSITupletsGenerator._drain_queue(q=self._tuplets_queue, count=self._dataset_size)

    def stop_tuplets_creation(self):
        # WSITupletsGenerator._join_workers(workers=self._tuplets_workers)
        WSITupletsGenerator._stop_workers(workers=self._tuplets_workers)
        self._tuplets_queue.close()

    def save_tuplets(self, tuplets_count, negative_examples_count, folds, workers_count, dir_path):
        self._queue_tuplets(
            tuplets_count=tuplets_count,
            queue_size=tuplets_count,
            negative_examples_count=negative_examples_count,
            folds=folds,
            workers_count=workers_count)

        WSITupletsGenerator._drain_queue_to_disk(q=self._tuplets_queue, count=tuplets_count, dir_path=dir_path, file_name_stem='tuple')

    def save_metadata(self, output_file_path):
        self._df.to_excel(output_file_path)

    def get_tuplet(self, index, replace=True):
        mod_index = numpy.mod(index, self._dataset_size)
        tuplet = self._tuplets[mod_index]

        # if replace is True:
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
            inner_radius_mm,
            outer_radius_mm,
            tile_size,
            desired_magnification,
            dataset_size,
            datasets_base_dir_path,
            dump_dir_path,
            dataset_ids,
            minimal_tiles_count,
            metadata_enhancement_dir_path):
        self._logger = logging.getLogger(name=self.__class__.__name__)
        self._inner_radius_mm = inner_radius_mm
        self._outer_radius_mm = outer_radius_mm
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._dataset_paths = WSITupletsGenerator._get_dataset_paths(
            dataset_ids=dataset_ids,
            datasets_base_dir_path=datasets_base_dir_path)
        self._minimal_tiles_count = minimal_tiles_count
        self._metadata_enhancement_dir_path = metadata_enhancement_dir_path
        self._dump_dir_path = dump_dir_path
        self._dataset_size = dataset_size
        self._slide_descriptors = []
        self._image_file_name_to_slide_descriptor = {}
        self._tuplets_queue = None
        self._tuplets_workers = None
        self._tuplets = None
        self._df = self._load_metadata()
        # self._df = self._df.iloc[[10]]

