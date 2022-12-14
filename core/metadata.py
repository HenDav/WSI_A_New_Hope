# python core
from __future__ import annotations
import os
import math
import re
from typing import List, Dict, Union, Optional, Callable, cast
from pathlib import Path
from abc import ABC, abstractmethod
import json
import itertools

# pandas
import pandas

# numpy
import numpy

# openslide
OPENSLIDE_PATH = r'C:\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        pass
else:
    pass

# wsi
from core import constants
from core import utils
from core.base import SeedableObject, OutputObject
from core.wsi import SlideContext, Slide, Tile

# tap
from tap import Tap

# gipmed
from core.parallel_processing import TaskParallelProcessor, ParallelProcessorTask


# =================================================
# MetadataGenerator Class
# =================================================
class MetadataBase(ABC):
    def __init__(
            self,
            datasets_base_dir_path: Path,
            tile_size: int,
            desired_magnification: int,
            **kw: object):
        super(MetadataBase, self).__init__(**kw)
        self._datasets_base_dir_path = datasets_base_dir_path
        self._tile_size = tile_size
        self._desired_magnification = desired_magnification
        self._dataset_paths = constants.get_dataset_paths(datasets_base_dir_path=datasets_base_dir_path)
        self._df = self._load_metadata()

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    @abstractmethod
    def _load_metadata(self) -> pandas.DataFrame:
        pass


# =================================================
# MetadataReader Class
# =================================================
# class MetadataReader:
#     def __init__(
#             self,
#             metadata_file_path: Path,
#             **kw: object):
#         self._metadata_file_path = metadata_file_path
#         super(MetadataReader, self).__init__(**kw)
#
#     def _load_metadata(self) -> pandas.DataFrame:
#         return pandas.read_csv(filepath_or_buffer=self._metadata_file_path)


# =================================================
# MetadataGenerator Class
# =================================================
class MetadataGenerator(OutputObject, MetadataBase):
    def __init__(
            self,
            datasets_base_dir_path: Path,
            tile_size: int,
            desired_magnification: int,
            metadata_file_path: Path,
            metadata_enhancement_dir_path: Path,
            log_file_path: Path,
            dataset_ids: List[str],
            minimal_tiles_count: int):
        self._metadata_file_path = metadata_file_path
        self._metadata_enhancement_dir_path = metadata_enhancement_dir_path
        self._log_file_path = log_file_path
        self._dataset_ids = dataset_ids
        self._minimal_tiles_count = minimal_tiles_count
        super(MetadataGenerator, self).__init__(log_file_path=log_file_path, datasets_base_dir_path=datasets_base_dir_path, tile_size=tile_size, desired_magnification=desired_magnification)

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    def save_metadata(self):
        self._metadata_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(path_or_buf=self._metadata_file_path)

    def _build_column_names(self):
        column_names = {}
        for dataset_id_prefix in constants.dataset_ids:
            column_names[dataset_id_prefix] = {}
            column_names[dataset_id_prefix][constants.file_column_name_shared] = constants.file_column_name
            column_names[dataset_id_prefix][constants.patient_barcode_column_name_shared] = constants.patient_barcode_column_name
            column_names[dataset_id_prefix][constants.dataset_id_column_name_shared] = constants.dataset_id_column_name
            column_names[dataset_id_prefix][constants.mpp_column_name_shared] = constants.mpp_column_name
            column_names[dataset_id_prefix][constants.scan_date_column_name_shared] = constants.scan_date_column_name
            column_names[dataset_id_prefix][constants.width_column_name_shared] = constants.width_column_name
            column_names[dataset_id_prefix][constants.height_column_name_shared] = constants.height_column_name
            column_names[dataset_id_prefix][constants.magnification_column_name_shared] = constants.magnification_column_name
            column_names[dataset_id_prefix][constants.er_status_column_name_shared] = constants.er_status_column_name
            column_names[dataset_id_prefix][constants.pr_status_column_name_shared] = constants.pr_status_column_name
            column_names[dataset_id_prefix][constants.her2_status_column_name_shared] = constants.her2_status_column_name
            column_names[dataset_id_prefix][constants.fold_column_name_shared] = constants.fold_column_name
            column_names[dataset_id_prefix][self._get_total_tiles_column_name()] = constants.total_tiles_column_name
            column_names[dataset_id_prefix][self._get_legitimate_tiles_column_name()] = constants.legitimate_tiles_column_name
            column_names[dataset_id_prefix][self._get_slide_tile_usage_column_name(dataset_id_prefix=dataset_id_prefix)] = constants.tile_usage_column_name

            if dataset_id_prefix.startswith(constants.dataset_id_sheba):
                column_names[dataset_id_prefix][constants.er_status_column_name_sheba] = constants.er_status_column_name
                column_names[dataset_id_prefix][constants.pr_status_column_name_sheba] = constants.pr_status_column_name
                column_names[dataset_id_prefix][constants.her2_status_column_name_sheba] = constants.her2_status_column_name
                column_names[dataset_id_prefix][constants.grade_column_name_sheba] = constants.grade_column_name
                column_names[dataset_id_prefix][constants.tumor_type_column_name_sheba] = constants.tumor_type_column_name

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
        padding = 40
        self._logger.info(msg=utils.generate_title_text(text=f'Metadata Generator Configuration'))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='datasets_base_dir_path', value=self._datasets_base_dir_path, indentation=1, padding=padding))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='metadata_enhancement_dir_path', value=self._metadata_enhancement_dir_path, indentation=1, padding=padding))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='log_file_path', value=self._log_file_path, indentation=1, padding=padding))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='tile_size', value=self._tile_size, indentation=1, padding=padding))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='desired_magnification', value=self._desired_magnification, indentation=1, padding=padding))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='dataset_ids', value=self._dataset_ids, indentation=1, padding=padding))

        dataset_paths_str = (f := lambda d: {k: f(v) for k, v in d.items()} if type(d) == dict else str(d))(self._dataset_paths)
        dataset_paths_str_dump = json.dumps(dataset_paths_str, indent=8)
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='dataset_paths', value=dataset_paths_str_dump, indentation=1, padding=padding, newline=True))

        self._logger.info(msg=utils.generate_captioned_bullet_text(text='minimal_tiles_count', value=self._minimal_tiles_count, indentation=1, padding=padding))

        self._logger.info(msg='')
        self._logger.info(msg=utils.generate_title_text(text=f'Metadata Processing'))
        df = None
        for _, dataset_id in enumerate(self._dataset_ids):
            self._logger.info(msg=utils.generate_captioned_bullet_text(text='Processing Metadata For', value=dataset_id, indentation=1, padding=padding))
            slide_metadata_file = os.path.join(self._dataset_paths[dataset_id], MetadataGenerator._get_slides_data_file_name(dataset_id=dataset_id))
            grid_metadata_file = os.path.join(self._dataset_paths[dataset_id], MetadataGenerator._get_grids_folder_name(desired_magnification=self._desired_magnification), constants.grid_data_file_name)
            slide_df = pandas.read_excel(io=slide_metadata_file)
            grid_df = pandas.read_excel(io=grid_metadata_file)
            current_df = pandas.DataFrame({**slide_df.set_index(keys=constants.file_column_name).to_dict(), **grid_df.set_index(keys=constants.file_column_name).to_dict()})
            current_df.reset_index(inplace=True)
            current_df.rename(columns={'index': constants.file_column_name}, inplace=True)

            current_df = self._prevalidate_metadata(
                df=current_df)

            current_df = self._rename_metadata(
                df=current_df,
                dataset_id_prefix=MetadataGenerator._get_dataset_id_prefix(dataset_id=dataset_id))

            current_df = self._enhance_metadata(
                df=current_df,
                dataset_id=dataset_id)

            current_df = MetadataGenerator._select_metadata(
                df=current_df)

            current_df = MetadataGenerator._standardize_metadata(
                df=current_df)

            current_df = self._postvalidate_metadata(
                df=current_df)

            if df is None:
                df = current_df
            else:
                df = pandas.concat((df, current_df))

        return df

    def _enhance_metadata_tcga(self, df: pandas.DataFrame):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)

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

        annotations_tcga = MetadataGenerator._extract_annotations(
            df=cell_genomics_tcga_file2_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_tcga,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_tcga,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_tcga,
            calculate_grade=MetadataGenerator._calculate_grade_tcga)

        enhanced_metadata = pandas.concat([annotations_tcga])
        df = pandas.merge(left=df, right=enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name])
        return df

    def _enhance_metadata_carmel(self, df):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)

        carmel_annotations_Batch11_26_10_21_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'Carmel', 'Carmel_annotations_Batch11_26-10-21.xlsx')))

        carmel_annotations_26_10_2021_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'Carmel', 'Carmel_annotations_26-10-2021.xlsx')))

        annotations1_carmel = MetadataGenerator._extract_annotations(
            df=carmel_annotations_Batch11_26_10_21_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_carmel,
            calculate_grade=MetadataGenerator._calculate_grade_carmel)

        annotations2_carmel = MetadataGenerator._extract_annotations(
            df=carmel_annotations_26_10_2021_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_carmel,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_carmel,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_carmel,
            calculate_grade=MetadataGenerator._calculate_grade_carmel)

        enhanced_metadata = pandas.concat([annotations1_carmel, annotations2_carmel])
        # try:
        df = pandas.merge(left=df, right=enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name])
        # except Exception:
        #     h = 5
        return df

    def _enhance_metadata_abctb(self, df):
        df = MetadataGenerator._add_slide_barcode_prefix(df=df)

        abctb_path_data_df = pandas.read_excel(
            io=os.path.normpath(os.path.join(self._metadata_enhancement_dir_path, 'ABCTB', 'ABCTB_Path_Data.xlsx')))

        annotations_abctb = MetadataGenerator._extract_annotations(
            df=abctb_path_data_df,
            patient_barcode_column_name=constants.patient_barcode_column_name_enhancement_abctb,
            calculate_slide_barcode_prefix=MetadataGenerator._calculate_slide_barcode_prefix_abctb,
            calculate_tumor_type=MetadataGenerator._calculate_tumor_type_abctb,
            calculate_grade=MetadataGenerator._calculate_grade_abctb)

        enhanced_metadata = pandas.concat([annotations_abctb])
        df = pandas.merge(left=df, right=enhanced_metadata, on=[constants.patient_barcode_column_name, constants.slide_barcode_prefix_column_name])
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
        if constants.bad_segmentation_column_name in df.columns:
            indices_of_slides_with_bad_seg = set(df.index[df[constants.bad_segmentation_column_name] == 1])
        else:
            indices_of_slides_with_bad_seg = set()

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_with_bad_seg))
        return df.iloc[valid_slide_indices]

    def _postvalidate_metadata(self, df):
        indices_of_slides_without_grid = set(df.index[df[constants.total_tiles_column_name] == -1])
        indices_of_slides_with_few_tiles = set(df.index[df[constants.legitimate_tiles_column_name] < self._minimal_tiles_count])

        all_indices = set(numpy.array(range(df.shape[0])))
        valid_slide_indices = numpy.array(list(all_indices - indices_of_slides_without_grid - indices_of_slides_with_few_tiles))
        return df.iloc[valid_slide_indices]

    @staticmethod
    def _build_path_suffixes() -> Dict:
        path_suffixes = {
            constants.dataset_id_tcga: f'Breast/{constants.dataset_id_tcga}',
            constants.dataset_id_abctb: f'Breast/{constants.dataset_id_abctb}_TIF',
        }

        for i in range(1, 12):
            path_suffixes[f'{constants.dataset_id_carmel}{i}'] = f'Breast/{constants.dataset_id_carmel.capitalize()}/Batch_{i}/{constants.dataset_id_carmel}{i}'

        for i in range(2, 7):
            path_suffixes[f'{constants.dataset_id_sheba}{i}'] = f'Breast/{constants.dataset_id_sheba.capitalize()}/Batch_{i}/{constants.dataset_id_sheba}{i}'

        return path_suffixes

    @staticmethod
    def _get_slides_data_file_name(dataset_id: str) -> str:
        return f'slides_data_{dataset_id}.xlsx'

    @staticmethod
    def _get_grids_folder_name(desired_magnification: int) -> str:
        return f'Grids_{desired_magnification}'

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
            return row[constants.patient_barcode_column_name_enhancement_tcga]
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
            return row[constants.file_column_name_enhancement_abctb]
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
            slide_barcode = row[constants.slide_barcode_column_name_enhancement_carmel]
            block_id = row[constants.block_id_column_name_enhancement_carmel]
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
            return row[constants.patient_barcode_column_name]
        except Exception:
            return 'NA'

    ###########
    ### ALL ###
    ###########
    @staticmethod
    def _calculate_slide_barcode_prefix(row: pandas.Series) -> str:
        try:
            dataset_id = row[constants.dataset_id_column_name]
            if dataset_id == 'TCGA':
                return row[constants.patient_barcode_column_name]
            elif dataset_id == 'ABCTB':
                return row[constants.file_column_name].replace('tif', 'ndpi')
            elif dataset_id.startswith('CARMEL'):
                return row[constants.slide_barcode_column_name_carmel][:-2]
            elif dataset_id == 'SHEBA':
                return row[constants.patient_barcode_column_name]
        except Exception:
            return 'NA'

    @staticmethod
    def _extract_annotations(df: pandas.DataFrame, patient_barcode_column_name: str, calculate_slide_barcode_prefix: Callable, calculate_tumor_type: Callable, calculate_grade: Callable) -> pandas.DataFrame:
        df[constants.slide_barcode_prefix_column_name] = df.apply(lambda row: calculate_slide_barcode_prefix(row), axis=1)
        df[constants.tumor_type_column_name] = df.apply(lambda row: calculate_tumor_type(row), axis=1)
        df[constants.grade_column_name] = df.apply(lambda row: calculate_grade(row), axis=1)

        annotations = df[[
            patient_barcode_column_name,
            constants.slide_barcode_prefix_column_name,
            constants.grade_column_name,
            constants.tumor_type_column_name]]
        annotations = annotations.rename(columns={patient_barcode_column_name: constants.patient_barcode_column_name})

        return annotations

    @staticmethod
    def _add_slide_barcode_prefix(df: pandas.DataFrame) -> pandas.DataFrame:
        df[constants.slide_barcode_prefix_column_name] = df.apply(lambda row: MetadataGenerator._calculate_slide_barcode_prefix(row), axis=1)
        return df

    @staticmethod
    def _select_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        df = df[[
            constants.file_column_name,
            constants.patient_barcode_column_name,
            constants.dataset_id_column_name,
            constants.mpp_column_name,
            constants.total_tiles_column_name,
            constants.legitimate_tiles_column_name,
            constants.width_column_name,
            constants.height_column_name,
            constants.magnification_column_name,
            constants.er_status_column_name,
            constants.pr_status_column_name,
            constants.her2_status_column_name,
            constants.grade_column_name,
            constants.tumor_type_column_name,
            constants.fold_column_name
        ]]
        return df

    @staticmethod
    def _standardize_metadata(df: pandas.DataFrame) -> pandas.DataFrame:
        pandas.options.mode.chained_assignment = None
        df = df[~df[constants.fold_column_name].isin(constants.invalid_values)]
        folds = list(df[constants.fold_column_name].unique())
        numeric_folds = [utils.to_int(fold) for fold in folds]
        max_val = numpy.max(numeric_folds) + 1
        df.loc[df[constants.fold_column_name] == 'test', constants.fold_column_name] = max_val
        df[constants.fold_column_name] = df[constants.fold_column_name].astype(int)
        df = df.replace(constants.invalid_values, constants.invalid_value)
        df = df.dropna()
        return df


# =================================================
# MetadataManagerTask Class
# =================================================
# class MetadataManagerTask(ParallelProcessorTask):
#     def __init__(self, row_index: int, metadata: pandas.DataFrame, dataset_paths: Dict[str, Path], desired_magnification: int, tile_size: int):
#         super().__init__()
#         self._row_index = row_index
#         self._metadata = metadata
#         self._dataset_paths = dataset_paths
#         self._desired_magnification = desired_magnification
#         self._tile_size = tile_size
#         self._slide_context = None
#
#     @property
#     def slide_context(self) -> Union[None, SlideContext]:
#         return self._slide_context
#
#     def process(self):
#         row = self._metadata.iloc[self._row_index]
#         dataset_id = row[constants.dataset_id_column_name]
#         dataset_path = self._dataset_paths[dataset_id]
#         self._slide_context = SlideContext(row=row, dataset_path=dataset_path, desired_magnification=self._desired_magnification, tile_size=self._tile_size)
#
#     def post_process(self):
#         pass


# =================================================
# SlideContextsManager Class
# =================================================
# class SlideContextsManager(OutputObject, SeedableObject, MetadataBase):
#     def __init__(
#             self,
#             name: str,
#             output_dir_path: Path,
#             datasets_base_dir_path: Path,
#             tile_size: int,
#             desired_magnification: int,
#             metadata_file_path: Path):
#         self._metadata_file_path = metadata_file_path
#         super().__init__(name=name, output_dir_path=output_dir_path, datasets_base_dir_path=datasets_base_dir_path, tile_size=tile_size, desired_magnification=desired_magnification)
#         self._slide_contexts = self._create_slide_contexts()
#         self._file_name_to_slide_context = self._create_file_name_to_slide_context()
#
#     @property
#     def slide_contexts_count(self) -> int:
#         return len(self._slide_contexts)
#
#     @property
#     def file_name_to_slide_context(self) -> Dict[str, SlideContext]:
#         return self._file_name_to_slide_context
#
#     def get_slide_context(self, index: int) -> SlideContext:
#         return self._slide_contexts[index]
#
#     def get_slide_context_by_file_name(self, file_name: str) -> SlideContext:
#         return self._file_name_to_slide_context[file_name]
#
#     def get_random_slide_context(self) -> SlideContext:
#         index = self._rng.integers(low=0, high=len(self._slide_contexts))
#         return self.get_slide_context(index=index)
#
#     def _load_metadata(self) -> pandas.DataFrame:
#         return pandas.read_csv(filepath_or_buffer=self._metadata_file_path)
#
#     def _create_slide_contexts(self) -> List[SlideContext]:
#         self._logger.info(msg=utils.generate_title_text(text=f'SlideContexts Creation'))
#         slide_contexts = []
#         for row_index in range(self._df.shape[0]):
#             slide_context = SlideContext(row_index=row_index, metadata=self._df, dataset_paths=self._dataset_paths, tile_size=self._tile_size, desired_magnification=self._desired_magnification)
#             slide_contexts.append(slide_context)
#             self._logger.info(msg=utils.generate_captioned_bullet_text(text='Creating SlideContext', value=slide_context.image_file_name, indentation=1, padding=30))
#         return slide_contexts
#
#     def _create_file_name_to_slide_context(self) -> Dict[str, SlideContext]:
#         file_name_to_slide_context = {}
#         for slide_context in self._slide_contexts:
#             file_name_to_slide_context[slide_context.image_file_name] = slide_context
#
#         return file_name_to_slide_context


# =================================================
# SlidesManagerTask Class
# =================================================
class SlidesManagerTask(ParallelProcessorTask):
    def __init__(self, row_index: int, metadata: pandas.DataFrame, dataset_paths: Dict[str, Path], desired_magnification: int, tile_size: int):
        super().__init__()
        self._row_index = row_index
        self._metadata = metadata
        self._dataset_paths = dataset_paths
        self._desired_magnification = desired_magnification
        self._tile_size = tile_size
        self._slide = None

    @property
    def slide(self) -> Union[None, Slide]:
        return self._slide

    def process(self):
        slide_context = SlideContext(row_index=self._row_index, metadata=self._metadata, dataset_paths=self._dataset_paths, desired_magnification=self._desired_magnification, tile_size=self._tile_size)
        self._slide = Slide(slide_context=slide_context)

    def post_process(self):
        pass


# =================================================
# SlidesManager Class
# =================================================
class SlidesManager(TaskParallelProcessor, SeedableObject, MetadataBase):
    def __init__(
            self,
            name: str,
            output_dir_path: Path,
            datasets_base_dir_path: Path,
            tile_size: int,
            desired_magnification: int,
            metadata_file_path: Path,
            num_workers: int):
        self._metadata_file_path = metadata_file_path
        self._slides = []
        self._current_slides = []
        self._slides_with_interior = []
        self._tile_to_slide_dict = self._create_tile_to_slide_dict()
        super().__init__(name=name, output_dir_path=output_dir_path, datasets_base_dir_path=datasets_base_dir_path, tile_size=tile_size, desired_magnification=desired_magnification, num_workers=num_workers)
        self._current_df = self._df
        self.process()
        self.join()

    @property
    def metadata(self) -> pandas.DataFrame:
        return self._df

    @property
    def slides_count(self) -> int:
        return self._current_df.shape[0]

    def get_slide_by_tile(self, tile: Tile) -> Slide:
        return self._tile_to_slide_dict[tile]

    def get_slides_ratio(self, ratio: float) -> List[Slide]:
        modulo = int(1 / ratio)
        return self.get_slides_modulo(modulo=modulo)

    def get_slides_modulo(self, modulo: int) -> List[Slide]:
        return self._slides[::modulo]

    def filter_folds(self, folds: Optional[List[int]]):
        if folds is not None:
            self._current_df = self._df[self._df[constants.fold_column_name].isin(folds)]
        else:
            self._current_df = self._df

        self._current_slides = self._get_slides()
        self._slides_with_interior = self._get_slides_with_interior_tiles()

    def get_slide(self, index: int) -> Slide:
        return self._current_slides[index]

    def get_random_slide(self) -> Slide:
        index = self._rng.integers(low=0, high=self._current_df.shape[0])
        return self.get_slide(index=index)

    def get_slide_with_interior(self, index: int) -> Slide:
        return self._slides_with_interior[index]

    def get_random_slide_with_interior(self) -> Slide:
        index = self._rng.integers(low=0, high=len(self._slides_with_interior))
        return self.get_slide_with_interior(index=index)

    def _load_metadata(self) -> pandas.DataFrame:
        return pandas.read_csv(filepath_or_buffer=self._metadata_file_path)

    def _post_join(self):
        self._slides = self._collect_slides()
        self._df = self._update_metadata()
        # print(f'TASKS LEN {len(self._tasks)}')
        # print(f'SLIDES LEN {len(self._slides)}')
        self._file_name_to_slide = self._create_file_name_to_slide_dict()
        self.filter_folds(folds=None)

    def _update_metadata(self) -> pandas.DataFrame:
        image_file_names = [slide.slide_context.image_file_name for slide in self._slides]
        return self._df[self._df[constants.file_column_name].isin(image_file_names)]

    def _collect_slides(self) -> List[Slide]:
        completed_tasks = cast(List[SlidesManagerTask], self._completed_tasks)
        return [task.slide for task in completed_tasks if task.slide is not None]

        # slides = []
        # for i, task in enumerate(self._completed_tasks):
        #     task = cast(SlidesManagerTask, task)
        #     if task.slide is not None:
        #         slides.append(task.slide)
        #
        # return slides

    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        tasks = []

        combinations = list(itertools.product(*[
            [*range(self._df.shape[0])],
            [self._df],
            [self._dataset_paths],
            [self._desired_magnification],
            [self._tile_size]]))

        for combination in combinations:
            tasks.append(SlidesManagerTask(
                row_index=combination[0],
                metadata=combination[1],
                dataset_paths=combination[2],
                desired_magnification=combination[3],
                tile_size=combination[4]))

        return tasks

    def _create_file_name_to_slide_dict(self) -> Dict[str, Slide]:
        file_name_to_slide = {}
        # print(f'slides len {len(self._slides)}')
        for slide in self._slides:
            # print(slide.slide_context.image_file_name)
            file_name_to_slide[slide.slide_context.image_file_name] = slide

        return file_name_to_slide

    def _create_tile_to_slide_dict(self) -> Dict[Tile, Slide]:
        tile_to_slide = {}
        for slide in self._slides:
            for tile in slide.tiles:
                tile_to_slide[tile] = slide

        return tile_to_slide

    def _get_slides(self) -> List[Slide]:
        # print(self._current_df[constants.file_column_name])
        # print(self._file_name_to_slide)
        return [self._file_name_to_slide[x] for x in self._current_df[constants.file_column_name]]

    def _get_slides_with_interior_tiles(self) -> List[Slide]:
        slides_with_interior_tiles = []
        for slide in self._slides:
            if slide.interior_tiles_count > 0:
                slides_with_interior_tiles.append(slide)

        return slides_with_interior_tiles


# =================================================
# MetadataGeneratorArgumentsParser Class
# =================================================
class MetadataGeneratorArgumentsParser(Tap):
    datasets_base_dir_path: Path
    tile_size: int
    desired_magnification: int
    metadata_file_path: Path
    metadata_enhancement_dir_path: Path
    log_file_path: Path
    dataset_ids: List[str]
    minimal_tiles_count: int
