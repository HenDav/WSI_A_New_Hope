from typing import Dict, List, Tuple

# General parameters
test_fold_id = 'test'
max_attempts = 10
white_ratio_threshold = 0.5
white_intensity_threshold = 170

# Invalid values
invalid_values = ['Missing Data', 'Not performed', '[Not Evaluated]', '[Not Available]']
invalid_value = 'NA'
invalid_fold_column_names = ['test fold idx breast', 'test fold idx', 'test fold idx breast - original for carmel']

# Dataset ids
dataset_id_abctb = 'ABCTB'
dataset_id_sheba = 'SHEBA'
dataset_id_carmel = 'CARMEL'
dataset_id_tcga = 'TCGA'
dataset_id_prefixes = [dataset_id_abctb, dataset_id_sheba, dataset_id_carmel, dataset_id_tcga]

# Grid data
bad_segmentation_column_name = 'bad segmentation'
grids_data_prefix = 'slides_data_'
grid_data_file_name = 'Grid_data.xlsx'

# Carmel
slide_barcode_column_name_carmel = 'slide barcode'
slide_barcode_column_name_enhancement_carmel = 'TissueID'
patient_barcode_column_name_enhancement_carmel = 'PatientIndex'
block_id_column_name_enhancement_carmel = 'BlockID'

# TCGA
patient_barcode_column_name_enhancement_tcga = 'Sample CLID'
slide_barcode_prefix_column_name_enhancement_tcga = 'Sample CLID'

# ABCTB
file_column_name_enhancement_abctb = 'Image File'
patient_barcode_column_name_enhancement_abctb = 'Identifier'

# SHEBA
er_status_column_name_sheba = 'ER '
pr_status_column_name_sheba = 'PR '
her2_status_column_name_sheba = 'HER-2 IHC '
grade_column_name_sheba = 'Grade'
tumor_type_column_name_sheba = 'Histology'

# Shared
file_column_name_shared = 'file'
patient_barcode_column_name_shared = 'patient barcode'
dataset_id_column_name_shared = 'id'
mpp_column_name_shared = 'MPP'
scan_date_column_name_shared = 'Scan Date'
width_column_name_shared = 'Width'
height_column_name_shared = 'Height'
magnification_column_name_shared = 'Manipulated Objective Power'
er_status_column_name_shared = 'ER status'
pr_status_column_name_shared = 'PR status'
her2_status_column_name_shared = 'Her2 status'
fold_column_name_shared = 'test fold idx'

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


def get_path_suffixes() -> Dict[str, str]:
    path_suffixes = {
        dataset_id_tcga: f'Breast/{dataset_id_tcga}',
        dataset_id_abctb: f'Breast/{dataset_id_abctb}_TIF',
    }

    for i in range(1, 12):
        path_suffixes[f'{dataset_id_carmel}{i}'] = f'Breast/{dataset_id_carmel.capitalize()}/Batch_{i}/{dataset_id_carmel}{i}'

    for i in range(2, 7):
        path_suffixes[f'{dataset_id_sheba}{i}'] = f'Breast/{dataset_id_sheba.capitalize()}/Batch_{i}/{dataset_id_sheba}{i}'

    return path_suffixes
