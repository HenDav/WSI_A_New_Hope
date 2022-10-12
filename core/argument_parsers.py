# python core
from typing import List, Optional

# tap
from tap import Tap


class MetadataArgumentsParser(Tap):
    tile_size: int
    desired_magnification: int
    enhancement_dir_path: str
    datasets_base_dir_path: str
    dataset_ids: List[str]
    minimal_tiles_count: int


class LossArgumentsParser(Tap):
    loss_class_name: str


class OptimizerArgumentsParser(Tap):
    optimizer_class_name: str
    lr: float


class CrossValidationArgumentsParser(Tap):
    folds: List[int]


class DatasetArgumentsParser(Tap):
    dataset_class_name: str


class FeatureExtractorArgumentsParser(Tap):
    feature_extractor_class_name: str


class ClassifierExtractorArgumentsParser(Tap):
    classifier_class_name: str


class SSLArgumentsParser(Tap):
    inner_radius: float
    outer_radius: float
    negative_examples_count: int

