# python core
from abc import ABC, abstractmethod
from typing import List
from enum import Enum

# pandas
import pandas

# numpy
import numpy

# torch
from torch.utils.data import Dataset

# gipmed
from core.metadata import SlidesManager
from core.base import SeedableObject
from core.wsi import SlideContext, Slide, Patch, PatchExtractor, RandomPatchExtractor, ProximatePatchExtractor, Target


# =================================================
# TupletsDataset Class
# =================================================
class WSIDataset(ABC, Dataset, SeedableObject):
    def __init__(self, slides_manager: SlidesManager, dataset_size: int):
        super(Dataset, self).__init__()
        super(SeedableObject, self).__init__()
        self._dataset_size = dataset_size
        self._slides_manager = slides_manager

    def __len__(self):
        return self._dataset_size

    @abstractmethod
    def __getitem__(self, index):
        pass

    def set_folds(self, folds: List[int]):
        self._slides_manager.filter_folds(folds=folds)


# =================================================
# SingleTargetTrainingDataset Class
# =================================================
class SingleTargetTrainingDataset(WSIDataset):
    def __init__(self, slides_manager: SlidesManager, dataset_size: int, target: Target):
        super().__init__(dataset_size=dataset_size, slides_manager=slides_manager)
        self._target = target

    def __getitem__(self, index):
        slide = self._slides_manager.get_random_slide()
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_target(target=self._target)
        return patch, label


# =================================================
# SingleTargetValidationDataset Class
# =================================================
class SingleTargetValidationDataset(WSIDataset):
    def __init__(self, slides_manager: SlidesManager, slides_delta: int, tiles_delta: int, target: Target):
        super().__init__(slides_manager=slides_manager)
        self._slides_delta = slides_delta
        self._tiles_delta = tiles_delta
        self._target = target

    def __getitem__(self, index):
        slide = self._slides_manager.get_random_slide()
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_target(target=self._target)
        return patch, label


# =================================================
# SSLDataset Class
# =================================================
class SSLDataset(WSIDataset):
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

    def __init__(self, slides_manager: SlidesManager, dataset_size: int, inner_radius_mm: float, negative_examples_count: int):
        super().__init__(dataset_size=dataset_size, slides_manager=slides_manager)
        self._inner_radius_mm = inner_radius_mm
        self._negative_examples_count = negative_examples_count

    def __getitem__(self, index):
        while True:
            patches = []

            slide = self._slides_manager.get_random_slide_with_interior()
            patch_extractor = RandomPatchExtractor(slide=slide)
            anchor_patch = patch_extractor.extract_patch(patch_validators=[SSLDataset._validate_histogram])
            if anchor_patch is None:
                continue

            patches.append(numpy.array(anchor_patch.image))

            patch_extractor = ProximatePatchExtractor(slide=slide, reference_patch=anchor_patch, inner_radius_mm=self._inner_radius_mm)
            positive_patch = patch_extractor.extract_patch(patch_validators=[])
            if positive_patch is None:
                continue

            patches.append(numpy.array(positive_patch.image))

            # for i in range(negative_examples_count):
            #     pass

            patches_tuplet = numpy.transpose(numpy.stack(patches), (0, 3, 1, 2))
            return patches_tuplet

    @staticmethod
    def _validate_histogram(patch: Patch) -> bool:
        patch_grayscale = patch.image.convert('L')
        hist, _ = numpy.histogram(a=patch_grayscale, bins=patch.slide_context.tile_size)
        white_ratio = numpy.sum(hist[SSLDataset._white_intensity_threshold:]) / (patch.slide_context.tile_size * patch.slide_context.tile_size)
        if white_ratio > SSLDataset._white_ratio_threshold:
            return False
        return True



# # =================================================
# # DatasetArgumentsParser Class
# # =================================================
# class DatasetArgumentsParser(ABC, Tap):
#     dataset_size: int
#
#
# # =================================================
# # SSLDatasetArgumentsParser Class
# # =================================================
# class SSLDatasetArgumentsParser(DatasetArgumentsParser, ArgumentsParser[SSLDataset]):
#     inner_radius_mm: float
#     negative_examples_count: int
#
#     def create(self) -> SSLDataset:
#         metadata_manager_arguments_parser = MetadataManagerArgumentsParser()
#         metadata_manager = metadata_manager_arguments_parser.create()
#         return SSLDataset(metadata_manager=metadata_manager, dataset_size=self.dataset_size, inner_radius_mm=self.inner_radius_mm, negative_examples_count=self.negative_examples_count)
