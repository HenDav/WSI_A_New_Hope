# python core
from abc import ABC, abstractmethod
from typing import List

# pandas
import pandas

# numpy
import numpy

# torch
from torch.utils.data import Dataset

# gipmed
from core.metadata import MetadataManager
from core.base import SeedableObject
from core.wsi import SlideContext, Slide, Patch, PatchExtractor


# =================================================
# TupletsDataset Class
# =================================================
class WSIDataset(ABC, Dataset, SeedableObject):
    def __init__(self, metadata_manager: MetadataManager, dataset_size: int):
        super(Dataset, self).__init__()
        super(SeedableObject, self).__init__()
        self._dataset_size = dataset_size
        self._metadata_manager = metadata_manager

    def __len__(self):
        return self._dataset_size

    @abstractmethod
    def __getitem__(self, index):
        pass

    def set_folds(self, folds: List[int]):
        self._metadata_manager.filter_folds(folds=folds)


# =================================================
# SSLDataset Class
# =================================================
class SSLDataset(WSIDataset):
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

    def __init__(self, metadata_manager: MetadataManager, dataset_size: int, inner_radius_mm: float, negative_examples_count: int):
        super(WSIDataset, self).__init__(dataset_size=dataset_size, metadata_manager=metadata_manager)
        self._inner_radius_mm = inner_radius_mm
        self._negative_examples_count = negative_examples_count

    def __getitem__(self, index):
        patches = []

        # index = self._rng.integers(low=0, high=len(self._slide_contexts))
        # slide_context = self._slide_contexts[index]
        # slide = Slide(slide_context=slide_context)
        slide = self._metadata_manager.get_random_slide()
        patch_extractor = PatchExtractor(slide=slide, inner_radius_mm=self._inner_radius_mm)

        anchor_patch = patch_extractor.extract_patch(patch_validators=[SSLDataset._validate_histogram])
        patches.append(numpy.array(anchor_patch.image))

        positive_patch = patch_extractor.extract_patch(patch_validators=[], reference_patch=anchor_patch)
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
