# python core
from abc import ABC, abstractmethod
from typing import List
from enum import Enum
from pathlib import Path

# pandas
import pandas

# numpy
import numpy
import torch

# torch
from torch.utils.data import Dataset

# gipmed
from core.metadata import SlidesManager
from core.base import SeedableObject
from core.wsi import SlideContext, Tile, Slide, Patch, PatchExtractor, RandomPatchExtractor, ProximatePatchExtractor, BioMarker
from core.parallel_processing import TaskParallelProcessor, BufferedParallelProcessor, GetItemPolicy


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
# WSIMultiProcessingDataset Class
# =================================================
class WSIMultiProcessingDataset(BufferedParallelProcessor, WSIDataset):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, queue_maxsize: int, buffer_size: int, slides_manager: SlidesManager, dataset_size: int, get_item_policy: GetItemPolicy):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, queue_maxsize=queue_maxsize, buffer_size=buffer_size, slides_manager=slides_manager, dataset_size=dataset_size)
        self._get_item_policy = get_item_policy
        self.process()

    @abstractmethod
    def _generate_item(self) -> object:
        pass

    def __getitem__(self, index):
        return self.get_item(index=index, get_item_policy=self._get_item_policy)


# =================================================
# SingleTargetTrainingDataset Class
# =================================================
class SingleTargetTrainingDataset(WSIMultiProcessingDataset):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, queue_maxsize: int, buffer_size: int, slides_manager: SlidesManager, dataset_size: int, target: BioMarker):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, queue_maxsize=queue_maxsize, buffer_size=buffer_size, slides_manager=slides_manager, dataset_size=dataset_size)
        self._target = target

    def _generate_item(self) -> object:
        slide = self._slides_manager.get_random_slide()
        patch_extractor = RandomPatchExtractor(slide=slide)
        patch = patch_extractor.extract_patch(patch_validators=[])
        label = slide.slide_context.get_biomarker_value(bio_marker=self._target)
        return patch, label


# =================================================
# SingleTargetValidationDataset Class
# =================================================
class SingleTargetValidationDataset(WSIDataset):
    def __init__(
            self,
            slides_manager: SlidesManager,
            slides_ratio: int,
            tiles_ratio: int,
            bio_marker: BioMarker):
        self._slides_ratio = slides_ratio
        self._tiles_ratio = tiles_ratio
        self._bio_marker = bio_marker
        self._tiles = self._get_tiles()
        super().__init__(slides_manager=slides_manager, dataset_size=len(self._tiles))

    def __getitem__(self, index):
        patch_images = []
        tile = self._tiles[index]
        slide = self._slides_manager.get_slide_by_tile(tile=tile)
        patch_images.append(tile.image)
        label = slide.slide_context.get_biomarker_value(bio_marker=self._bio_marker)
        images_tuplet = numpy.transpose(numpy.stack(patch_images), (0, 3, 1, 2))

        return {
            'input': images_tuplet,
            'label': label,
            'slide_id': slide.slide_context.row_index
        }

    def _get_tiles(self) -> List[Tile]:
        tiles = []
        for slide in self._slides_manager.get_slides_ratio(ratio=self._slides_ratio):
            for tile in slide.get_tiles_ratio(ratio=self._tiles_ratio):
                tiles.append(tile)

        return tiles


# =================================================
# SSLDataset Class
# =================================================
class SSLDataset(WSIMultiProcessingDataset):
    _white_ratio_threshold = 0.5
    _white_intensity_threshold = 170

    def __init__(self, slides_manager: SlidesManager, dataset_size: int, inner_radius_mm: float, negative_examples_count: int):
        super().__init__(dataset_size=dataset_size, slides_manager=slides_manager)
        self._inner_radius_mm = inner_radius_mm
        self._negative_examples_count = negative_examples_count

    def __getitem__(self, index):
        while True:
            patch_images = []

            slide = self._slides_manager.get_random_slide_with_interior()
            patch_extractor = RandomPatchExtractor(slide=slide)
            anchor_patch = patch_extractor.extract_patch(patch_validators=[SSLDataset._validate_histogram])
            if anchor_patch is None:
                continue

            patch_images.append(numpy.array(anchor_patch.image))

            patch_extractor = ProximatePatchExtractor(slide=slide, reference_patch=anchor_patch, inner_radius_mm=self._inner_radius_mm)
            positive_patch = patch_extractor.extract_patch(patch_validators=[])
            if positive_patch is None:
                continue

            patch_images.append(numpy.array(positive_patch.image))

            # for i in range(negative_examples_count):
            #     pass

            images_tuplet = numpy.transpose(numpy.stack(patch_images), (0, 3, 1, 2))
            return images_tuplet

    @staticmethod
    def _validate_histogram(patch: Patch) -> bool:
        patch_grayscale = patch.image.convert('L')
        hist, _ = numpy.histogram(a=patch_grayscale, bins=patch.slide_context.tile_size)
        white_ratio = numpy.sum(hist[SSLDataset._white_intensity_threshold:]) / (patch.slide_context.tile_size * patch.slide_context.tile_size)
        if white_ratio > SSLDataset._white_ratio_threshold:
            return False
        return True
