# python core
import os
import pickle
from pathlib import Path
import math
import itertools
from typing import List, Dict, Union, Callable, Optional, cast
from abc import ABC, abstractmethod
from enum import Enum, auto

# pandas
import pandas

# numpy
import numpy

# openslide
OPENSLIDE_PATH = r'C:\openslide-win64-20220811\bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# PIL
from PIL import Image

# opencv
import cv2

# matplotlib
from matplotlib import pyplot as plt

# wsi
from core.parallel_processing import ParallelProcessor, ParallelProcessorTask
from core import constants, utils


# =================================================
# Targets Class
# =================================================
class Target(Enum):
    ER = auto()
    PR = auto()
    HER2 = auto()


# =================================================
# SlideContext Class
# =================================================
class SlideContext:
    def __init__(self, row: pandas.DataFrame, dataset_path: Path, desired_magnification: int, tile_size: int):
        self._row = row
        self._dataset_path = dataset_path
        self._desired_magnification = desired_magnification
        self._tile_size = tile_size
        self._image_file_name = self._row[constants.file_column_name].item()
        self._image_file_path = dataset_path / self._image_file_name
        self._dataset_id = self._row[constants.dataset_id_column_name].item()
        self._image_file_name_stem = self._image_file_path.stem
        self._image_file_name_suffix = self._image_file_path.suffix
        self._magnification = row[constants.magnification_column_name].item()
        self._mpp = utils.magnification_to_mpp(magnification=self._magnification)
        self._legitimate_tiles_count = row[constants.legitimate_tiles_column_name].item()
        self._fold = row[constants.fold_column_name].item()
        self._desired_downsample = self._magnification / self._desired_magnification
        self._slide = openslide.open_slide(self._image_file_path)
        self._level, self._level_downsample = self._get_best_level_for_downsample()
        self._selected_level_tile_size = self._tile_size * self._level_downsample
        self._zero_level_tile_size = self._tile_size * self._desired_downsample
        self._er = self._row[constants.er_status_column_name]
        self._pr = self._row[constants.pr_status_column_name]
        self._her2 = self._row[constants.her2_status_column_name]

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    @property
    def desired_magnification(self) -> int:
        return self._desired_magnification

    @property
    def image_file_name(self) -> str:
        return self._image_file_name

    @property
    def image_file_path(self) -> Path:
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

    @property
    def er(self) -> bool:
        return self._er

    @property
    def pr(self) -> bool:
        return self._pr

    @property
    def her2(self) -> bool:
        return self._her2

    def mm_to_pixels(self, mm: float) -> int:
        pixels = int(mm / (self._mpp / 1000))
        return pixels

    def read_region_around_pixel(self, pixel: numpy.ndarray) -> Image:
        top_left_pixel = (pixel - self.selected_level_tile_size / 2).astype(int)
        region = self.slide.read_region(top_left_pixel, self.level, (self.selected_level_tile_size, self.selected_level_tile_size)).convert('RGB')
        if self.selected_level_tile_size != self.tile_size:
            region = region.resize((self.tile_size, self.tile_size))
        return region

    def get_target(self, target: Target) -> bool:
        if target is Target.ER:
            return self._er
        elif target is Target.PR:
            return self._pr
        elif target is Target.HER2:
            return self._her2

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


# =================================================
# SlideElement Class
# =================================================
class SlideElement:
    def __init__(self, slide_context: SlideContext):
        self._slide_context = slide_context

    @property
    def slide_context(self) -> SlideContext:
        return self._slide_context


# =================================================
# Tile Class
# =================================================
class Tile(SlideElement):
    def __init__(self, slide_context: SlideContext, tile_location: numpy.ndarray):
        super().__init__(slide_context=slide_context)
        self._slide_context = slide_context
        self._tile_location = tile_location.astype(numpy.int64)

    @property
    def tile_location(self) -> numpy.ndarray:
        return self._tile_location

    def get_random_pixel(self) -> numpy.ndarray:
        pixel = self._tile_location * self._slide_context.zero_level_tile_size + self._slide_context.zero_level_half_tile_size
        offset = (self._slide_context.zero_level_half_tile_size * numpy.random.uniform(size=2)).astype(int)
        pixel = pixel + offset
        return pixel


# =================================================
# TilesManager Class
# =================================================
class TilesManager(ABC, SlideElement):
    def __init__(self, slide_context: SlideContext, locations: Optional[numpy.array]):
        super().__init__(slide_context=slide_context)

        if locations is None:
            self._locations = self._create_tile_locations()
        else:
            self._locations = locations

        self._tiles = self._create_tiles_list()
        self._location_to_tile = self._create_tiles_dict()
        self._interior_tiles = self._create_interior_tiles_list()

    @property
    def tiles_count(self) -> int:
        return len(self._tiles)

    def get_random_tile(self) -> Tile:
        tile_index = int(numpy.random.randint(len(self._tiles), size=1))
        return self._tiles[tile_index]

    def get_random_interior_tile(self) -> Tile:
        tile_index = int(numpy.random.randint(len(self._interior_tiles), size=1))
        return self._interior_tiles[tile_index]

    def get_random_pixel(self) -> numpy.ndarray:
        tile = self.get_random_tile()
        return tile.get_random_pixel()

    def get_random_interior_pixel(self) -> numpy.ndarray:
        tile = self.get_random_interior_tile()
        return tile.get_random_pixel()

    def get_tile_at_pixel(self, pixel: numpy.ndarray) -> Optional[Tile]:
        tile_location = (pixel / self._slide_context.zero_level_tile_size).astype(numpy.int64)
        return self._location_to_tile.get(tile_location.tobytes())

    def is_interior_tile(self, tile: Tile) -> bool:
        for i in range(3):
            for j in range(3):
                current_tile_location = tile.tile_location + numpy.array([i, j])
                if not self._is_valid_tile_location(tile_location=current_tile_location):
                    return False
        return True

    def _create_tile_locations(self) -> numpy.ndarray:
        grid_file_path = os.path.normpath(os.path.join(self._slide_context.dataset_path, f'Grids_{self._slide_context.desired_magnification}', f'{self._slide_context.image_file_name_stem}--tlsz{self._slide_context.tile_size}.data'))
        with open(grid_file_path, 'rb') as file_handle:
            locations = numpy.array(pickle.load(file_handle))
            locations[:, 0], locations[:, 1] = locations[:, 1], locations[:, 0].copy()

        return locations

    def _create_tiles_list(self) -> List[Tile]:
        tiles = []
        for i in range(self._locations.shape[0]):
            tiles.append(Tile(slide_context=self._slide_context, tile_location=self._locations[i, :]))
        return tiles

    def _create_interior_tiles_list(self) -> List[Tile]:
        interior_tiles = []
        for tile in self._tiles:
            if self.is_interior_tile(tile=tile):
                interior_tiles.append(tile)

        return interior_tiles

    def _create_tiles_dict(self) -> Dict[bytes, Tile]:
        tiles_dict = {}
        for tile in self._tiles:
            tiles_dict[tile.tile_location.tobytes()] = tile

        return tiles_dict

    def _is_valid_tile_location(self, tile_location: numpy.ndarray) -> bool:
        if tile_location.tobytes() in self._location_to_tile:
            return True
        return False


# =================================================
# ConnectedComponent Class
# =================================================
class ConnectedComponent(TilesManager):
    def __init__(self, slide_context: SlideContext, locations: numpy.ndarray):
        super().__init__(slide_context=slide_context, locations=locations)
        self._top_left_tile_location = numpy.array([numpy.min(self._locations[0, :]), numpy.min(self._locations[1, :])])
        self._bottom_right_tile_location = numpy.array([numpy.max(self._locations[0, :]), numpy.max(self._locations[1, :])])

    @property
    def top_left_tile_location(self) -> numpy.ndarray:
        return self._top_left_tile_location

    @property
    def bottom_right_tile_location(self) -> numpy.ndarray:
        return self._bottom_right_tile_location

    def calculate_bounding_box_aspect_ratio(self):
        box = (self.bottom_right_tile_location - self.top_left_tile_location) + 1
        return box[0] / box[1]


# =================================================
# Patch Class
# =================================================
class Patch(SlideElement):
    def __init__(self, slide_context: SlideContext, center_pixel: numpy.ndarray):
        super().__init__(slide_context=slide_context)
        self._center_pixel = center_pixel
        self._image = self._slide_context.read_region_around_pixel(pixel=center_pixel)

    @property
    def image(self) -> Image:
        return self._image

    @property
    def center_pixel(self) -> numpy.ndarray:
        return self._center_pixel

    # def get_white_ratio(self, white_intensity_threshold: int) -> float:
    #     patch_grayscale = self._image.convert('L')
    #     hist, _ = numpy.histogram(a=patch_grayscale, bins=self._slide_context.tile_size)
    #     white_ratio = numpy.sum(hist[white_intensity_threshold:]) / (self._slide_context.tile_size * self._slide_context.tile_size)
    #     return white_ratio


# =================================================
# Slide Class
# =================================================
class Slide(TilesManager):
    def __init__(self, slide_context: SlideContext, min_component_ratio: float = 0.92, max_aspect_ratio_diff: float = 0.02):
        super().__init__(slide_context=slide_context, locations=None)
        self._min_component_ratio = min_component_ratio
        self._max_aspect_ratio_diff = max_aspect_ratio_diff
        self._bitmap = self._create_tile_bitmap(plot_bitmap=False)
        self._components = self._create_connected_components()

    @property
    def components(self):
        return self._components

    def get_component(self, component_index: int) -> ConnectedComponent:
        return self._components[component_index]

    def get_random_component(self) -> ConnectedComponent:
        component_index = int(numpy.random.randint(len(self._components), size=1))
        return self.get_component(component_index=component_index)

    def get_component_at_pixel(self, pixel: numpy.ndarray) -> Optional[ConnectedComponent]:
        tile_location = (pixel / self._slide_context.zero_level_tile_size).astype(numpy.int64)
        return self._location_to_tile.get(tile_location.tobytes())

    def _create_tile_bitmap(self, plot_bitmap: bool = False) -> Image:
        indices = (numpy.array(self._locations) / self._slide_context.zero_level_tile_size).astype(int)
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

    def _create_connected_component_from_bitmap(self, bitmap: numpy.ndarray) -> ConnectedComponent:
        valid_tile_indices = numpy.where(bitmap)
        locations = numpy.array([valid_tile_indices[0], valid_tile_indices[1]]).transpose()
        return ConnectedComponent(slide_context=self._slide_context, locations=locations)

    def _create_connected_components(self) -> List[ConnectedComponent]:
        components_count, components_labels, components_stats, _ = cv2.connectedComponentsWithStats(self._bitmap)
        components = []

        for component_id in range(1, components_count):
            bitmap = (components_labels == component_id).astype(int)
            connected_component = self._create_connected_component_from_bitmap(bitmap=bitmap)
            components.append(connected_component)

        components_sorted = sorted(components, key=lambda item: item.valid_tiles_count, reverse=True)
        largest_component = components_sorted[0]
        largest_component_aspect_ratio = largest_component.calculate_bounding_box_aspect_ratio()
        largest_component_size = largest_component.tiles_count
        valid_components = [largest_component]
        for component in components_sorted[1:]:
            current_aspect_ratio = component.calculate_bounding_box_aspect_ratio()
            current_component_size = component.tiles_count
            if (numpy.abs(largest_component_aspect_ratio - current_aspect_ratio) < self._max_aspect_ratio_diff) and ((current_component_size / largest_component_size) > self._min_component_ratio):
                valid_components.append(component)

        return valid_components


# =================================================
# PatchExtractor Class
# =================================================
class PatchExtractor(ABC):
    def __init__(self, slide: Slide, max_attempts: int = 10):
        self._slide = slide
        self._max_attempts = max_attempts

    @abstractmethod
    def _extract_center_pixel(self) -> Optional[numpy.ndarray]:
        pass

    def extract_patch(self, patch_validators: List[Callable[[Patch], bool]]) -> Optional[Patch]:
        attempts = 0
        while True:
            center_pixel = self._extract_center_pixel()
            if center_pixel is None:
                attempts = attempts + 1
                continue

            patch = Patch(slide_context=self._slide.slide_context, center_pixel=center_pixel)

            patch_validation_failed = False
            for patch_validator in patch_validators:
                if not patch_validator(patch):
                    patch_validation_failed = True
                    break

            if patch_validation_failed is True:
                attempts = attempts + 1
                continue

            return patch
        return None


# =================================================
# RandomPatchExtractor Class
# =================================================
class RandomPatchExtractor(PatchExtractor):
    def __init__(self, slide: Slide):
        super().__init__(slide=slide)

    def _extract_center_pixel(self) -> Optional[numpy.ndarray]:
        component = self._slide.get_random_component()
        tile = component.get_random_interior_tile()
        pixel = tile.get_random_pixel()
        return pixel


# =================================================
# ProximatePatchExtractor Class
# =================================================
class ProximatePatchExtractor(PatchExtractor):
    _max_attempts = 10

    def __init__(self, slide: Slide, reference_patch: Patch, inner_radius_mm: float):
        super().__init__(slide=slide)
        self._reference_patch = reference_patch
        self._inner_radius_mm = inner_radius_mm
        self._inner_radius_pixels = self._slide.slide_context.mm_to_pixels(mm=inner_radius_mm)

    def _extract_center_pixel(self) -> Optional[numpy.ndarray]:
        pixel = self._reference_patch.center_pixel
        angle = 2 * numpy.pi * numpy.random.uniform(size=1)[0]
        direction = numpy.array([numpy.cos(angle), numpy.sin(angle)])
        proximate_pixel = (pixel + self._inner_radius_pixels * direction).astype(int)
        tile = self._slide.get_tile_at_pixel(pixel=proximate_pixel)
        if (tile is None) and self._slide.is_interior_tile(tile=cast(Tile, tile)):
            return proximate_pixel

        return None


# # =================================================
# # SlideContextsGeneratorTask Class
# # =================================================
# class SlideContextsGeneratorTask(ParallelProcessorTask):
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
#
#
# # =================================================
# # SlideContextsGenerator Class
# # =================================================
# class SlideContextsGenerator(ParallelProcessor):
#     def __init__(
#             self,
#             metadata: pandas.DataFrame,
#             datasets_base_dir_path: Path,
#             desired_magnification: int,
#             tile_size: int):
#         self._metadata = metadata
#         self._datasets_base_dir_path = datasets_base_dir_path
#         self._desired_magnification = desired_magnification
#         self._tile_size = tile_size
#         self._dataset_paths = constants.get_dataset_paths(datasets_base_dir_path=datasets_base_dir_path)
#         self._slide_contexts = []
#         self._file_name_to_slide_context = {}
#         super().__init__()
#
#     @property
#     def slide_contexts(self) -> List[SlideContext]:
#         return self._slide_contexts
#
#     @property
#     def file_name_to_slide_context(self) -> Dict[str, SlideContext]:
#         return self._file_name_to_slide_context
#
#     def _post_process(self):
#         for completed_task in self._completed_tasks:
#             self._slide_contexts.append(completed_task.slide_context)
#             self._file_name_to_slide_context[completed_task.slide_context.image_file_name] = completed_task.slide_context
#
#     def _generate_tasks(self) -> List[ParallelProcessorTask]:
#         tasks = []
#         row_indices = list(range(self._metadata.shape[0]))
#         combinations = list(itertools.product(*[
#             row_indices,
#             [self._metadata],
#             [self._dataset_paths],
#             [self._tile_size],
#             [self._desired_magnification]]))
#
#         for combination in combinations:
#             tasks.append(SlideContextsGeneratorTask(
#                 row_index=combination[0],
#                 metadata=combination[1],
#                 dataset_paths=combination[2],
#                 tile_size=combination[3],
#                 desired_magnification=combination[4],
#             ))
#
#         return tasks
