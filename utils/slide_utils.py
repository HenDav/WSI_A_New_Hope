# python peripherals
import math

# numpy
import numpy

# openslide
import openslide


def read_region_around_point(slide: openslide.OpenSlide, point: numpy.ndarray, tile_size: int, selected_level_tile_size: int, level: int):
    top_left_point = (point - selected_level_tile_size / 2).astype(int)
    tile = slide.read_region(top_left_point, level, (selected_level_tile_size, selected_level_tile_size)).convert('RGB')
    if selected_level_tile_size != tile_size:
        tile = tile.resize((tile_size, tile_size))
    return tile


def get_best_level_for_downsample(slide: openslide.OpenSlide, desired_downsample: int, tile_size: int):
    level = 0
    level_downsample = 1
    if desired_downsample < 1:
        tile_size = int(desired_downsample * tile_size)
    else:
        for i, downsample in enumerate(slide.level_downsamples):
            if math.isclose(desired_downsample, downsample, rel_tol=1e-3):
                level = i
                level_downsample = 1
                break
            elif downsample < desired_downsample:
                level = i
                level_downsample = int(desired_downsample / slide.level_downsamples[level])

    # A tile of size (tile_size, tile_size) in an image downsampled by 'level_downsample', will cover the same image portion of a tile of size (adjusted_tile_size, adjusted_tile_size) in the original image
    selected_level_tile_size = tile_size * level_downsample

    return level, selected_level_tile_size


def get_pixel_to_mm(downsample: int, image_file_suffix: str):
    if image_file_suffix == '.svs':
        return 0.001 / downsample

    raise 'Unknown image type'


def get_mm_to_pixel(downsample: int, image_file_suffix: str):
    return 1 / get_pixel_to_mm(downsample=downsample, image_file_suffix=image_file_suffix)
