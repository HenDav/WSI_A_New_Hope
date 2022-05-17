import math
import numpy as np


def read_region_around_point(slide, point, tile_size, adjusted_tile_size, level):
    top_left_point = (point - adjusted_tile_size / 2).astype(int)
    tile = slide.read_region(top_left_point, level, (adjusted_tile_size, adjusted_tile_size)).convert('RGB')
    if adjusted_tile_size != tile_size:
        tile = tile.resize((tile_size, tile_size))
    return tile


def get_best_level_for_downsample(slide, desired_downsample, tile_size):
    level = None
    level_downsample = None
    for i, downsample in enumerate(slide.level_downsamples):
        if math.isclose(desired_downsample, downsample, rel_tol=1e-3):
            level = i
            level_downsample = 1
            break

        elif downsample < desired_downsample:
            level = i
            level_downsample = int(desired_downsample / slide.level_downsamples[level])

    # A tile of size (tile_size, tile_size) in an image downsampled by 'level_downsample', will cover the same image portion of a tile of size (adjusted_tile_size, adjusted_tile_size) in the original image
    adjusted_tile_size = tile_size * level_downsample

    return level, adjusted_tile_size


def get_pixel_to_mm(downsample, image_file_suffix):
    if image_file_suffix == '.svs':
        return 0.001 / downsample

    raise 'Unknown image type'


def get_mm_to_pixel(downsample, image_file_suffix):
    return 1 / get_pixel_to_mm(downsample=downsample, image_file_suffix=image_file_suffix)
