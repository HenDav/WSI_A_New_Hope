import utils
import argparse

parser = argparse.ArgumentParser(description='Data preparation script')
parser.add_argument('--data_collection', type=bool, default=False, help='need to collect data from main data file?')
parser.add_argument('--segmentation', type=bool, default=False, help='need to make segmentation map for all files?')
parser.add_argument('--grid', type=bool, default=False, help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', type=bool, default=False, help='need to compute statistical data?')
parser.add_argument('--hard_copy', type=bool, default=False, help='make hard copy of tiles?')

args = parser.parse_args()

if args.data_collection:
    utils.make_slides_xl_file()
if args.segmentation:
    utils.make_segmentations(rewrite=False)
    utils.copy_segImages()
if args.grid:
    utils.make_grid(tile_sz=args.tile_size)
if args.stats:
    utils.compute_normalization_values()
if args.hard_copy:
    utils.make_tiles_hard_copy()

print('Data Preparation sequence is Done !')



