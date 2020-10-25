import utils_data_managment
import argparse

parser = argparse.ArgumentParser(description='Data preparation script')
"""
parser.add_argument('--data_collection', type=bool, default=False, help='need to collect data from main data file?')
parser.add_argument('--segmentation', type=bool, default=False, help='need to make segmentation map for all files?')
parser.add_argument('--grid', type=bool, default=True, help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=64, help='size of tiles')
parser.add_argument('--stats', type=bool, default=False, help='need to compute statistical data?')
parser.add_argument('--hard_copy', type=bool, default=False, help='make hard copy of tiles?')
parser.add_argument('--data_folder', type=str, default='All Data/TCGA', help='location of data folder')
"""
parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='need to collect data from main data file?')
parser.add_argument('--segmentation', dest='segmentation', action='store_true', help='need to make segmentation map for all files?')
parser.add_argument('--grid', dest='grid', action='store_true', help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', dest='stats', action='store_true', help='need to compute statistical data?')
parser.add_argument('--hard_copy', dest='hard_copy', action='store_true', help='make hard copy of tiles?')
parser.add_argument('--data_folder', type=str, default='All Data/TCGA', help='location of data folder')
args = parser.parse_args()

if __name__ =='__main__':
    if args.data_collection:
        utils_data_managment.make_slides_xl_file(path=args.data_folder)
    if args.segmentation:
        utils_data_managment.make_segmentations(data_path=args.data_folder + '/', rewrite=False)
    if args.grid:
        utils_data_managment.make_grid(data_path=args.data_folder, tile_sz=args.tile_size)
    if args.stats:
        utils_data_managment.compute_normalization_values()
    if args.hard_copy:
        utils_data_managment.make_tiles_hard_copy(tile_size=args.tile_size)

    print('Data Preparation sequence is Done !')



