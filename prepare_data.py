import utils
import argparse

parser = argparse.ArgumentParser(description='Data preparation script')
parser.add_argument('--data_collection', type=bool, default=False, help='need to collect data from main data file?')
parser.add_argument('--segmentation', type=bool, default=False, help='need to make segmentation map for all files?')
parser.add_argument('--grid', type=bool, default=False, help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', type=bool, default=False, help='need to compute statistical data?')
parser.add_argument('--hard_copy', type=bool, default=False, help='make hard copy of tiles?')
parser.add_argument('--data_folder', type=str, default='tcga-data', help='location of data folder')
parser.add_argument('--format', type=str, default='TCGA', help='format of data folder - TCGA/ABCTB/MIRAX')
args = parser.parse_args()

if __name__ =='__main__':
    if args.data_collection:
        utils.make_slides_xl_file(path=args.data_folder)
    if args.segmentation:
        utils.make_segmentations(data_path=args.data_folder + '/', rewrite=False, data_format=args.format)
        #utils.copy_segImages(data_path=args.data_folder, data_format=args.format)
    if args.grid:
        utils.make_grid(data_path=args.data_folder, tile_sz=args.tile_size)
    if args.stats:
        utils.compute_normalization_values()
    if args.hard_copy:
        utils.make_tiles_hard_copy()

    print('Data Preparation sequence is Done !')



