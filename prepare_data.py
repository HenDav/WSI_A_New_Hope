import utils_data_managment
import argparse
import os

parser = argparse.ArgumentParser(description='Data preparation script')

parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='need to collect data from main data file?')
parser.add_argument('--segmentation', dest='segmentation', action='store_true', help='need to make segmentation map for all files?')
parser.add_argument('--grid', dest='grid', action='store_true', help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', dest='stats', action='store_true', help='need to compute statistical data?')
parser.add_argument('--hard_copy', dest='hard_copy', action='store_true', help='make hard copy of tiles?')
parser.add_argument('--data_folder', type=str, default='All Data/HEROHE', help='location of data folder')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='type of dataset to use')
args = parser.parse_args()

if __name__ =='__main__':
    if args.data_collection:
        utils_data_managment.make_slides_xl_file(DataSet=args.dataset)
    if args.segmentation:
        utils_data_managment.make_segmentations(data_path=args.data_folder)
        '''
        data_dirs = [f.path for f in os.scandir(args.data_folder) if f.is_dir()]
        for data_dir in data_dirs:
            utils_data_managment.make_segmentations(data_path=data_dir, rewrite=False)
        '''
    if args.grid:
        utils_data_managment.make_grid(DataSet=args.dataset, tile_sz=args.tile_size)
    if args.stats:
        utils_data_managment.compute_normalization_values(DataSet=args.dataset)
    if args.hard_copy:
        utils_data_managment.make_tiles_hard_copy(tile_size=args.tile_size)

    print('Data Preparation sequence is Done !')



