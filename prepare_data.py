import utils_data_managment
import argparse
import os
import glob

parser = argparse.ArgumentParser(description='Data preparation script')

parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='need to collect data from main data file?')
parser.add_argument('--segmentation', dest='segmentation', action='store_true', help='need to make segmentation map for all files?')
parser.add_argument('--grid', dest='grid', action='store_true', help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', dest='stats', action='store_true', help='need to compute statistical data?')
parser.add_argument('--hard_copy', dest='hard_copy', action='store_true', help='make hard copy of tiles?')
#parser.add_argument('--data_folder', type=str, default='All Data/HEROHE', help='location of data folder')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='type of dataset to use (HEROHE/TCGA/LUNG)')
parser.add_argument('--data_root', type=str, default='All Data', help='location of data root folder')
parser.add_argument('--tissue_coverage', type=float, default=0.5, help='min. tissue % for a valid tile') #RanS 26.11.20
parser.add_argument('--sl2im', dest='sl2im', action='store_true', help='convert slides to png images?')
args = parser.parse_args()

if __name__ =='__main__':
    if args.data_collection:
        utils_data_managment.make_slides_xl_file(DataSet=args.dataset, ROOT_DIR=args.data_root)
    if args.segmentation:
        utils_data_managment.make_segmentations(DataSet=args.dataset, ROOT_DIR=args.data_root)
    if args.grid:
        utils_data_managment.make_grid(DataSet=args.dataset, ROOT_DIR=args.data_root, tile_sz=args.tile_size, tissue_coverage=args.tissue_coverage)
    if args.stats:
        utils_data_managment.compute_normalization_values(DataSet=args.dataset, ROOT_DIR=args.data_root)
    if args.hard_copy:
        utils_data_managment.make_tiles_hard_copy(tile_size=args.tile_size)
    if args.sl2im:
        utils_data_managment.herohe_slides2images()

    print('Data Preparation sequence is Done !')



