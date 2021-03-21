import utils_data_managment
import argparse
import os
import glob
from utils import get_cpu

parser = argparse.ArgumentParser(description='Data preparation script')

parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='need to collect data from main data file?')
parser.add_argument('--segmentation', dest='segmentation', action='store_true', help='need to make segmentation map for all files?')
parser.add_argument('--grid', dest='grid', action='store_true', help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', dest='stats', action='store_true', help='need to compute statistical data?')
parser.add_argument('--hard_copy', dest='hard_copy', action='store_true', help='make hard copy of tiles?')
#parser.add_argument('--data_folder', type=str, default='All Data/HEROHE', help='location of data folder')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='type of dataset to use (HEROHE/TCGA/LUNG)')
parser.add_argument('--data_root', type=str, default='All Data', help='location of data root folder')
parser.add_argument('--tissue_coverage', type=float, default=0.5, help='min. tissue % for a valid tile') #RanS 26.11.20
parser.add_argument('--sl2im', dest='sl2im', action='store_true', help='convert slides to png images?')
parser.add_argument('--mag', type=int, default=20, help='desired magnification of patches') #RanS 15.2.21
parser.add_argument('--out_path', type=str, default='', help='path for output files')
parser.add_argument('--added_extension', type=str, default='', help='extension to be added to new slides_data file and Grids path')
parser.add_argument('--SegData_path', type=str, default='', help='extension of the SegData path')
args = parser.parse_args()

num_workers = get_cpu()

if __name__ =='__main__':

    out_path = args.data_root
    if args.out_path != '':
        out_path = args.out_path

    if args.data_collection:
        utils_data_managment.make_slides_xl_file(DataSet=args.dataset, ROOT_DIR=args.data_root, out_path=out_path)
    if args.segmentation:
        utils_data_managment.make_segmentations(DataSet=args.dataset, ROOT_DIR=args.data_root, out_path=out_path)
    if args.grid:
        '''utils_data_managment.make_grid(DataSet=args.dataset, ROOT_DIR=args.data_root, tile_sz=args.tile_size,
                                       tissue_coverage=args.tissue_coverage, desired_mag=args.mag, out_path=out_path)'''

        utils_data_managment.make_grid(DataSet=args.dataset,
                                       ROOT_DIR=args.data_root,
                                       tile_sz=args.tile_size,
                                       tissue_coverage=args.tissue_coverage,
                                       desired_magnification=args.mag,
                                       added_extension=args.added_extension,
                                       different_SegData_path_extension=args.SegData_path,
                                       num_workers=num_workers)
    if args.stats:
        utils_data_managment.compute_normalization_values(DataSet=args.dataset, ROOT_DIR=args.data_root)
    if args.hard_copy:
        utils_data_managment.make_tiles_hard_copy(tile_size=args.tile_size)
    if args.sl2im:
        utils_data_managment.herohe_slides2images()

    utils_data_managment.make_grid(DataSet='HEROHE',
                                   ROOT_DIR='All Data',
                                   tile_sz=128,
                                   tissue_coverage=0.5,
                                   desired_magnification=10,
                                   added_extension='_new-Cov_05',
                                   different_SegData_path_extension='')


    print('Data Preparation sequence is Done !')
