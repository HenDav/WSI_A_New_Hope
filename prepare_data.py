import utils_data_managment
import argparse
import os
from utils import get_cpu
import smtplib, ssl

parser = argparse.ArgumentParser(description='Data preparation script')

parser.add_argument('--data_collection', dest='data_collection', action='store_true', help='need to collect data from main data file?')
parser.add_argument('--segmentation', dest='segmentation', action='store_true', help='need to make segmentation map for all files?')
parser.add_argument('--grid', dest='grid', action='store_true', help='need to make grid coordinates for all files?')
parser.add_argument('--tile_size', type=int, default=256, help='size of tiles')
parser.add_argument('--stats', dest='stats', action='store_true', help='need to compute statistical data?')
parser.add_argument('--hard_copy', dest='hard_copy', action='store_true', help='make hard copy of tiles?')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='type of dataset to use (HEROHE/TCGA/LUNG)')
parser.add_argument('--data_root', type=str, default='All Data', help='location of data root folder')
parser.add_argument('--tissue_coverage', type=float, default=0.3, help='min. tissue % for a valid tile')
parser.add_argument('--sl2im', dest='sl2im', action='store_true', help='convert slides to png images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--out_path', type=str, default='', help='path for output files')
parser.add_argument('--added_extension', type=str, default='', help='extension to be added to new slides_data file and Grids path')
parser.add_argument('--SegData_path', type=str, default='', help='extension of the SegData path')
parser.add_argument('--oversized_HC_tiles', action='store_true', help='create larger tiles to support random shift')
parser.add_argument('--as_jpg', action='store_true', help='save tiles as jpg')
args = parser.parse_args()

num_workers = get_cpu()

if __name__ =='__main__':

    out_path = args.data_root
    if args.out_path != '':
        out_path = args.out_path

    if args.data_collection:
        utils_data_managment.make_slides_xl_file(DataSet=args.dataset, ROOT_DIR=args.data_root, out_path=out_path)
    if args.segmentation:
        utils_data_managment.make_segmentations(DataSet=args.dataset,
                                                ROOT_DIR=args.data_root,
                                                out_path=out_path,
                                                num_workers=num_workers)
    if args.grid:
        utils_data_managment.make_grid(DataSet=args.dataset,
                                       ROOT_DIR=args.data_root,
                                       tile_sz=args.tile_size,
                                       tissue_coverage=args.tissue_coverage,
                                       desired_magnification=args.mag,
                                       added_extension=args.added_extension,
                                       different_SegData_path_extension=args.SegData_path,
                                       num_workers=num_workers)
    if args.stats:
        utils_data_managment.compute_normalization_values(DataSet=args.dataset,
                                                          ROOT_DIR=args.data_root,
                                                          tile_size=args.tile_size)
    if args.hard_copy:
        utils_data_managment.make_tiles_hard_copy(DataSet=args.dataset,
                                                  ROOT_DIR=args.data_root,
                                                  tile_sz=args.tile_size,
                                                  num_tiles=-1,
                                                  desired_magnification=args.mag,
                                                  added_extension=args.added_extension,
                                                  num_workers=num_workers,
                                                  oversized_HC_tiles=args.oversized_HC_tiles,
                                                  as_jpg=args.as_jpg)
    if args.sl2im:
        utils_data_managment.herohe_slides2images()


    print('Data Preparation sequence is Done !')

    # finished training, send email if possible
    if os.path.isfile('mail_cfg.txt'):
        with open("mail_cfg.txt", "r") as f:
            text = f.readlines()
            receiver_email = text[0][:-1]
            password = text[1]

        port = 465  # For SSL
        sender_email = "gipmed.python@gmail.com"

        message = 'Subject: finished running prepare_data'

        # Create a secure SSL context
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
            print('email sent to ' + receiver_email)

    utils_data_managment.make_grid(DataSet='TCGA',
                                   tile_sz=256,
                                   tissue_coverage=0.5,
                                   num_workers=2
                                   )