import os
import pandas as pd
import argparse
from glob import glob
from shutil import copyfile
import numpy as np
import os, shutil

# multiply the slides in the input folder, in order to simulate a large dataset

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copyfile(s, d)

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('--in_dir', type=str, default=r'/mnt/gipmed_new/Data/Breast/ABCTB/mrxs_50test_temp/ABCTB - Copy', help='input dir')
#parser.add_argument('--in_dir', type=str, default=r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB', help='input dir')

args = parser.parse_args()
in_dir = args.in_dir

slides_data_file = glob(os.path.join(in_dir, 'slides_data_*.xlsx'))
if len(slides_data_file) > 1:
    raise IOError('more than one slides_data file')
elif len(slides_data_file) == 0:
    raise IOError('no slides_data file found')
else:
    slides_data_DF = pd.read_excel(slides_data_file[0], engine='openpyxl')

grid_data_file = os.path.join(in_dir, 'Grids_10', 'Grid_data.xlsx')
if not os.path.isfile(grid_data_file):
    raise IOError('no grid file found')
else:
    grid_data_DF = pd.read_excel(grid_data_file, engine='openpyxl')

new_names = np.arange(1, 10)
slides_data_DF_new = slides_data_DF
grid_data_DF_new = grid_data_DF
for ind, row in slides_data_DF.iterrows():
    print('processing ' + row['file'])
    fn = os.path.join(in_dir, row['file'])
    base_name, file_extension = os.path.splitext(row['file'])

    dn = os.path.join(in_dir, base_name)
    grid_name = os.path.join(in_dir, 'Grids_10', base_name + '--tlsz256.data')
    #if slide file and folder exist:
    if os.path.isfile(fn) and (file_extension != '.mrxs' or os.path.isdir(dn)):
        for ind in new_names:
            new_dir_name = dn + '_' + str(ind)
            new_file_name = new_dir_name + file_extension
            new_grid_file_name = os.path.join(in_dir, 'Grids_10', base_name + '_' + str(ind) + '--tlsz256.data')
            copyfile(fn, new_file_name)
            copyfile(grid_name, new_grid_file_name)
            if file_extension == '.mrxs':
                os.mkdir(new_dir_name)
                copytree(dn, new_dir_name)
            new_row = row
            new_row['file'] = base_name + '_' + str(ind) + file_extension
            slides_data_DF_new = slides_data_DF_new.append(new_row)
            grid_row = grid_data_DF.loc[grid_data_DF['file'] == row['file']]
            grid_data_DF_new = grid_data_DF_new.append(grid_row)
        print("Source path renamed to destination path successfully.")

slides_data_DF_new.to_excel(os.path.join(in_dir, 'slides_data_new.xlsx'))
grid_data_DF_new.to_excel(os.path.join(in_dir, 'Grids_10', 'Grid_data_new.xlsx'))
print('finished')
