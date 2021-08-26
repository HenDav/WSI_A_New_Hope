import glob
import os
out_dir = r'/mnt/gipmed_new/Data/Breast/Carmel/Batch_9/CARMEL9'

file_list = glob.glob(out_dir + '/*')

for fn in file_list:
    os.rename(fn, fn + '.mrxs')

print('finished')