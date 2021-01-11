import numpy as np
from random import shuffle
import csv

N_samples = 1553 #number of slides in file
test_ratio = 0.25 #percentage to be marked as "test"
n_folds = 5 #number of cross-validation folds
out_file = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\All Data\folds.csv'


fold_size = int(N_samples*(1-test_ratio)/n_folds)
N_test = N_samples - fold_size*n_folds

folds = ['test']*N_test
for ii in np.arange(1, n_folds + 1):
    folds.extend(list(np.ones(fold_size)*ii))
shuffle(folds)

with open(out_file, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(folds)