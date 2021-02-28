import numpy as np
from random import shuffle
import csv

dataset = 'ABCTB'

if dataset == 'Carmel123':
    N_samples = 1553 #number of slides in file
    test_ratio = 0.25 #percentage to be marked as "test"
    val_ratio = 0 #percentage to be marked as "validation"
    n_folds = 5 #number of cross-validation folds
    out_file = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\All Data\folds.csv'
elif dataset == 'TCGA':
    N_samples = 3113 #number of slides in file
    test_ratio = 0.2 #percentage to be marked as "test"
    val_ratio = 0.2 #percentage to be marked as "validation"
    n_folds = 1 #number of cross-validation folds
    out_file = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\All Data\folds_TCGA.csv'
elif dataset == 'HEROHE':
    N_samples = 510 #number of slides in file
    test_ratio = 0.2 #percentage to be marked as "test"
    val_ratio = 0.2 #percentage to be marked as "validation"
    n_folds = 1 #number of cross-validation folds
    out_file = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\All Data\folds_HEROHE.csv'
elif dataset == 'ABCTB':
    N_samples = 3016 #number of slides in file
    test_ratio = 0.25  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 5  # number of cross-validation folds
    out_file = r'C:\Users\User\Dropbox\Technion work 2020\Code\WSI_MIL\WSI_MIL\All Data\folds_ABCTB.csv'


fold_size = int(N_samples*(1-test_ratio - val_ratio)/n_folds)
N_val = int(N_samples * val_ratio)
N_test = N_samples - N_val - fold_size*n_folds

folds = ['test']*N_test
folds.extend(['val']*N_val)

if n_folds == 1:
    folds.extend(['train']*fold_size)
else:
    for ii in np.arange(1, n_folds + 1):
        folds.extend(list(np.ones(fold_size)*ii))

shuffle(folds)

with open(out_file, 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(folds)