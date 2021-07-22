import numpy as np
from random import shuffle
import csv
import pandas as pd

dataset = 'CARMEL1-8'

if dataset == 'CARMEL1-8':
    N_samples = 2272
    test_ratio = 0 #percentage to be marked as "test"
    val_ratio = 0 #percentage to be marked as "validation"
    n_folds = 5 #number of cross-validation folds
elif dataset == 'Carmel123':
    N_samples = 1553 #number of slides in file
    test_ratio = 0.25 #percentage to be marked as "test"
    val_ratio = 0 #percentage to be marked as "validation"
    n_folds = 5 #number of cross-validation folds
    out_file = r'/All Data/folds.csv'
elif dataset == 'TCGA':
    N_samples = 3113 #number of slides in file
    test_ratio = 0.2 #percentage to be marked as "test"
    val_ratio = 0.2 #percentage to be marked as "validation"
    n_folds = 1 #number of cross-validation folds
    out_file = r'/All Data/folds_TCGA.csv'
elif dataset == 'HEROHE':
    N_samples = 510 #number of slides in file
    test_ratio = 0.2 #percentage to be marked as "test"
    val_ratio = 0.2 #percentage to be marked as "validation"
    n_folds = 1 #number of cross-validation folds
    out_file = r'/All Data/folds_HEROHE.csv'
elif dataset == 'ABCTB':
    N_samples = 3016 #number of slides in file
    test_ratio = 0.25  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 5  # number of cross-validation folds
    out_file = r'/All Data/folds_ABCTB.csv'


fold_size = int(N_samples*(1-test_ratio - val_ratio)/n_folds)
N_val = int(N_samples * val_ratio)
if test_ratio==0:
    #enlarge the last fold
    last_fold_size = N_samples - fold_size*(n_folds-1)
    N_test = 0
else:
    N_test = N_samples - N_val - fold_size*n_folds
    last_fold_size = fold_size

folds = ['test']*N_test
folds.extend(['val']*N_val)

if n_folds == 1:
    folds.extend(['train']*fold_size)
else:
    #for ii in np.arange(1, n_folds + 1):
    for ii in np.arange(1, n_folds):
        folds.extend(list(np.ones(fold_size)*ii))
    folds.extend(list(np.ones(last_fold_size) * n_folds))

shuffle(folds)

if dataset=='CARMEL1-8':
    folds_df = pd.DataFrame({'PatientIndex': np.arange(1,N_samples+1), 'folds': folds})
    # write folds per patient to slides_data file
    for ii in range(1, 9):
        fn = r'C:\ran_data\Carmel_Slides_examples\folds_labels\slides_data_CARMEL' + str(ii) + '.xlsx'
        fout = r'C:\ran_data\Carmel_Slides_examples\folds_labels\slides_data_CARMEL' + str(ii) + '_folds.xlsx'
        meta_data_DF = pd.read_excel(fn)
        print('finished ' + str(ii))
        meta_data_DF = meta_data_DF.merge(folds_df, on='PatientIndex', how='left')
        meta_data_DF.to_excel(fout)
else:
    #write folds per slide to file
    with open(out_file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(folds)

print('finished')