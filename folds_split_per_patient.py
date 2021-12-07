import numpy as np
from random import shuffle
import csv
import pandas as pd

dataset = 'HEROHE'
patient_column_name = 'patient barcode' #default

if dataset == 'Carmel123':
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
    in_file = r'C:\ran_data\HEROHE_examples\slides_data_HEROHE.xlsx'
    N_samples = 510 #number of slides in file
    test_ratio = 0 #percentage to be marked as "test"
    val_ratio = 0 #percentage to be marked as "validation"
    n_folds = 4 #number of cross-validation folds
    out_file = r'C:\ran_data\HEROHE_examples\slides_data_HEROHE_folds.xlsx'
elif dataset == 'ABCTB':
    in_file = r'C:\ran_data\ABCTB\ABCTB_examples\slides_data_full.xlsx'
    test_ratio = 0.25  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 5  # number of cross-validation folds
    out_file = r'C:\ran_data\ABCTB\ABCTB_examples\slides_data_full_folds.xlsx'
elif dataset == 'TCGA_LUNG':
    in_file = r'C:\ran_data\TCGA_lung\slides_data_TCGA_LUNG.xlsx'
    test_ratio = 0  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 5  # number of cross-validation folds
    out_file = r'C:\ran_data\TCGA_lung\slides_data_full_folds.xlsx'
elif dataset == 'LEUKEMIA':
    in_file = r'C:\ran_data\BoneMarrow\slides_data_LEUKEMIA.xlsx'
    test_ratio = 0  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 5  # number of cross-validation folds
    out_file = r'C:\ran_data\BoneMarrow\slides_data_LEUKEMIA_folds.xlsx'
    patient_column_name = 'PatientID'
elif dataset == 'Ipatimup':
    in_file = r'C:\ran_data\Covilha+Ipatimup\slides_data_Ipatimup.xlsx'
    test_ratio = 0  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 4  # number of cross-validation folds
    out_file = r'C:\ran_data\Covilha+Ipatimup\slides_data_Ipatimup_folds.xlsx'
    patient_column_name = 'ID'
elif dataset == 'Covilha':
    in_file = r'C:\ran_data\Covilha+Ipatimup\slides_data_Covilha.xlsx'
    test_ratio = 0  # percentage to be marked as "test"
    val_ratio = 0  # percentage to be marked as "validation"
    n_folds = 4  # number of cross-validation folds
    out_file = r'C:\ran_data\Covilha+Ipatimup\slides_data_Covilha_folds.xlsx'
    patient_column_name = 'ID'
else:
    raise IOError('dataset unknown')

slides_data = pd.read_excel(in_file)
patients = np.unique(slides_data[patient_column_name])
N_patients = patients.shape[0]

fold_size = int(N_patients*(1-test_ratio - val_ratio)/n_folds)
N_val = int(N_patients * val_ratio)
N_test = N_patients - N_val - fold_size*n_folds

folds = ['test']*N_test

#RanS 2.12.21
if test_ratio == 0: #replace test values with random folds
    folds = list(np.random.randint(1, n_folds+1, size=N_test))

folds.extend(['val']*N_val)

if n_folds == 1:
    folds.extend(['train']*fold_size)
else:
    for ii in np.arange(1, n_folds + 1):
        folds.extend(list(np.ones(fold_size, dtype=int)*ii))

shuffle(folds)

patients_folds_df = pd.DataFrame({'patient': patients, 'fold_new': folds})
#slides_data['new fold'] = 'Missing Data'
#slides_data['new fold']

slides_data = slides_data.merge(right=patients_folds_df, left_on=patient_column_name, right_on='patient', how='outer')
slides_data.to_excel(out_file)

print('finished')
