import os
import re, glob
inference_files = {}

is_herohe = False
exp = '_g10'
fold = '1'
target = 'PR'
dataset = 'CAT'
subdir = ''
#subdir = '6slides'
#subdir = 'CARMEL'
#subdir = 'test_inference_aug21_TCGA_corrections'
is_other = False

if is_herohe or dataset == 'CARMEL' or dataset == 'CAT' or subdir == 'CARMEL':
    patient_level = False
else:
    patient_level = True
save_csv = True

#patient_level = False #temp

if is_other:
    inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4', dataset + '_runs', 'other', 'exp' + str(exp), 'Inference')
else:
    inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4', dataset + '_runs', target, 'exp' + str(exp), 'Inference')


inference_dir = os.path.join(inference_dir, subdir)
#auto find epochs
file_list = glob.glob(inference_dir+'\*.data')
epochs = [int(re.findall(r"\bModel_Epoch_+\d+\b", os.path.basename(fn))[0][12:]) for fn in file_list]
epochs.sort()

key_list = [''.join((target, '_fold', str(fold), '_exp', str(exp), '_epoch', str(epoch), '_test_500')) for epoch in epochs]

if is_herohe:
    val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500_herohe.data')) for epoch in epochs]
else:
    val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500.data')) for epoch in epochs]
#val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']', '-Tiles_500.data')) for epoch in epochs]
#key_list = [''.join(('exp', str(exp), '_fold', str(fold), '_epoch', str(epoch), '_test_500_herohe')) for epoch in epochs]

inference_files = dict(zip(key_list, val_list))
inference_name = target + '_fold' + str(fold) + '_exp' + str(exp)
