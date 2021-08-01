import numpy as np
import os
import re, glob
inference_files = {}

is_herohe = False
exp = 303
fold = 5
target = 'ER'
#dataset = 'CARMEL'
dataset = 'ABCTB_TCGA'

if is_herohe or dataset == 'CARMEL':
    patient_level = False
else:
    patient_level = True
save_csv = True

#patient_level = False

#inference_dir = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_DX_runs\exp' + str(exp) +r'\Inference'
inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4', dataset + '_runs', target, 'exp' + str(exp), 'Inference')
#inference_dir = os.path.join(r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs', 'other', 'exp' + str(exp), 'Inference')

#auto find epochs
file_list = glob.glob(inference_dir+'\*.data')
epochs = [int(re.findall(r"\bModel_Epoch_+\d+\b", os.path.basename(fn))[0][12:]) for fn in file_list]
epochs.sort()

#epochs = [900, 920, 940, 960, 980, 1000]
'''if target == 'ER' or target == 'OR':
    epochs = list(np.arange(900,1001,20))
else:
    epochs = list(np.arange(400, 1001, 100))'''

#epochs = list(np.arange(400, 1001, 100))
#epochs = [1000] #temp
#epochs = list(np.arange(900, 1351, 50)) #temp

key_list = [''.join((target, '_fold', str(fold), '_exp', str(exp), '_epoch', str(epoch), '_test_500')) for epoch in epochs]

if is_herohe:
    val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500_herohe.data')) for epoch in epochs]
else:
    val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']_', target, '-Tiles_500.data')) for epoch in epochs]
#val_list = [''.join(('Model_Epoch_', str(epoch), '-Folds_[', str(fold), ']', '-Tiles_500.data')) for epoch in epochs]
#key_list = [''.join(('exp', str(exp), '_fold', str(fold), '_epoch', str(epoch), '_test_500_herohe')) for epoch in epochs]

inference_files = dict(zip(key_list, val_list))
inference_name = target + '_fold' + str(fold) + '_exp' + str(exp)
