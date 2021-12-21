import pickle
import os

#dirname = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\Her2\exp308\Inference\features'
#dirname = r'C:\Pathnet_results\MIL_general_try4\CAT_runs\Her2\exp412\Inference\features_debug'
dirname = r'C:\Pathnet_results\MIL_general_try4\temp'

fname = 'Model_Epoch_1000-Folds_[1]_Ki67-Tiles_500_features_slides_700.data'

with open(os.path.join(dirname, fname), 'rb') as filehandle:
    inference_data = pickle.load(filehandle)

labels, targets, scores, patch_scores, slide_names, features, slide_datasets, patch_locs = inference_data





print('file successfully read')

