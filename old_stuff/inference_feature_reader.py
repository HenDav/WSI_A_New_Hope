import pickle
import os

dirname = r'C:\Pathnet_results\MIL_general_try4\ABCTB_TCGA_runs\ER\exp293\Inference'
fname = 'Model_Epoch_1000-Folds_[1]_ER-Tiles_500_features_slides_100.data'

with open(os.path.join(dirname, fname), 'rb') as filehandle:
    inference_data = pickle.load(filehandle)

labels, targets, scores, patch_scores, slide_names, features = inference_data

print('file successfully read')

