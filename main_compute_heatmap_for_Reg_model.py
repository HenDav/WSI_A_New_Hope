"""
This is a run file which is used to compute a Heat Map that shows which pixels are mostly used for computing a score
for a slide.
The computation uses a trained model and works.
"""

import PreActResNets
from datasets import Infer_Dataset
import torch
from torch.utils.data import DataLoader
import utils
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='WSI_REG Slide Heat Map Production')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=241, help='Use model from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=1395, help='Use this epoch model for inference')
#parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
#parser.add_argument('-f', '--folds', type=list, nargs="+", default=[1], help=' folds to infer')
args = parser.parse_args()

if sys.platform == 'darwin':
    args.experiment = 321
    args.from_epoch = 1400

DEVICE = utils.device_gpu_cpu()

print('Loading trained model...')
args.output_dir, args.test_fold, _, _, _, _, _, args.dataset, args.target, _, args.model_name, args.mag = utils.run_data(experiment=args.experiment)

# loading basic model type
model = eval(args.model_name)
model_data_loaded = torch.load(os.path.join(args.output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()
model.is_HeatMap = True

TILE_SIZE = 128
args.dataset = 'TCGA'

if sys.platform == 'linux':
    TILE_SIZE = 1024


inf_dset = Infer_Dataset(DataSet=args.dataset,
                         tile_size=TILE_SIZE,
                         tiles_per_iter=1,
                         target_kind=args.target,
                         folds=args.test_fold,
                         num_tiles=args.num_tiles,
                         desired_slide_magnification=args.mag
                         )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_SLIDES = len(inf_dset.image_file_names)
print('NUM_SLIDES: ', str(NUM_SLIDES))

all_targets = []
all_scores, all_labels = np.zeros((NUM_SLIDES)), np.zeros((NUM_SLIDES))
patch_scores = np.empty((NUM_SLIDES, args.num_tiles))
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
patch_scores[:] = np.nan
slide_num = 0
# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0

with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, _, slide_file, patient) in enumerate(tqdm(inf_loader)):
        if new_slide:
            n_tiles = inf_loader.dataset.num_tiles[slide_num]  # RanS 1.7.21
            #scores_0, scores_1 = [np.zeros(0)] * NUM_MODELS, [np.zeros(0)] * NUM_MODELS
            #scores_0, scores_1 = [np.zeros(n_tiles)] * NUM_MODELS, [np.zeros(n_tiles)] * NUM_MODELS #RanS 1.7.21
            scores_0 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)] # RanS 12.7.21
            scores_1 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)]  # RanS 12.7.21
            if args.save_features:
                #feature_arr = [np.zeros((n_tiles, 512))] * NUM_MODELS #RanS 1.7.21
                feature_arr = [np.zeros((n_tiles, 512)) for ii in range(NUM_MODELS)]  # RanS 1.7.21
            target_current = target
            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)
        #print("data.shape: ", str(data.shape)) #temp RanS 22.7.21
        data, target = data.to(DEVICE), target.to(DEVICE)



