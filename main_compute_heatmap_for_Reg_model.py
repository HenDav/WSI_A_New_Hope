"""
This is a run file which is used to compute a Heat Map that shows which pixels are mostly used for computing a score
for a slide.
The computation uses a trained model and works.
"""

import PreActResNets
from datasets import Full_Slide_Inference_Dataset
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
run_data_output = utils.run_data(experiment=args.experiment)
args.output_dir, args.test_fold, args.dataset, args.target, args.model_name, args.mag =\
    run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Dataset Name'],\
    run_data_output['Receptor'], run_data_output['Model Name'], run_data_output['Desired Slide Magnification']
# loading basic model type
model = eval(args.model_name)
model_data_loaded = torch.load(os.path.join(args.output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()
model.is_HeatMap = True
model.to(DEVICE)

TILE_SIZE = 128
args.dataset = 'TCGA'

if sys.platform == 'linux':
    TILE_SIZE = 1024


inf_dset = Full_Slide_Inference_Dataset(DataSet=args.dataset,
                                        tile_size=TILE_SIZE,
                                        tiles_per_iter=1,
                                        target_kind=args.target,
                                        folds=args.test_fold,
                                        desired_slide_magnification=args.mag
                                        )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

NUM_SLIDES = len(inf_dset.image_file_names)

all_targets = []
all_scores, all_labels = np.zeros((NUM_SLIDES)), np.zeros((NUM_SLIDES))
#patch_scores = np.empty((NUM_SLIDES, args.num_tiles))
#patch_scores[:] = np.nan
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
slide_num = 0
# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0

new_slide = True
with torch.no_grad():
    for batch_idx, MiniBatch_Dict in enumerate(tqdm(inf_loader)):
        # Unpacking the data:
        data = MiniBatch_Dict['Data']
        target = MiniBatch_Dict['Label']
        last_batch = MiniBatch_Dict['Is Last Batch']
        slide_file = MiniBatch_Dict['Slide Filename']

        if new_slide:
            new_slide = False
            n_tiles = inf_loader.dataset.num_tiles[slide_num]

        data = data.squeeze(0)
        #print("data.shape: ", str(data.shape)) #temp RanS 22.7.21
        data, target = data.to(DEVICE), target.to(DEVICE)

        tile_sized_heatmap = model(data)

        print()

