import nets_mil
import datasets
import utils
import sys
import argparse
from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import pickle
import random
random.seed(0)
parser = argparse.ArgumentParser(description='WSI_MIL Slide inference with visualization')
parser.add_argument('-ex', '--experiment', type=int, default=285, help='model to use')
parser.add_argument('-fe', '--from_epoch', type=int, default=995, help='model to use')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[1], help=' folds to infer')
args = parser.parse_args()



args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
data_path = ''

if sys.platform == 'darwin':
    args.experiment = 1
    args.from_epoch = 1035

# Load saved model:
print('Loading pre-saved model from Exp. {} and Epoch {}'.format(args.experiment, args.from_epoch))
run_data_output = utils.run_data(experiment=args.experiment)
output_dir, TILE_SIZE, args.target, model_name, desired_magnification =\
    run_data_output['Location'], run_data_output['Tile Size'], run_data_output['Receptor'], run_data_output['Model Name'],\
    run_data_output['Desired Slide Magnification']
model = eval(model_name)

# loading model parameters from the specific epoch
model_data_loaded = torch.load(os.path.join(output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()
model.infer = True
model.features_part = True

if sys.platform == 'darwin':
    TILE_SIZE = 128

inf_dset = datasets.Full_Slide_Inference_Dataset(DataSet=args.dataset,
                                                 tile_size=TILE_SIZE,
                                                 tiles_per_iter=20,
                                                 target_kind=args.target,
                                                 folds=args.folds,
                                                 desired_slide_magnification=desired_magnification)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

NUM_SLIDES = len(inf_dset.valid_slide_indices)

new_slide = True

all_targets = []
all_scores_for_class_1, all_labels = np.zeros(NUM_SLIDES), np.zeros(NUM_SLIDES)
# all_weights_after_sftmx = np.zeros((NUM_SLIDES, args.num_tiles, NUM_MODELS))
slide_num = 0

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos, correct_neg = 0, 0

# Open folders to save the data:
if not os.path.isdir(os.path.join(data_path, output_dir, 'Full Slides Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Full Slides Inference'))
if not os.path.isdir(
        os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch))):
    os.mkdir(os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch)))

with torch.no_grad():
    for batch_idx, (data, target, _, last_batch, num_tiles, slide_name, eq_grid_locs, is_tissue_tiles, equivalent_grid_size) in enumerate(tqdm(inf_loader)):
        '''for batch_idx in range(4965, 4969): ####1127,1129
        data, target, _, last_batch, num_tiles, slide_name, eq_grid_locs, is_tissue_tiles, equivalent_grid_size = inf_dset[batch_idx] ####'''

        if new_slide:
            equivalent_slide_heat_map = np.zeros((equivalent_grid_size)) # This heat map should be filled with the weights.
            equivalent_slide_grid_locs = []
            scores_0, scores_1 = np.zeros(0), np.zeros(0)
            target_current = target
            slide_batch_num = 0
            new_slide = False

            all_features = np.zeros([num_tiles, model.M], dtype='float32')
            all_weights_before_sftmx = np.zeros([1, num_tiles], dtype='float32')

        '''sz = data.shape  ####
        data = data.reshape(1, sz[0], sz[1], sz[2], sz[3])  ####
        print(batch_idx, last_batch, data.shape) ####'''
        data, target = data.to(DEVICE), target.to(DEVICE)

        model.to(DEVICE)
        features, weights_before_sftmx = model(data)
        num_samples = data.shape[1]

        all_features[slide_batch_num * num_samples: (slide_batch_num + 1) * num_samples, :] = features.detach().cpu().numpy()
        all_weights_before_sftmx[:, slide_batch_num * num_samples: (slide_batch_num + 1) * num_samples] = weights_before_sftmx.detach().cpu().numpy()

        '''all_features[slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter, :] = features.detach().cpu().numpy()
        all_weights_before_sftmx[:, slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter] = weights_before_sftmx.detach().cpu().numpy()'''

        equivalent_slide_grid_locs.extend(eq_grid_locs)
        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target.item())

            if target.item() == 1:
                total_pos += 1
            elif target.item() == 0:
                total_neg += 1

            all_features, all_weights_before_sftmx = torch.from_numpy(all_features).to(DEVICE), torch.from_numpy(all_weights_before_sftmx).to(DEVICE)

            model.features_part = False

            scores, weights = model(data, all_features, all_weights_before_sftmx)
            model.features_part = True

            # Fill the "equivalent_slide_heat_map":
            for idx_weight, loc in enumerate(equivalent_slide_grid_locs):
                equivalent_slide_heat_map[loc[0], loc[1]] = weights[:, idx_weight]

            # Save the heat map to file
            file_name = os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch),
                                     slide_name[0].split('/')[-1] + '_HeatMap.data')
            with open(file_name, 'wb') as filehandle:
                pickle.dump(equivalent_slide_heat_map, filehandle)

            scores = torch.nn.functional.softmax(scores, dim=1)
            _, predicted = scores.max(1)

            scores_0 = np.concatenate((scores_0, scores[:, 0].cpu().detach().numpy()))
            scores_1 = np.concatenate((scores_1, scores[:, 1].cpu().detach().numpy()))

            if target.item() == 1 and predicted.item() == 1:
                correct_pos += 1
            elif target.item() == 0 and predicted.item() == 0:
                correct_neg += 1

            # all_weights_after_sftmx[slide_num, :, model_num] = weights.detach().cpu().numpy()
            all_scores_for_class_1[slide_num] = scores_1
            all_labels[slide_num] = predicted.item()

            slide_num += 1

# Computing performance data for all models (over all slides scores data):

fpr, tpr, _ = roc_curve(all_targets, all_scores_for_class_1)
roc_auc = auc(fpr, tpr)

# Save roc_curve to file:
file_name = os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch)
                         + '-Folds_' + str(args.folds) + '-  All_Tiles.data')

inference_data = [fpr, tpr, all_labels, all_targets, all_scores_for_class_1, total_pos,
                  correct_pos, total_neg, correct_neg, len(inf_dset)]

with open(file_name, 'wb') as filehandle:
    pickle.dump(inference_data, filehandle)

experiment = args.experiment
print('For model from Experiment {} and Epoch {}: {} / {} correct classifications'
      .format(experiment,
              args.from_epoch,
              int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()),
              len(all_labels)))
print('AUC = {} '.format(roc_auc))

print('Done !')