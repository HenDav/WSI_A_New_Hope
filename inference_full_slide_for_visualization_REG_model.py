import PreActResNets
from datasets import Batched_Full_Slide_Inference_Dataset
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
from pathlib import Path
import pandas as pd

random.seed(0)
parser = argparse.ArgumentParser(description='WSI_MIL Slide inference with visualization for Regular model')
parser.add_argument('-ex', '--experiment', type=int, default=293, help='model to use')
parser.add_argument('-fe', '--from_epoch', type=int, default=1000, help='model to use')
args = parser.parse_args()

DEVICE = utils.device_gpu_cpu()
data_path = ''

if sys.platform == 'darwin':
    args.experiment = 321
    args.from_epoch = 1400

# Load saved model:
print('Loading pre-saved model from Exp. {} and Epoch {}'.format(args.experiment, args.from_epoch))
output_dir, _, _, TILE_SIZE, _, _, _, _, args.target, _, model_name, desired_magnification = utils.run_data(experiment=args.experiment)
model = eval(model_name)

# loading model parameters from the specific epoch
model_data_loaded = torch.load(os.path.join(output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()

if sys.platform == 'darwin':
    TILE_SIZE = 256

inf_dset = Batched_Full_Slide_Inference_Dataset(tile_size=TILE_SIZE,
                                                tiles_per_iter=100,
                                                target_kind=args.target)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

NUM_SLIDES = len(inf_dset.slides)

new_slide = True

all_targets = []
all_scores_for_class_1, all_labels = np.zeros(NUM_SLIDES), np.zeros(NUM_SLIDES)
# all_weights_after_sftmx = np.zeros((NUM_SLIDES, args.num_tiles, NUM_MODELS))
slide_num = 0

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos, correct_neg = 0, 0

# Create folders to save the data:
path_for_output = 'Inference/Full_Slide_Inference/' + inf_dset.DataSet
Path(path_for_output).mkdir(parents=True, exist_ok=True)


original_x_dict, original_y_dict, score_dict = {}, {}, {}
new_x_dict, new_y_dict = {}, {}
slide_size_dict = {}
'''
if not os.path.isdir('Inference'):
    os.mkdir('Inference')
if not os.path.isdir(os.path.join(data_path, output_dir, 'Full Slides Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Full Slides Inference'))
if not os.path.isdir(
        os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch))):
    os.mkdir(os.path.join(data_path, output_dir, 'Full Slides Inference', 'Model_Epoch_' + str(args.from_epoch)))
'''
with torch.no_grad():
    for batch_idx, minibatch in enumerate(tqdm(inf_loader)):
        data = minibatch['Data']
        target = minibatch['Target']
        last_batch = minibatch['Is Last Batch']
        num_tiles = minibatch['Initial Num Tiles']
        slide_name = minibatch['Slide Filename']
        eq_grid_locs = minibatch['Equivalent Grid']
        is_tissue_tiles = minibatch['Is Tissue Tiles']
        equivalent_grid_size = minibatch['Equivalent Grid Size']
        original_locations = minibatch['Level 0 Locations']
        slide_dims = minibatch['Slide Dimensions']

        if new_slide:
            print('Working on Slide {}'.format(slide_name))
            equivalent_slide_heat_map = np.ones((equivalent_grid_size)) * (-1)  # This heat map should be filled with the weights.
            equivalent_slide_grid_locs, original_slide_grid_locs = [], []
            scores_1 = np.zeros(0)
            target_current = target
            #slide_batch_num = 0
            new_slide = False

        if target != target_current:
            raise Exception('Slide Target changed during inference...')

        data = data.to(DEVICE)
        # target = target.to(DEVICE)

        model.to(DEVICE)
        output, _ = model(data)
        num_samples = data.shape[1]

        scores = torch.nn.functional.softmax(output, dim=1)[:, 1].cpu().detach().numpy()
        scores_1 = np.concatenate((scores_1, scores))

        equivalent_slide_grid_locs.extend(eq_grid_locs)
        original_slide_grid_locs.extend(original_locations)
        #slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target_current.item())
            '''
            if target.item() == 1:
                total_pos += 1
            elif target.item() == 0:
                total_neg += 1
            '''

            # Fill the "equivalent_slide_heat_map" and the various data dicts:
            slide_size_dict[slide_name[0]] = {'Height': slide_dims['Height'].item(),
                                              'Width': slide_dims['Width'].item()}

            y_coordinates, x_coordinates, new_y_coordinates, new_x_coordinates, tile_scores = [], [], [], [], []
            for idx_score, loc in enumerate(equivalent_slide_grid_locs):
                #print(equivalent_slide_heat_map.shape, loc[0], loc[1])
                equivalent_slide_heat_map[loc[0], loc[1]] = scores_1[idx_score]

                y_coordinates.append(original_slide_grid_locs[idx_score][0].item())
                x_coordinates.append(original_slide_grid_locs[idx_score][1].item())
                new_y_coordinates.append(loc[0].item())
                new_x_coordinates.append(loc[1].item())
                tile_scores.append(scores_1[idx_score])

            original_y_dict[slide_name[0]] = pd.Series(y_coordinates)
            original_x_dict[slide_name[0]] = pd.Series(x_coordinates)
            new_y_dict[slide_name[0]] = pd.Series(new_y_coordinates)
            new_x_dict[slide_name[0]] = pd.Series(new_x_coordinates)
            score_dict[slide_name[0]] = pd.Series(tile_scores)

            # Save the heat map to file
            basic_file_name = os.path.join(path_for_output,
                                           '_'.join([args.target, 'BatchOfSlides', 'Exp', str(args.experiment),
                                                     'Epoch', str(args.from_epoch), 'Inference_Full_Slide']))
            file_name = os.path.join(basic_file_name + '_' + slide_name[0] + '_ScoreHeatMap.xlsx')
            pd.DataFrame(equivalent_slide_heat_map).to_excel(file_name)

            '''
            if target.item() == 1 and predicted.item() == 1:
                correct_pos += 1
            elif target.item() == 0 and predicted.item() == 0:
                correct_neg += 1
            

            all_scores_for_class_1[slide_num] = scores_1
            all_labels[slide_num] = predicted.item()
            '''
            slide_num += 1

# Saving all tile score data:
pd.DataFrame(original_y_dict).transpose().to_csv(basic_file_name + '_y.csv')
pd.DataFrame(original_x_dict).transpose().to_csv(basic_file_name + '_x.csv')
pd.DataFrame(new_y_dict).transpose().to_csv(basic_file_name + '_new_y.csv')
pd.DataFrame(new_x_dict).transpose().to_csv(basic_file_name + '_new_x.csv')
pd.DataFrame(score_dict).transpose().to_csv(basic_file_name + '_tile_scores.csv')
pd.DataFrame(slide_size_dict).transpose().to_csv(basic_file_name + '_slide_dimensions.csv')

# Computing performance data for all models (over all slides scores data):
'''
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
'''
print('Done !')