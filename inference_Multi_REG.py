import utils
from torch.utils.data import DataLoader
import torch
import datasets
import nets
import PreActResNets
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import sys
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
#parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=79, help='Use models from this experiment')
parser.add_argument('-ex', '--experiment', type=int, default=79, help='Use models from this experiment') #temp RanS 1.2.21
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=[11210, 12000], help='Use this epoch models for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=50, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[1], help=' folds to infer')
parser.add_argument('--mag', type=int, default=20, help='desired magnification of patches') #RanS 8.2.21
args = parser.parse_args()

args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
data_path = ''

print('Loading pre-saved models:')
models = []
for counter, epoch in enumerate(args.from_epoch):
    print('  Exp. {} and Epoch {}'.format(args.experiment, epoch))
    if counter == 0:  # Basic meta data will be taken from the first model
        output_dir, _, _, TILE_SIZE, _, _, _, _, args.target, _, model_name = utils.run_data(experiment=args.experiment)

        # we need to make some root modifications according to the computer we're running at.
        if sys.platform == 'linux':
            data_path = ''
        elif sys.platform == 'win32':
            output_dir = output_dir.replace(r'/', '\\')
            data_path = os.getcwd()

    # loading basic model type
    model = eval(model_name)
    # loading model parameters from the specific epoch
    model_data_loaded = torch.load(os.path.join(data_path, output_dir, 'Model_CheckPoints',
                                                'model_data_Epoch_' + str(epoch) + '.pt'), map_location='cpu')
    model.load_state_dict(model_data_loaded['model_state_dict'])
    model.eval()
    models.append(model)



TILE_SIZE = 128
tiles_per_iter = 20
if sys.platform == 'linux':
    TILE_SIZE = 256
    tiles_per_iter = 150
elif sys.platform == 'win32':
    TILE_SIZE = 256

inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=tiles_per_iter,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles,
                                  mag=args.mag
                                  )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
NUM_SLIDES = len(inf_dset.valid_slide_indices)

all_targets = []
all_scores, all_labels = np.zeros((NUM_SLIDES, NUM_MODELS)), np.zeros((NUM_SLIDES, NUM_MODELS))
patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))
patch_scores[:] = np.nan
slide_num = 0

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos, correct_neg = [0] * NUM_MODELS, [0] * NUM_MODELS

with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_patches, file_name) in enumerate(tqdm(inf_loader)):
        if new_slide:
            scores_0, scores_1 = [np.zeros(0)] * NUM_MODELS, [np.zeros(0)] * NUM_MODELS
            target_current = target
            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)

        data, target = data.to(DEVICE), target.to(DEVICE)

        for index, model in enumerate(models):
            model.to(DEVICE)

            scores = model(data)

            scores_0[index] = np.concatenate((scores_0[index], scores[:, 0].cpu().detach().numpy()))
            scores_1[index] = np.concatenate((scores_1[index], scores[:, 1].cpu().detach().numpy()))

        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target.cpu().numpy()[0][0])
            if target == 1:
                total_pos += 1
            else:
                total_neg +=1

            for model_num in range(NUM_MODELS):
                current_slide_tile_scores = np.vstack((scores_0[model_num], scores_1[model_num]))

                predicted = current_slide_tile_scores.mean(1).argmax()
                #print('len(scores_1[model_num]):', len(scores_1[model_num])) #temp
                patch_scores[slide_num, model_num, :len(scores_1[model_num])] = scores_1[model_num]
                all_scores[slide_num, model_num] = scores_1[model_num].mean()
                all_labels[slide_num, model_num] = predicted

                if target == 1 and predicted == 1:
                    correct_pos[model_num] += 1
                elif target == 0 and predicted == 0:
                    correct_neg[model_num] += 1

            slide_num += 1

for model_num in range(NUM_MODELS):
    fpr_train, tpr_train, _ = roc_curve(all_targets, all_scores[:, model_num])

    # Save roc_curve to file:
    if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
        os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

    file_name = os.path.join(data_path, output_dir, 'Inference', 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + '.data')
    inference_data = [fpr_train, tpr_train, all_labels[:, model_num], all_targets, all_scores[:, model_num],
                      total_pos, correct_pos[model_num], total_neg, correct_neg[model_num], len(inf_dset), np.squeeze(patch_scores[:, model_num,:])]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)


    print('For model from Epoch {}: {} / {} correct classifications'
          .format(args.from_epoch[model_num],
                  int(len(all_labels[:, model_num]) - np.abs(np.array(all_targets) - np.array(all_labels[:, model_num])).sum()),
                  len(all_labels[:, model_num])))
    print('Done !')
