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
parser.add_argument('-ex', '--experiment', type=int, default=210, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=1320, help='Use this epoch model for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=10, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[1], help=' folds to infer')
args = parser.parse_args()

args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
data_path = ''

print('Loading pre-saved model from Exp. {} and epoch {}'.format(args.experiment, args.from_epoch))
output_dir, _, _, TILE_SIZE, _, _, _, _, args.target, _, model_name = utils.run_data(experiment=args.experiment)

TILE_SIZE = 128
tiles_per_iter = 2
if sys.platform == 'linux':
    TILE_SIZE = 256
    #data_path = '/home/womer/project'
    data_path = '' #RanS 13.1.21
    tiles_per_iter = 150
elif sys.platform == 'win32':
    TILE_SIZE = 256
    output_dir = output_dir.replace(r'/', '\\')
    data_path = os.getcwd()

# Load saved model:
#model = eval(model_name)
model = PreActResNets.PreActResNet50_Ron()
model_data_loaded = torch.load(os.path.join(data_path, output_dir,
                                            'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])

inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=tiles_per_iter,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles
                                  )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

in_between = True

all_scores = []
all_labels = []
all_targets = []

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos, correct_neg = 0, 0

model.eval()
with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_patches, file_name) in enumerate(tqdm(inf_loader)):
        if in_between:
           scores_0, scores_1 = np.zeros(0), np.zeros(0)
           target_current = target
           slide_batch_num = 0
           in_between = False

        data = data.squeeze(0)

        data, target = data.to(DEVICE), target.to(DEVICE)
        model.to(DEVICE)

        scores = model(data)

        outputs = torch.nn.functional.softmax(scores, dim=1)
        scores_0 = np.concatenate((scores_0, outputs[:, 0].cpu().detach().numpy()))
        scores_1 = np.concatenate((scores_1, outputs[:, 1].cpu().detach().numpy()))

        slide_batch_num += 1

        if last_batch:
            in_between = True

            current_slide_tile_scores = np.vstack((scores_0, scores_1))

            predicted = current_slide_tile_scores.mean(1).argmax()

            all_scores.append(scores_1.mean())
            all_labels.append(predicted)  # all_labels.append(0 if all_scores[-1] < 0.5 else 1)
            all_targets.append(target.cpu().numpy()[0][0])

            # Sanity check:
            if target_current != target.cpu().numpy()[0][0]:
                print('Target Error!!')


            if target == 1:
                total_pos += 1
                if all_labels[-1] == 1:
                    correct_pos += 1
            elif target == 0:
                total_neg += 1
                if all_labels[-1] == 0:
                    correct_neg += 1

fpr_train, tpr_train, _ = roc_curve(all_targets, all_scores)

# Save roc_curve to file:
if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

file_name = os.path.join(data_path, output_dir, 'Inference', 'Model_Epoch_' + str(args.from_epoch)
                         + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + '.data')
inference_data = [fpr_train, tpr_train, all_labels, all_targets, all_scores, total_pos, correct_pos, total_neg, correct_neg, len(inf_dset)]

with open(file_name, 'wb') as filehandle:
    pickle.dump(inference_data, filehandle)

print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
print('Done !')
