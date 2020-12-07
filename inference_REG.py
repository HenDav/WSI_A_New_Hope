import utils
from torch.utils.data import DataLoader
import torch
from nets import PreActResNet50
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import sys
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', type=int, default=79, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=12000, help='Use this epoch model for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=5, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[2], help=' folds to infer')
### parser.add_argument('-ev', dest='eval', action='store_true', help='Use eval mode (or train mode')
args = parser.parse_args()

args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
data_path = ''

# Load saved model:
model = PreActResNet50()

print('Loading pre-saved model from Exp. {} and epoch {}'.format(args.experiment, args.from_epoch))
output_dir, _, _, TILE_SIZE, _, _, _ = utils.run_data(experiment=args.experiment)

TILE_SIZE = 128
if sys.platform == 'linux':
    TILE_SIZE = 256
    data_path = '/home/womer/project'

model_data_loaded = torch.load(os.path.join(data_path, output_dir,
                                            'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

model.load_state_dict(model_data_loaded['model_state_dict'])


inf_dset = utils.Infer_WSI_MILdataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      tiles_per_iter=150,
                                      folds=args.folds,
                                      print_timing=True,
                                      DX=False,
                                      num_tiles=args.num_tiles)
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

in_between = True

all_scores = []
all_labels = []
all_targets = []

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
true_pos, true_neg = 0, 0

model.eval()
with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_patches) in enumerate(tqdm(inf_loader)):
        if in_between:
            scores_1 = np.zeros(0)
            target_current = target
            slide_batch_num = 0
            in_between = False

        data, target = data.to(DEVICE), target.to(DEVICE)
        model.to(DEVICE)

        out = model(data)
        outputs = torch.nn.functional.softmax(out, dim=1)

        ### score_current, _ = outputs.max(1)
        scores_1 = np.concatenate((scores_1, outputs[:, 1].cpu().detach().numpy()))

        slide_batch_num += 1

        if last_batch:
            in_between = True

            all_scores.append(scores_1.mean())
            all_labels.append(0 if all_scores[-1] < 0.5 else 1)
            all_targets.append(target.cpu().numpy()[0][0])

            # Sanity check:
            if target_current != target.cpu().numpy()[0][0]:
                print('Target Error!!')

            if target == 1:
                total_pos += 1
                if all_labels[-1] == 1:
                    true_pos += 1
            elif target == 0:
                total_neg += 1
                if all_labels[-1] == 0:
                    true_neg += 1

fpr_train, tpr_train, _ = roc_curve(all_targets, all_scores)

# Save roc_curve to file:
if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

file_name = os.path.join(data_path, output_dir, 'Inference', 'Model_Epoch_' + str(args.from_epoch)
                         + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + '.data')
inference_data = [fpr_train, tpr_train, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, len(inf_dset)]
with open(file_name, 'wb') as filehandle:
    pickle.dump(inference_data, filehandle)

print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
print('Done !')
