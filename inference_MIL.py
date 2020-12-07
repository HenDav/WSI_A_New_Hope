import utils
from torch.utils.data import DataLoader
import torch
from nets_mil import ResNet34_GN_GatedAttention
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(description='WSI_MIL Slide inference')
parser.add_argument('-ex', '--experiment', type=int, default=71, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=910, help='Use this epoch model for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[2], help=' folds to infer')
parser.add_argument('-ev', dest='eval', action='store_true', help='Use eval mode (or train mode')
args = parser.parse_args()

args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
#data_path = ''
data_path = r'C:\ran_data\HEROHE_examples' #temp RanS 5.11.20

# Load saved model:
model = ResNet34_GN_GatedAttention()

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
                                      folds=args.folds,
                                      print_timing=True,
                                      DX=False,
                                      num_tiles=args.num_tiles)
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model.infer = True
model.part_1 = True
in_between = True
all_scores = []
all_labels = []
all_targets = []

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
true_pos, true_neg = 0, 0

'''
if args.eval:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
else:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = True
'''

model.eval()
with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_patches) in enumerate(tqdm(inf_loader)):
        if in_between:
            all_features = np.zeros([num_patches, model.M], dtype='float32')
            all_weights = np.zeros([1, num_patches], dtype='float32')
            slide_batch_num = 0
            in_between = False

        data, target = data.to(DEVICE), target.to(DEVICE)
        model.to(DEVICE)
        features, weights = model(data)
        #print('iter: {}, file: {}'.format(batch_idx, inf_dset.current_file))
        #print(features.size(), all_features[slide_batch_num * inf_dset.tiles_per_iter : (slide_batch_num + 1) * inf_dset.tiles_per_iter, :].shape)
        all_features[slide_batch_num * inf_dset.tiles_per_iter : (slide_batch_num + 1) * inf_dset.tiles_per_iter, :] = features.cpu().numpy()
        all_weights[:, slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter] = weights.cpu().numpy()
        slide_batch_num += 1

        if last_batch:
            model.part_1, model.part_2 = False, True
            in_between = True

            all_features, all_weights = torch.from_numpy(all_features).to(DEVICE), torch.from_numpy(all_weights).to(DEVICE)
            score, label, _ = model(data, all_features, all_weights)
            all_scores.append(score.cpu().numpy()[0][0])
            all_labels.append(label.cpu().numpy()[0][0])
            all_targets.append(target.cpu().numpy()[0][0])

            if target == 1:
                total_pos += 1
                if label == 1:
                    true_pos += 1
            elif target == 0:
                total_neg += 1
                if label == 0:
                    true_neg += 1

            model.part_1, model.part_2 = True, False

fpr_train, tpr_train, _ = roc_curve(all_targets, all_scores)

# Save roc_curve to file:
if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

file_name = os.path.join(data_path, output_dir, 'Inference', 'Model_Epoch_' + str(args.from_epoch)
                         + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + ('-EVAL_' if args.eval else '-TRAIN_') + 'MODE'
                         + '.data')
inference_data = [fpr_train, tpr_train, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, len(inf_dset)]
with open(file_name, 'wb') as filehandle:
    pickle.dump(inference_data, filehandle)

#plt.plot(fpr_train, tpr_train)
#plt.show()

print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
print('Done !')
