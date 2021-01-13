import utils
import datasets
from torch.utils.data import DataLoader
import torch
from nets_mil import ResNet34_GN_GatedAttention
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib import pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import pickle
from sklearn.utils import resample

parser = argparse.ArgumentParser(description='WSI_MIL Slide inference')
parser.add_argument('-ex', '--experiment', type=int, default=71, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=910, help='Use this epoch model for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[2], help=' folds to infer')
parser.add_argument('-ev', dest='eval', action='store_true', help='Use eval mode (or train mode')
#parser.add_argument('--model_path', type=str, default='/home/womer/project', help='path of saved model') #RanS 28.12.20
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error') #RanS 28.12.20
#parser.add_argument('--target', default='Her2', type=str, help='label: Her2/ER/PR/EGFR/PDL1/RedSquares') # RanS 7.12.20
parser.add_argument('--model', default='resnet50_gn', type=str, help='resnet50_gn / receptornet') # RanS 15.12.20
args = parser.parse_args()
eps = 1e-7

args.folds = list(map(int, args.folds))

DEVICE = utils.device_gpu_cpu()
data_path = ''
#data_path = r'C:\ran_data\HEROHE_examples' #temp RanS 5.11.20
#data_path = args.model_path
# Load saved model:
#model = ResNet34_GN_GatedAttention()
# Load model
model = utils.get_model(args.model)

print('Loading pre-saved model from Exp. {} and epoch {}'.format(args.experiment, args.from_epoch))
output_dir, _, _, TILE_SIZE, _, _, _, _, target, _ = utils.run_data(experiment=args.experiment)

# Tile size definition:
TILE_SIZE = 128
if sys.platform in ['linux', 'win32']:
    TILE_SIZE = 256

#model_data_loaded = torch.load(os.path.join(data_path, 'Model_CheckPoints',
model_data_loaded = torch.load(os.path.join(data_path, output_dir, 'Model_CheckPoints',
                                            'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

model.load_state_dict(model_data_loaded['model_state_dict'])


#inf_dset = datasets.Infer_WSI_MILdataset(DataSet=args.dataset,
inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      folds=args.folds,
                                      num_tiles=args.num_tiles,
                                      target_kind=target)
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model.infer = True
model.infer_part = 1
#model.part_1 = True
in_between = True
all_scores = []
all_labels = []
all_targets = []

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
true_pos, true_neg = 0, 0
num_correct = 0

model.eval()
with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_patches) in enumerate(tqdm(inf_loader)):
        if in_between:
            try:
                all_features = np.zeros([num_patches, model.M], dtype='float32')
            except:
                all_features = np.zeros([num_patches, 512], dtype='float32') #temp RanS 11.1.21, for REG models
            all_weights = np.zeros([1, num_patches], dtype='float32')
            slide_batch_num = 0
            in_between = False

        data, target = data.to(DEVICE), target.to(DEVICE)
        model.to(DEVICE)
        features, weights = model(data)
        all_features[slide_batch_num * inf_dset.tiles_per_iter : (slide_batch_num + 1) * inf_dset.tiles_per_iter, :] = features.cpu().numpy()
        all_weights[:, slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter] = weights.cpu().numpy()
        slide_batch_num += 1

        if last_batch:
            #model.part_1, model.part_2 = False, True
            model.infer_part = 2
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
            num_correct += label.eq(target).cpu().detach().int().item()

            #model.part_1, model.part_2 = True, False
            model.infer_part = 1

#fpr_train, tpr_train, _ = roc_curve(all_targets, all_scores)

acc_err, bacc_err, roc_auc_err = np.nan, np.nan, np.nan
if not args.bootstrap:
    acc = 100 * float(num_correct) / len(inf_loader)
    bacc = 100. * ((true_pos + eps) / (total_pos + eps) + (true_neg + eps) / (total_neg + eps)) / 2
    roc_auc = np.nan
    if not all(all_targets == all_targets[0]):  # more than one label
        fpr, tpr, _ = roc_curve(all_targets, all_scores)
        roc_auc = auc(fpr, tpr)
else:  # bootstrap, RanS 16.12.20
    n_iterations = 100
    # run bootstrap
    roc_auc_array, acc_array, bacc_array = np.empty(n_iterations), np.empty(n_iterations), np.empty(n_iterations)
    roc_auc_array[:], acc_array[:], bacc_array[:] = np.nan, np.nan, np.nan

    for ii in range(n_iterations):
        # resample bags, each bag is a sample
        scores_resampled, preds_resampled, targets_resampled = resample(all_scores, all_labels, all_targets)
        fpr, tpr, _ = roc_curve(targets_resampled, scores_resampled)

        num_correct_i = np.sum(preds_resampled == targets_resampled)
        true_pos_i = np.sum(targets_resampled + preds_resampled == 2)
        total_pos_i = np.sum(targets_resampled == 0)
        true_neg_i = np.sum(targets_resampled + preds_resampled == 0)
        total_neg_i = np.sum(targets_resampled == 1)
        acc_array[ii] = 100 * float(num_correct_i) / len(inf_loader)
        bacc_array[ii] = 100. * ((true_pos_i + eps) / (total_pos_i + eps) + (true_neg_i + eps) / (total_neg_i + eps)) / 2
        if not all(targets_resampled == targets_resampled[0]):  # more than one label
            roc_auc_array[ii] = roc_auc_score(targets_resampled, scores_resampled)

    roc_auc = np.nanmean(roc_auc_array)
    roc_auc_err = np.nanstd(roc_auc_array)
    acc = np.nanmean(acc_array)
    acc_err = np.nanstd(acc_array)
    bacc = np.nanmean(bacc_array)
    bacc_err = np.nanstd(bacc_array)

# Save roc_curve to file:
if not os.path.isdir(os.path.join(data_path, 'Inference')):
    os.mkdir(os.path.join(data_path, 'Inference'))

file_name = os.path.join(data_path, 'Inference', 'Model_Epoch_' + str(args.from_epoch)
                         + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + ('-EVAL_' if args.eval else '-TRAIN_') + 'MODE'
                         + '.data')
#inference_data = [fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, len(inf_dset)]
inference_data = [roc_auc, roc_auc_err, acc, acc_err, bacc, bacc_err,
                  all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, len(inf_dset)]
with open(file_name, 'wb') as filehandle:
    pickle.dump(inference_data, filehandle)

#plt.plot(fpr_train, tpr_train)
#plt.show()

print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
print('AUC = {} '.format(roc_auc))
print('Done !')
