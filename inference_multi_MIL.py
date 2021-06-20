import utils
import datasets
from torch.utils.data import DataLoader
import torch
import nets_mil
import nets_mil_1
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
parser.add_argument('-ex', '--experiment', type=int, default=[304], help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=[1000], help='Use this epoch model for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=10, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, default=[3], help=' folds to infer')
parser.add_argument('-sts', '--save_tile_scores', dest='save_tile_scores', action='store_true', help='save tile scores')
args = parser.parse_args()

args.folds = list(map(int, args.folds))

if sys.platform == 'darwin':
    args.experiment = [1]
    args.from_epoch = [1035]
    args.save_tile_scores = True


if len(args.experiment) > 1:
    if len(args.experiment) != len(args.from_epoch):
        raise Exception("number of from_epoch(-fe) should be equal to number of experiment(-ex)")
    else:
        different_experiments = True
        Output_Dirs = []
else:
    different_experiments = False

DEVICE = utils.device_gpu_cpu()
data_path = ''

# Load saved model:
print('Loading pre-saved models:')
models = []


for counter in range(len(args.from_epoch)):
    epoch = args.from_epoch[counter]
    experiment = args.experiment[counter] if different_experiments else args.experiment[0]

    print('  Exp. {} and Epoch {}'.format(experiment, epoch))
    # Basic meta data will be taken from the first model (ONLY if all inferences are done from the same experiment)
    if counter == 0:
        output_dir, _, _, TILE_SIZE, _, _, _, _, args.target, _, model_name, desired_magnification = utils.run_data(experiment=experiment)
        if different_experiments:
            Output_Dirs.append(output_dir)
        fix_data_path = True
    elif counter > 0 and different_experiments:
        output_dir, _, _, _, _, _, _, _, target, _, model_name, desired_magnification = utils.run_data(experiment=experiment)
        Output_Dirs.append(output_dir)
        fix_data_path = True

    if fix_data_path:
        # we need to make some root modifications according to the computer we're running at.
        if sys.platform == 'linux':
            data_path = ''
        elif sys.platform == 'win32':
            output_dir = output_dir.replace(r'/', '\\')
            data_path = os.getcwd()

        fix_data_path = False

        # Verifying that the target receptor is not changed:
        if counter > 1 and args.target != target:
            raise Exception("Target Receptor is changed between models - DataSet cannot support this action")

    # Tile size definition:
    if sys.platform == 'darwin':
        TILE_SIZE = 128

    # loading basic model type
    model = eval(model_name)
    # loading model parameters from the specific epoch
    model_data_loaded = torch.load(os.path.join(data_path, output_dir, 'Model_CheckPoints',
                                                'model_data_Epoch_' + str(epoch) + '.pt'), map_location='cpu')
    model.load_state_dict(model_data_loaded['model_state_dict'])
    model.eval()
    model.infer = True
    model.features_part = True
    models.append(model)

inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=20,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles,
                                  desired_slide_magnification=desired_magnification)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
NUM_SLIDES = len(inf_dset.valid_slide_indices)

all_targets = []
all_scores_for_class_1, all_labels = np.zeros((NUM_SLIDES, NUM_MODELS)), np.zeros((NUM_SLIDES, NUM_MODELS))
#all_weights_after_sftmx = np.zeros((NUM_SLIDES, args.num_tiles, NUM_MODELS))
tile_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))
tile_scores[:] = np.nan
all_patient_barcodes = []
slide_names = []
slide_num = 0

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos, correct_neg = [0] * NUM_MODELS, [0] * NUM_MODELS

if args.save_tile_scores:
    all_slides_tile_scores_list = []
    all_slides_weights_list = []
    all_slides_scores_list = []
    for _ in range(len(models)):
        all_slides_tile_scores_list.append({})
        all_slides_weights_list.append({})
        all_slides_scores_list.append({})

with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, num_tiles, slide_name, patient_barcode) in enumerate(tqdm(inf_loader)):
        if new_slide:
            all_patient_barcodes.append(patient_barcode[0])
            slide_names.append(slide_name)
            scores_0, scores_1 = [np.zeros(0)] * NUM_MODELS, [np.zeros(0)] * NUM_MODELS
            target_current = target
            slide_batch_num = 0
            new_slide = False

            all_features = np.zeros([num_tiles, model.M, NUM_MODELS], dtype='float32')
            all_weights_before_sftmx = np.zeros([1, num_tiles, NUM_MODELS], dtype='float32')

        data, target = data.to(DEVICE), target.to(DEVICE)

        for model_num, model in enumerate(models):
            model.to(DEVICE)
            features, weights_before_sftmx = model(data)

            all_features[slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter, :, model_num] = features.detach().cpu().numpy()
            all_weights_before_sftmx[:, slide_batch_num * inf_dset.tiles_per_iter: (slide_batch_num + 1) * inf_dset.tiles_per_iter, model_num] = weights_before_sftmx.detach().cpu().numpy()

        slide_batch_num += 1

        if last_batch:
            # Save tile features and last models layer to file:
            if args.save_tile_scores:
                slide_tile_scores_list = utils.extract_tile_scores_for_slide(all_features, models)
                for idx in range(len(models)):
                    if len(slide_tile_scores_list[idx]) != args.num_tiles:
                        new_slide_tile_scores_list = np.zeros(args.num_tiles, )
                        new_slide_tile_scores_list[:len(slide_tile_scores_list[idx]), ] = slide_tile_scores_list[idx]
                        slide_tile_scores_list[idx] = new_slide_tile_scores_list

                    all_slides_tile_scores_list[idx][slide_name[0].split('/')[-1]] = slide_tile_scores_list[idx]

            new_slide = True

            all_targets.append(target.item())

            if target.item() == 1:
                total_pos += 1
            elif target.item() == 0:
                total_neg += 1

            all_features, all_weights_before_sftmx = torch.from_numpy(all_features).to(DEVICE), torch.from_numpy(all_weights_before_sftmx).to(DEVICE)

            for model_num, model in enumerate(models):
                model.features_part = False
                model.to(DEVICE)

                scores, weights = model(data, all_features[:, :, model_num], all_weights_before_sftmx[:, :, model_num])
                model.features_part = True

                scores = torch.nn.functional.softmax(scores, dim=1)
                _, predicted = scores.max(1)

                scores_0[model_num] = np.concatenate((scores_0[model_num], scores[:, 0].cpu().detach().numpy()))
                scores_1[model_num] = np.concatenate((scores_1[model_num], scores[:, 1].cpu().detach().numpy()))

                if target.item() == 1 and predicted.item() == 1:
                    correct_pos[model_num] += 1
                elif target.item() == 0 and predicted.item() == 0:
                    correct_neg[model_num] += 1

                if args.save_tile_scores:
                    if weights.shape[1] != args.num_tiles:
                        new_weights = torch.zeros(1, args.num_tiles)
                        new_weights[:, :weights.shape[1]] = weights
                        weights = new_weights

                    all_slides_weights_list[model_num][slide_name[0].split('/')[-1]] = weights.detach().cpu().numpy().reshape(weights.detach().cpu().numpy().shape[1],)
                    all_slides_scores_list[model_num][slide_name[0].split('/')[-1]] = scores_1[model_num]

                all_scores_for_class_1[slide_num, model_num] = scores_1[model_num]
                all_labels[slide_num, model_num] = predicted.item()

            slide_num += 1

if args.save_tile_scores:
    if not different_experiments:
        Output_Dirs = output_dir
    utils.save_all_slides_and_models_data(all_slides_tile_scores_list, all_slides_scores_list, all_slides_weights_list, models, Output_Dirs, args.from_epoch, data_path)

# Computing performance data for all models (over all slides scores data):
for model_num in range(NUM_MODELS):
    if different_experiments:
        output_dir = Output_Dirs[model_num]

    # We'll now gather the data for computing performance per patient:
    all_targets_per_patient, all_scores_for_class_1_per_patient = utils.gather_per_patient_data(all_targets, all_scores_for_class_1[:, model_num], all_patient_barcodes)

    fpr_patient, tpr_patient, _ = roc_curve(all_targets_per_patient, all_scores_for_class_1_per_patient)
    roc_auc_patient = auc(fpr_patient, tpr_patient)

    fpr, tpr, _ = roc_curve(all_targets, all_scores_for_class_1[:, model_num])
    roc_auc = auc(fpr, tpr)

    # Save roc_curve to file:
    if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
        os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

    file_name = os.path.join(data_path, output_dir, 'Inference', 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '-Tiles_' + str(args.num_tiles) + '.data')

    inference_data = [fpr, tpr, fpr_patient, tpr_patient, all_labels[:, model_num], all_targets, all_scores_for_class_1[:, model_num], total_pos,
                      correct_pos[model_num], total_neg, correct_neg[model_num], len(inf_dset), np.squeeze(tile_scores[:, model_num, :]), slide_names]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)

    experiment = args.experiment[model_num] if different_experiments else args.experiment[0]
    print('For model from Experiment {} and Epoch {}: {} / {} correct classifications'
          .format(experiment,
                  args.from_epoch[model_num],
                  int(len(all_labels[:, model_num]) - np.abs(np.array(all_targets) - np.array(all_labels[:, model_num])).sum()),
                  len(all_labels[:, model_num])))
    print('AUC per Slide = {} '.format(roc_auc))
    print('AUC per Patient = {} '.format(roc_auc_patient))

print('Done !')
