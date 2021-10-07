import utils
import datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import nets_mil
from tqdm import tqdm
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys
import matplotlib.pyplot as plt
from cycler import cycler

parser = argparse.ArgumentParser(description='WSI_MIL Features Slide inference')
parser.add_argument('-ex', '--experiment', type=int, default=10436, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=[500], help='Use this epoch model for inference')
parser.add_argument('-sts', '--save_tile_scores', dest='save_tile_scores', action='store_true', help='save tile scores')
#parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles to use')
#parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
#parser.add_argument('-f', '--folds', type=list, default=[2], help=' folds to infer')
#parser.add_argument('--model', default='resnet50_gn', type=str, help='resnet50_gn / receptornet') # RanS 15.12.20
args = parser.parse_args()

EPS = 1e-7

custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))

# Device definition:
DEVICE = utils.device_gpu_cpu()

# Get number of available CPUs:
cpu_available = utils.get_cpu()

# Data type definition:
DATA_TYPE = 'Features'

# Loss criterion definition:
criterion = nn.CrossEntropyLoss()

# Load saved model:
print('Loading pre-saved model from Exp. {} and epoch {}'.format(args.experiment, args.from_epoch))
run_data_output = utils.run_data(experiment=args.experiment)
output_dir, test_fold, dataset, target, model_name, free_bias, CAT_only =\
    run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Dataset Name'], run_data_output['Receptor'],\
    run_data_output['Model Name'], run_data_output['Free Bias'], run_data_output['CAT Only']
if sys.platform == 'darwin':
    # fix output_dir:
    if output_dir.split('/')[1] == 'home':
        output_dir = '/'.join(output_dir.split('/')[-2:])

    # if target in ['ER', 'ER_Features']:
    #if test_fold == 1:
    #elif test_fold == 2:
    if dataset == 'FEATURES: Exp_293-ER-TestFold_1':
        dset = 'TCGA_ABCTB'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Fold_1/Test'

    elif dataset == 'FEATURES: Exp_299-ER-TestFold_2':
        dset = 'TCGA_ABCTB'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Test'

    # elif target in ['PR', 'PR_Features']:
    #if test_fold == 1:
    elif dataset == 'FEATURES: Exp_309-PR-TestFold_1':
        dset = 'TCGA_ABCTB'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Fold_1/Test'

    # elif target in ['Her2', 'Her2_Features']:
    # if test_fold == 1:
    elif dataset == 'FEATURES: Exp_308-Her2-TestFold_1':
        dset = 'TCGA_ABCTB'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Fold_1/Test'

    #if test_fold == 1:
    elif dataset == 'FEATURES: Exp_355-ER-TestFold_1':
        dset = 'CAT'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test'

    elif dataset == 'FEATURES: Exp_358-ER-TestFold_1':
        dset = 'CARMEL'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Test'

    args.save_tile_scores = False
    is_per_patient = True
    #is_per_patient = False if args.save_tile_scores else True
    carmel_only = True

# Get data:
if dataset == 'Combined Features':
    inf_dset = datasets.Combined_Features_for_MIL_Training_dataset(is_all_tiles=True,
                                                                   target=target,
                                                                   is_train=False,
                                                                   test_fold=test_fold,
                                                                   is_per_patient=is_per_patient)

else:
    inf_dset = datasets.Features_MILdataset(dataset=dset,
                                            data_location=test_data_dir,
                                            target=target,
                                            is_per_patient=is_per_patient,
                                            is_all_tiles=True,
                                            is_train=False,
                                            carmel_only=carmel_only,
                                            test_fold=test_fold)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)
legend_labels = []

if args.save_tile_scores and len(args.from_epoch) > 1:
    raise Exception('When saving tile scores, there should be only one model')

# When saving the data we save data for all features (including carmel) so there is no need to save it again when
# working only on carmel slides
if args.save_tile_scores and not carmel_only:
    all_slides_weights_before_sftmx_list = []
    all_slides_weights_after_sftmx_list = []
    all_slides_tile_scores_list = []
    all_slides_scores_list = []

    all_slides_weights_before_sftmx_list.append({})
    all_slides_weights_after_sftmx_list.append({})
    all_slides_tile_scores_list.append({})
    all_slides_scores_list.append({})

    if dataset == 'Combined Features':
        all_slides_tile_scores_list_ran = {'CAT': [{}], 'CARMEL': [{}]}
        all_slides_scores_list_ran = {'CAT': [{}], 'CARMEL': [{}]}
    else:
        all_slides_tile_scores_list_ran = []
        all_slides_scores_list_ran = []
        all_slides_tile_scores_list_ran.append({})
        all_slides_scores_list_ran.append({})

# Load model
model = eval(model_name)
if free_bias:
    model.create_free_bias()
if CAT_only:
    model.CAT_only = True

total_loss, total_tiles_infered = 0, 0
for model_num, model_epoch in enumerate(args.from_epoch):
    model_data_loaded = torch.load(os.path.join(output_dir,
                                                'Model_CheckPoints',
                                                'model_data_Epoch_' + str(model_epoch) + '.pt'), map_location='cpu')

    model.load_state_dict(model_data_loaded['model_state_dict'])

    if model_data_loaded['bias_CAT'] != None:
        model.bias_CAT = model_data_loaded['bias_CAT']
        model.bias_CARMEL = model_data_loaded['bias_CARMEL']

    scores_reg = [] if dataset != 'Combined Features' else {inf_dset.dataset_list[0]: [], inf_dset.dataset_list[1]: []}
    all_scores_mil, all_labels_mil, all_targets = [], [], []
    total, correct_pos, correct_neg = 0, 0, 0
    total_pos, total_neg = 0, 0
    true_targets, scores_mil = np.zeros(0), np.zeros(0)
    correct_labeling = 0

    model.to(DEVICE)
    model.infer = True
    model.eval()

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(tqdm(inf_loader)):
            if dataset == 'Combined Features':
                total_tiles_infered += minibatch[list(minibatch.keys())[0]]['tile scores'].size(1)  #  count the number of tiles in each minibatch
                target = minibatch['CAT']['targets']
                data_CAT = minibatch['CAT']['features']
                data_CARMEL = minibatch['CARMEL']['features']

                data_CAT, data_CARMEL, target = data_CAT.to(DEVICE), data_CARMEL.to(DEVICE), target.to(DEVICE)
                data = {'CAT': data_CAT,
                        'CARMEL': data_CARMEL}
            else:
                target = minibatch['targets']
                data = minibatch['features']
                data, target = data.to(DEVICE), target.to(DEVICE)

            if model_num == 0:
                if dataset == 'Combined Features':
                    scores_reg['CAT'].append(minibatch['CAT']['tile scores'].mean().cpu().item())
                    scores_reg['CARMEL'].append(minibatch['CARMEL']['tile scores'].mean().cpu().item())
                else:
                    scores_reg.append(minibatch['scores'].mean().cpu().item())

            outputs, weights_after_sftmx, weights_before_softmax = model(x=None, H=data)

            minibatch_loss = criterion(outputs, target)
            total_loss += minibatch_loss

            if dataset == 'Combined Features':
                if type(weights_after_sftmx) == list:  # This will work on the model Combined_MIL_Feature_Attention_MultiBag_DEBUG
                    if len(weights_after_sftmx) == 2:
                        weights_after_sftmx = {'CAT': weights_after_sftmx[0].cpu().detach().numpy(),
                                               'CARMEL': weights_after_sftmx[1].cpu().detach().numpy()
                                               }
                    elif len(weights_after_sftmx) == 1:
                        weights_after_sftmx = {'CAT': weights_after_sftmx[0].cpu().detach().numpy(),
                                               'CARMEL': None
                                               }

                else:
                    for key in list(weights_after_sftmx.keys()):
                        weights_after_sftmx[key] = weights_after_sftmx[key].cpu().detach().numpy()
                        weights_before_softmax[key] = weights_before_softmax[key].cpu().detach().numpy()
            else:
                weights_after_sftmx = weights_after_sftmx.cpu().detach().numpy()
                weights_before_softmax = weights_before_softmax.cpu().detach().numpy()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            if args.save_tile_scores and not carmel_only:
                if dataset == 'Combined Features':
                    for key in list(minibatch.keys()):
                        slide_name = minibatch[key]['slide name']
                        tile_scores_ran = minibatch[key]['tile scores'].cpu().detach().numpy()[0]

                        all_slides_scores_list_ran[key][model_num][slide_name[0]] = minibatch[key]['slide scores'].cpu().detach().numpy()

                        if len(tile_scores_ran) != 500:
                            new_tile_scores_ran = np.zeros(500, )
                            new_tile_scores_ran[:len(tile_scores_ran), ] = tile_scores_ran
                            tile_scores_ran = new_tile_scores_ran

                        all_slides_tile_scores_list_ran[key][model_num][slide_name[0]] = tile_scores_ran

                        features_to_save = torch.transpose(data[key].squeeze(0), 1, 0)
                        slide_tile_scores_list = utils.extract_tile_scores_for_slide(features_to_save, [model])

                        if len(slide_tile_scores_list[0]) != 500:
                            new_slide_tile_scores_list = np.zeros(500, )
                            new_slide_tile_scores_list[:len(slide_tile_scores_list[0]), ] = slide_tile_scores_list[0]
                            slide_tile_scores_list[0] = new_slide_tile_scores_list

                        if weights_after_sftmx.shape[1] != 500:
                            new_weights = np.zeros((1, 500))
                            new_weights[:, :weights_after_sftmx.shape[1]] = weights_after_sftmx
                            weights_after_sftmx = new_weights

                        if weights_before_sftmx.shape[1] != 500:
                            new_weights = np.zeros((1, 500))
                            new_weights[:, :weights_before_sftmx.shape[1]] = weights_before_sftmx
                            weights_before_sftmx = new_weights

                        all_slides_tile_scores_list[model_num][slide_name[0]] = slide_tile_scores_list[0]
                        all_slides_weights_before_sftmx_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(
                            weights_before_sftmx.shape[1], )
                        all_slides_weights_after_sftmx_list[model_num][slide_name[0]] = weights_after_sftmx.reshape(
                            weights_after_sftmx.shape[1], )
                        # all_slides_weights_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                        all_slides_scores_list[model_num][slide_name[0]] = outputs[:, 1].cpu().detach().numpy()

                else:
                    slide_name = minibatch['slide name']
                    tile_scores_ran = minibatch['tile scores'].cpu().detach().numpy()[0]

                    all_slides_scores_list_ran[model_num][slide_name[0]] = minibatch['scores'].cpu().detach().numpy()

                    if len(tile_scores_ran) != 500:
                        new_tile_scores_ran = np.zeros(500, )
                        new_tile_scores_ran[:len(tile_scores_ran), ] = tile_scores_ran
                        tile_scores_ran = new_tile_scores_ran

                    all_slides_tile_scores_list_ran[model_num][slide_name[0]] = tile_scores_ran

                    features_to_save = torch.transpose(data.squeeze(0), 1, 0)
                    slide_tile_scores_list = utils.extract_tile_scores_for_slide(features_to_save, [model])

                    if len(slide_tile_scores_list[0]) != 500:
                        new_slide_tile_scores_list = np.zeros(500, )
                        new_slide_tile_scores_list[:len(slide_tile_scores_list[0]), ] = slide_tile_scores_list[0]
                        slide_tile_scores_list[0] = new_slide_tile_scores_list

                    if weights_after_sftmx.shape[1] != 500:
                        new_weights = np.zeros((1, 500))
                        new_weights[:, :weights_after_sftmx.shape[1]] = weights_after_sftmx
                        weights_after_sftmx = new_weights

                    if weights_before_sftmx.shape[1] != 500:
                        new_weights = np.zeros((1, 500))
                        new_weights[:, :weights_before_sftmx.shape[1]] = weights_before_sftmx
                        weights_before_sftmx = new_weights

                    all_slides_tile_scores_list[model_num][slide_name[0]] = slide_tile_scores_list[0]
                    all_slides_weights_before_sftmx_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                    all_slides_weights_after_sftmx_list[model_num][slide_name[0]] = weights_after_sftmx.reshape(weights_after_sftmx.shape[1], )
                    #all_slides_weights_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                    all_slides_scores_list[model_num][slide_name[0]] = outputs[:, 1].cpu().detach().numpy()

            scores_mil = np.concatenate((scores_mil, outputs[:, 1].cpu().detach().numpy()))
            true_targets = np.concatenate((true_targets, target.cpu().detach().numpy()))

            total += target.size(0)
            total_pos += target.eq(1).sum().item()
            total_neg += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()

            correct_pos += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg += predicted[target.eq(0)].eq(0).sum().item()

            all_targets.append(target.cpu().detach().numpy().item())
            all_labels_mil.append(predicted.cpu().detach().numpy().item())
            all_scores_mil.append(outputs[:, 1].cpu().detach().numpy().item())

    if args.save_tile_scores and not carmel_only:
        all_slides_score_dict = {'MIL': all_slides_scores_list,
                                 'REG': all_slides_scores_list_ran}
        all_tile_scores_dict = {'MIL': all_slides_tile_scores_list,
                                'REG': all_slides_tile_scores_list_ran}

        utils.save_all_slides_and_models_data(all_tile_scores_dict, all_slides_score_dict,
                                              all_slides_weights_before_sftmx_list, all_slides_weights_after_sftmx_list,
                                              [model], output_dir, args.from_epoch, '')
    if model_num == 0:
        if dataset == 'Combined Features':
            fpr_reg, tpr_reg, roc_auc_reg = {}, {}, {}
            fpr_reg['CAT'], tpr_reg['CAT'], _ = roc_curve(true_targets, np.array(scores_reg['CAT']))
            fpr_reg['CARMEL'], tpr_reg['CARMEL'], _ = roc_curve(true_targets, np.array(scores_reg['CARMEL']))
            roc_auc_reg['CAT'] = auc(fpr_reg['CAT'], tpr_reg['CAT'])
            roc_auc_reg['CARMEL'] = auc(fpr_reg['CARMEL'], tpr_reg['CARMEL'])
            plt.plot(fpr_reg['CAT'], tpr_reg['CAT'])
            plt.plot(fpr_reg['CARMEL'], tpr_reg['CARMEL'])
        else:
            fpr_reg, tpr_reg, _ = roc_curve(true_targets, np.array(scores_reg))
            roc_auc_reg = auc(fpr_reg, tpr_reg)
            plt.plot(fpr_reg, tpr_reg)

        postfix = 'Patient' if is_per_patient else 'Slide'
        if dataset == 'Combined Features':
            for key in list(minibatch.keys()):
                label_reg = 'REG [' + key + '] Per ' + postfix + ' AUC='
                legend_labels.append(label_reg + str(round(roc_auc_reg[key], 3)) + ')')
        else:
            label_reg = 'REG Per ' + postfix + ' AUC='
            legend_labels.append(label_reg + str(round(roc_auc_reg, 3)) + ')')

        label_MIL = 'Model' + str(model_epoch) + ': MIL Per ' + postfix + ' AUC='



    #acc = 100 * correct_labeling / total
    #balanced_acc = 100 * (correct_pos / (total_pos + EPS) + correct_neg / (total_neg + EPS)) / 2

    fpr_mil, tpr_mil, _ = roc_curve(true_targets, scores_mil)
    roc_auc_mil = auc(fpr_mil, tpr_mil)
    plt.plot(fpr_mil, tpr_mil)
    legend_labels.append(label_MIL + str(round(roc_auc_mil, 3)) + ')')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)
title = 'Inference Per {} for Model {}, Loss: {}, Per {} Tiles'.format('Patient' if is_per_patient else 'Slide',
                                                                       args.experiment,
                                                                       total_loss,
                                                                       total_tiles_infered)
plt.title(title)

if is_per_patient:
    graph_name = 'feature_mil_inference_per_patient_CARMEL_ONLY.png' if carmel_only else 'feature_mil_inference_per_patient.png'
else:
    graph_name = 'feature_mil_inference_per_slide_CARMEL_ONLY.png' if carmel_only else 'feature_mil_inference_per_slide.png'

plt.savefig(os.path.join(output_dir, graph_name))
print('Done')
