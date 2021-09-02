import utils
import datasets
from torch.utils.data import DataLoader
import torch
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
parser.add_argument('-ex', '--experiment', type=int, default=10399, help='Continue train of this experiment')
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

# Load saved model:
print('Loading pre-saved model from Exp. {} and epoch {}'.format(args.experiment, args.from_epoch))
output_dir, test_fold, _, _, _, _, _, dataset, target, _, model_name, _ = utils.run_data(experiment=args.experiment)

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

    args.save_tile_scores = True
    is_per_patient = False if args.save_tile_scores else True

# Get data:
inf_dset = datasets.Features_MILdataset(dataset=dset,
                                        data_location=test_data_dir,
                                        target=target,
                                        is_per_patient=is_per_patient,
                                        is_all_tiles=True,
                                        is_train=False,
                                        test_fold=test_fold)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)
legend_labels = []

if args.save_tile_scores and len(args.from_epoch) > 1:
    raise Exception('When saving tile scores, there should be only one model')

if args.save_tile_scores:
    all_slides_weights_before_sftmx_list = []
    all_slides_weights_after_sftmx_list = []
    #all_slides_weights_list = []
    all_slides_tile_scores_list = []
    all_slides_scores_list = []

    all_slides_weights_before_sftmx_list.append({})
    all_slides_weights_after_sftmx_list.append({})
    #all_slides_weights_list.append({})
    all_slides_tile_scores_list.append({})
    all_slides_scores_list.append({})

    all_slides_tile_scores_list_ran = []
    all_slides_scores_list_ran = []
    all_slides_tile_scores_list_ran.append({})
    all_slides_scores_list_ran.append({})

# Load model
model = eval(model_name)
for model_num, model_epoch in enumerate(args.from_epoch):
    model_data_loaded = torch.load(os.path.join(output_dir,
                                                'Model_CheckPoints',
                                                'model_data_Epoch_' + str(model_epoch) + '.pt'), map_location='cpu')

    model.load_state_dict(model_data_loaded['model_state_dict'])

    scores_reg = []
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
            labels = minibatch['labels']
            target = minibatch['targets']
            data = minibatch['features']

            if model_num == 0:
                scores_reg.append(minibatch['scores'].mean().cpu().item())

            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs, weights_after_sftmx, weights_before_softmax = model(x=None, H=data)
            #print(data.shape, target.shape, outputs.shape, weights.shape)

            weights_after_sftmx = weights_after_sftmx.cpu().detach().numpy()
            weights_before_sftmx = weights_before_softmax.cpu().detach().numpy()
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            if args.save_tile_scores:
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

    if args.save_tile_scores:
        all_slides_score_dict = {'MIL': all_slides_scores_list,
                                 'REG': all_slides_scores_list_ran}
        all_tile_scores_dict = {'MIL': all_slides_tile_scores_list,
                                'REG': all_slides_tile_scores_list_ran}

        utils.save_all_slides_and_models_data(all_tile_scores_dict, all_slides_score_dict,
                                              all_slides_weights_before_sftmx_list, all_slides_weights_after_sftmx_list,
                                              [model], output_dir, args.from_epoch, '')
    if model_num == 0:
        fpr_reg, tpr_reg, _ = roc_curve(true_targets, np.array(scores_reg))
        roc_auc_reg = auc(fpr_reg, tpr_reg)
        plt.plot(fpr_reg, tpr_reg)
        if is_per_patient:
            label_reg = 'REG Per Patient AUC='
            label_MIL = 'Model' + str(model_epoch) + ': MIL Per Patient AUC='
        else:
            label_reg = 'REG Per Slide AUC='
            label_MIL = 'Model' + str(model_epoch) + ': MIL Per Slide AUC='
        legend_labels.append(label_reg + str(round(roc_auc_reg, 3)) + ')')

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

if is_per_patient:
    graph_name = 'feature_mil_inference_per_patient.png'
else:
    graph_name = 'feature_mil_inference_per_slide.png'

plt.savefig(os.path.join(output_dir, graph_name))
print('Done')
