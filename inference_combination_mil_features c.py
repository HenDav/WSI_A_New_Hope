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

'''
parser = argparse.ArgumentParser(description='WSI_MIL Features Slide inference')
parser.add_argument('-ex', '--experiment', type=int, default=10390, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=[495], help='Use this epoch model for inference')
parser.add_argument('-sts', '--save_tile_scores', dest='save_tile_scores', action='store_true', help='save tile scores')
args = parser.parse_args()
'''
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

# Definition of data and MIL model location:
Dataset_location = {'TCGA_ABCTB': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test/',
                    'CAT': r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test'}
MIL_models = {'TCGA_ABCTB': {'experiment': 351,
                             'epoch': 500},
              'CAT': {'experiment': 10391,
                      'epoch': 495}
              }

# Load saved model:
for data_name in MIL_models.keys():
    experiment, epoch = MIL_models[data_name]['experiment'], MIL_models[data_name]['epoch']

    print('Loading pre-saved model from Exp. {} and epoch {}'.format(experiment, epoch))
    output_dir, test_fold, _, _, _, _, _, dataset, target, _, model_name, _ = utils.run_data(experiment=experiment)
    # fix output_dir:
    if output_dir.split('/')[1] == 'home':
        output_dir = '/'.join(output_dir.split('/')[-2:])

    if dataset == 'FEATURES: Exp_293-ER-TestFold_1':
        dataset = 'TCGA_ABCTB'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test'

    elif dataset == 'FEATURES: Exp_355-ER-TestFold_1':
        dataset = 'CAT'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test'

    MIL_models[data_name]['output_dir'] = output_dir
    MIL_models[data_name]['test_fold'] = test_fold
    MIL_models[data_name]['dataset'] = dataset
    MIL_models[data_name]['target'] = target
    MIL_models[data_name]['model_name'] = model_name
    MIL_models[data_name]['test_data_dir'] = test_data_dir

save_tile_scores = True

# Get data:
# Since we're working with ready feature vectors we'll initiate 2 dataset, each for 1 Regular model that creates

TCGA_ABCTB_inf_dset = datasets.Features_MILdataset(dataset='TCGA_ABCTB',
                                                   data_location=MIL_models['TCGA_ABCTB']['test_data_dir'],
                                                   target=target,
                                                   is_per_patient=False,
                                                   is_all_tiles=True,
                                                   is_train=False,
                                                   test_fold=test_fold)

CAT_inf_dset = datasets.Features_MILdataset(dataset='CAT',
                                            data_location=MIL_models['CAT']['test_data_dir'],
                                            target=target,
                                            is_per_patient=False,
                                            is_all_tiles=True,
                                            is_train=False,
                                            test_fold=test_fold)

TCGA_ABCTB_inf_loader = DataLoader(TCGA_ABCTB_inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
CAT_inf_loader = DataLoader(CAT_inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

MIL_models['TCGA_ABCTB']['inf loader'] = TCGA_ABCTB_inf_loader
MIL_models['CAT']['inf loader'] = CAT_inf_loader

fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)
legend_labels = []

if save_tile_scores:
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

# Load models
slide_names_for_inference = []
slide_scores_and_weights = {}
for model_num, key in enumerate(MIL_models.keys()):
    model_dict = MIL_models[key]
    model = eval(model_dict['model_name'])
    model_data_loaded = torch.load(os.path.join(model_dict['output_dir'],
                                                'Model_CheckPoints',
                                                'model_data_Epoch_' + str(model_dict['epoch']) + '.pt'), map_location='cpu')

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

    MIL_models[key]['bias difference'] = model.classifier.bias.detach().cpu().numpy()[1] - model.classifier.bias.detach().cpu().numpy()[0]
    MIL_models[key]['slide names'] = []
    MIL_models[key]['reg scores list'] = []
    MIL_models[key]['true_targets'], MIL_models[key]['scores_mil'] = np.zeros(0), np.zeros(0)

    with torch.no_grad():
        for batch_idx, minibatch in enumerate(tqdm(MIL_models[key]['inf loader'])):
            slide_name = minibatch['slide name'][0]
            if slide_name.split('-')[0] != 'TCGA':
                continue

            labels = minibatch['labels']
            target = minibatch['targets']
            data = minibatch['features']

            MIL_models[key][slide_name] = {}
            MIL_models[key][slide_name]['true target'] = target.item()

            '''if model_num == 0:
                scores_reg.append(minibatch['scores'].mean().cpu().item())'''

            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs, weights_after_softmax, weights_before_softmax = model(x=None, H=data)

            weights_before_sftmx = weights_before_softmax.cpu().detach().numpy()
            weights_after_sftmx = weights_after_softmax.cpu().detach().numpy()

            #slide_scores_and_weights[minibatch['slide name'][0]] = [minibatch['features'].squeeze(0), weights_before_sftmx]

            outputs = torch.nn.functional.softmax(outputs, dim=1)

            MIL_models[key]['slide names'].append(minibatch['slide name'][0])

            if save_tile_scores:
                tile_scores_ran = minibatch['tile scores'].cpu().detach().numpy()[0]

                #all_slides_scores_list_ran[model_num][slide_name] = minibatch['scores'].cpu().detach().numpy()

                if len(tile_scores_ran) != 500:
                    new_tile_scores_ran = np.zeros(500, )
                    new_tile_scores_ran[:len(tile_scores_ran), ] = tile_scores_ran
                    tile_scores_ran = new_tile_scores_ran

                MIL_models[key][slide_name]['reg tile scores'] = tile_scores_ran
                MIL_models[key][slide_name]['reg slide score'] = minibatch['scores'].cpu().item()
                MIL_models[key]['reg scores list'].append(minibatch['scores'].cpu().item())
                #all_slides_tile_scores_list_ran[model_num][slide_name[0]] = tile_scores_ran

                features_to_save = torch.transpose(data.squeeze(0), 1, 0)
                slide_tile_scores_list = utils.extract_tile_scores_for_slide(features_to_save, [model])

                if len(slide_tile_scores_list[0]) != 500:
                    new_slide_tile_scores_list = np.zeros(500, )
                    new_slide_tile_scores_list[:len(slide_tile_scores_list[0]), ] = slide_tile_scores_list[0]
                    slide_tile_scores_list[0] = new_slide_tile_scores_list

                MIL_models[key][slide_name]['MIL tile score without bias'] = slide_tile_scores_list[0]

                if weights_after_sftmx.shape[1] != 500:
                    new_weights = np.zeros((1, 500))
                    new_weights[:, :weights_after_sftmx.shape[1]] = weights_after_sftmx
                    weights_after_sftmx = new_weights

                MIL_models[key][slide_name]['MIL weights after softmax'] = weights_after_sftmx

                if weights_before_sftmx.shape[1] != 500:
                    new_weights = np.zeros((1, 500))
                    new_weights[:, :weights_before_sftmx.shape[1]] = weights_before_sftmx
                    weights_before_sftmx = new_weights

                MIL_models[key][slide_name]['MIL weights before softmax'] = weights_before_sftmx

                '''
                all_slides_tile_scores_list[model_num][slide_name[0]] = slide_tile_scores_list[0]
                all_slides_weights_before_sftmx_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                all_slides_weights_after_sftmx_list[model_num][slide_name[0]] = weights_after_sftmx.reshape(weights_after_sftmx.shape[1], )
                #all_slides_weights_list[model_num][slide_name[0]] = weights_before_sftmx.reshape(weights_before_sftmx.shape[1], )
                all_slides_scores_list[model_num][slide_name[0]] = outputs[:, 1].cpu().detach().numpy()
                '''
            #scores_mil = np.concatenate((scores_mil, outputs[:, 1].cpu().detach().numpy()))
            #true_targets = np.concatenate((true_targets, target.cpu().detach().numpy()))

            MIL_models[key]['scores_mil'] = np.concatenate((MIL_models[key]['scores_mil'], outputs[:, 1].cpu().detach().numpy()))
            MIL_models[key]['true_targets'] = np.concatenate((MIL_models[key]['true_targets'], target.cpu().detach().numpy()))


            '''total += target.size(0)
            total_pos += target.eq(1).sum().item()
            total_neg += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()

            correct_pos += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg += predicted[target.eq(0)].eq(0).sum().item()

            all_targets.append(target.cpu().detach().numpy().item())
            all_labels_mil.append(predicted.cpu().detach().numpy().item())
            all_scores_mil.append(outputs[:, 1].cpu().detach().numpy().item())'''

    if save_tile_scores:
        '''
        all_slides_score_dict = {'MIL': all_slides_scores_list,
                                 'REG': all_slides_scores_list_ran}
        all_tile_scores_dict = {'MIL': all_slides_tile_scores_list,
                                'REG': all_slides_tile_scores_list_ran}
        '''
        '''utils.save_all_slides_and_models_data(all_tile_scores_dict, all_slides_score_dict,
                                              all_slides_weights_before_sftmx_list, all_slides_weights_after_sftmx_list,
                                              [model], model_dict['output_dir'], model_dict['epoch'], '')'''

    fpr_reg, tpr_reg, _ = roc_curve(MIL_models[key]['true_targets'], np.array(MIL_models[key]['reg scores list']))
    roc_auc_reg = auc(fpr_reg, tpr_reg)
    plt.plot(fpr_reg, tpr_reg)
    label = 'REG Per Slide for model ' + key + ', AUC='
    legend_labels.append(label + str(round(roc_auc_reg, 3)) + ')')


    fpr_mil, tpr_mil, _ = roc_curve(MIL_models[key]['true_targets'], MIL_models[key]['scores_mil'])
    roc_auc_mil = auc(fpr_mil, tpr_mil)
    plt.plot(fpr_mil, tpr_mil)
    legend_labels.append('MIL Per Slide for model ' + key + ': Exp. ' + str(MIL_models[key]['experiment']) + ' Epoch ' + str(MIL_models[key]['epoch']) + ', AUC=' + str(round(roc_auc_mil, 3)) + ')')

# Now we'll combine the results of the two mil models:
slides_0 = set(MIL_models['TCGA_ABCTB']['slide names'])
slides_1 = set(MIL_models['CAT']['slide names'])
diff_0 = slides_0 - slides_1
diff_1 = slides_1 - slides_0
if len(diff_0) == 0:
    common_slides = slides_0
elif len(diff_1) == 0:
    common_slides = slides_1
else:
    raise Exception('Could not find a common list of slides')

bias_TCGA = MIL_models['TCGA_ABCTB']['bias difference']
bias_CAT = MIL_models['CAT']['bias difference']

MIL_models['combined'] = {}
MIL_models['combined']['mil scores'] = []
MIL_models['combined']['true targets'] = []

for slide_name in list(common_slides):
    tile_scores_TCGA = MIL_models['TCGA_ABCTB'][slide_name]['MIL tile score without bias']
    weights_TCGA = MIL_models['TCGA_ABCTB'][slide_name]['MIL weights before softmax']
    tile_scores_CAT = MIL_models['CAT'][slide_name]['MIL tile score without bias']
    weights_CAT = MIL_models['CAT'][slide_name]['MIL weights before softmax']

    if weights_TCGA.shape[1] != 500 or weights_CAT.shape[1] != 500:
        raise Exception('Weights dimension different from 500')

    all_weights = np.zeros((1, 1000))
    all_weights[:, :500] = weights_TCGA
    all_weights[:, 500:] = weights_CAT

    all_weights_after_softmax = torch.nn.functional.softmax(torch.tensor(all_weights)).numpy()

    weights_TCGA_after_softmax = all_weights_after_softmax[:, :500]
    weights_CAT_after_softmax = all_weights_after_softmax[:, 500:]

    slide_score_TCGA = np.matmul(weights_TCGA_after_softmax, tile_scores_TCGA) + bias_TCGA
    slide_score_CAT = np.matmul(weights_CAT_after_softmax, tile_scores_CAT) + bias_CAT

    combined_slide_score = 1 / (1 + np.exp(- slide_score_TCGA - slide_score_CAT))

    MIL_models['combined']['mil scores'].append(combined_slide_score.item())

    if MIL_models['TCGA_ABCTB'][slide_name]['true target'] != MIL_models['CAT'][slide_name]['true target']:
        raise Exception('Slide has multiple targets')

    MIL_models['combined']['true targets'].append(MIL_models['TCGA_ABCTB'][slide_name]['true target'])
    MIL_models['combined'][slide_name] = {'MIL weights before softmax': all_weights,
                                          'MIL weights after softmax': all_weights_after_softmax,
                                          'slide score': combined_slide_score}


fpr_mil, tpr_mil, _ = roc_curve(np.array(MIL_models['combined']['true targets']), np.array(MIL_models['combined']['mil scores']))
roc_auc_mil = auc(fpr_mil, tpr_mil)
plt.plot(fpr_mil, tpr_mil)
legend_labels.append('COMBINED MIL: Per Slide, AUC=' + str(round(roc_auc_mil, 3)) + ')')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)

plt.savefig(os.path.join(output_dir, 'feature_mil_inference.png'))
print('Done')
