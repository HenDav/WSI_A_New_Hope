from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np
import pandas as pd


custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))
fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)

inference_files = {}
#inference_files['exp38_epoch558_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_100.data'
#inference_files['exp38_epoch558_test_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_10.data'
#inference_files['exp36_epoch72_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp36\Inference\Model_Epoch_72-Folds_[1]-Tiles_100.data'
#inference_files['test_2'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_2.data'
'''inference_files['exp38_epoch482_test_3'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_3.data'
inference_files['exp38_epoch482_test_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_10.data'
inference_files['exp38_epoch482_test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_100.data'
inference_files['exp38_epoch482_test_200'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[1]-Tiles_200.data' '''
#inference_files['exp38_epoch482_train_10'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_482-Folds_[2, 3, 4, 5, 6]-Tiles_10.data'
#inference_files['test_100'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_558-Folds_[1]-Tiles_100.data'
#inference_files['test_500'] = 'Data from gipdeep/runs/79/Inference/Model_Epoch_12000-Folds_[2]-Tiles_500.data'
#inference_files['test_1000'] = 'Data from gipdeep/runs/79/Inference/Model_Epoch_12000-Folds_[2]-Tiles_1000.data'
'''inference_files['TCGA, mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp70\Inference\Model_Epoch_34-Folds_[1]-Tiles_500.data'
inference_files['TCGA, no mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp63,64\exp63\Inference\Model_Epoch_48-Folds_[1]-Tiles_500.data'
inference_files['TCGA dx only, mean pixel regularization, test set, 500 patches'] = r'C:\Pathnet_results\MIL_general_try4\exp77\Inference\Model_Epoch_72-Folds_[1]-Tiles_500.data' '''
#inference_files['ex38_epoch1080_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[1]-Tiles_500.data'
#inference_files['ex38_epoch1080_train_fold2_20'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[2]-Tiles_20.data'
#inference_files['ex38_epoch1060_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1060-Folds_[1]-Tiles_500.data'
#inference_files['ex38_epoch1040_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1040-Folds_[1]-Tiles_500.data'
#inference_files['ex38_epoch1020_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1020-Folds_[1]-Tiles_500.data'
#inference_files['ex38_epoch1000_test_500'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1000-Folds_[1]-Tiles_500.data'
#inference_files['ex38_epoch1080_test_500_single_infer'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[1]-Tiles_500_inference_REG.data'
#inference_files['rons_epoch1607_test_500'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\Model_Epoch_1607-Folds_[1]-Tiles_500.data'
#inference_files['rons_epoch1607_test_20_resnet_v2'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_070221.data'
#inference_files['rons_epoch1607_test_20_resnet_v2_no_softmax'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_no_softmax_070221.data'
#inference_files['rons_epoch1607_test_20_resnet_v2_no_softmax_fold2'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_resnet_v2_no_softmax_fold2_070221.data'
'''inference_files['rons_epoch1467_20_patches_fold1_test_mag10'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_mag10_resnet_v2_no_softmax_fold1_080221.data' 
inference_files['rons_epoch1467_20_patches_fold1_test_mag20'] = r'C:\Pathnet_results\Rons_TCGA\fold1_compare\fold_1_ER - Ron\Inference\out1_20tiles_mag20_resnet_v2_no_softmax_fold1_080221.data' '''
'''inference_files['TCGA_ER_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
inference_files['TCGA_ER_fold1_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1480-Folds_[1]-Tiles_500.data'
inference_files['TCGA_ER_fold1_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1460-Folds_[1]-Tiles_500.data'
inference_files['TCGA_ER_fold1_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1440-Folds_[1]-Tiles_500.data'
inference_files['TCGA_ER_fold1_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\TCGA_ER_mag_10\exp141\Inference\Model_Epoch_1420-Folds_[1]-Tiles_500.data' '''

'''inference_files['exp177_epoch740_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\our_slides_comparison_180321\Model_Epoch_740-Folds_[1]-Tiles_100.data'
inference_files['exp177_epoch760_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\our_slides_comparison_180321\Model_Epoch_760-Folds_[1]-Tiles_100.data'
inference_files['rons_epoch1467_test_100'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\inference\our_slides_comparison_180321\Model_Epoch_rons_model-Folds_[1]-Tiles_100.data' '''

'''inference_files['exp226_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 226\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
inference_files['exp239_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 239\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'
inference_files['exp240_fold2_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 240\Inference\Model_Epoch_1500-Folds_[2]-Tiles_500.data'
inference_files['exp241_fold3_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 241\Inference\Model_Epoch_1500-Folds_[3]-Tiles_500.data'
inference_files['exp242_fold4_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 242\Inference\Model_Epoch_1500-Folds_[4]-Tiles_500.data'
inference_files['exp243_fold5_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\Onco\exp 243\Inference\Model_Epoch_1500-Folds_[5]-Tiles_500.data' '''

"""inference_files['exp177_fold1_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_700-Folds_[1]-Tiles_500.data'
inference_files['exp177_fold1_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_720-Folds_[1]-Tiles_500.data'
inference_files['exp177_fold1_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_740-Folds_[1]-Tiles_500.data'
inference_files['exp177_fold1_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_760-Folds_[1]-Tiles_500.data'
inference_files['exp177_fold1_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_780-Folds_[1]-Tiles_500.data'
inference_files['exp177_fold1_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp177\Inference\Model_Epoch_800-Folds_[1]-Tiles_500.data'"""

"""inference_files['exp227_fold3_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_700-Folds_[3]-Tiles_500.data'
inference_files['exp227_fold3_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_720-Folds_[3]-Tiles_500.data'
inference_files['exp227_fold3_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_740-Folds_[3]-Tiles_500.data'
inference_files['exp227_fold3_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_760-Folds_[3]-Tiles_500.data'
inference_files['exp227_fold3_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_780-Folds_[3]-Tiles_500.data'
inference_files['exp227_fold3_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp227\Inference\Model_Epoch_800-Folds_[3]-Tiles_500.data'"""

"""inference_files['exp203_fold2_epoch700_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_700-Folds_[2]-Tiles_500.data'
inference_files['exp203_fold2_epoch720_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_720-Folds_[2]-Tiles_500.data'
inference_files['exp203_fold2_epoch740_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_740-Folds_[2]-Tiles_500.data'
inference_files['exp203_fold2_epoch760_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_760-Folds_[2]-Tiles_500.data'
inference_files['exp203_fold2_epoch780_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_780-Folds_[2]-Tiles_500.data'
inference_files['exp203_fold2_epoch800_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp203\Inference\Model_Epoch_800-Folds_[2]-Tiles_500.data'"""

"""inference_files['exp228_fold3_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1400-Folds_[3]-Tiles_500.data'
inference_files['exp228_fold3_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1420-Folds_[3]-Tiles_500.data'
inference_files['exp228_fold3_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1440-Folds_[3]-Tiles_500.data'
inference_files['exp228_fold3_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1460-Folds_[3]-Tiles_500.data'
inference_files['exp228_fold3_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1480-Folds_[3]-Tiles_500.data'
inference_files['exp228_fold3_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp228\Inference\Model_Epoch_1500-Folds_[3]-Tiles_500.data'"""

"""inference_files['exp229_fold1_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1400-Folds_[1]-Tiles_500.data'
inference_files['exp229_fold1_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1420-Folds_[1]-Tiles_500.data'
inference_files['exp229_fold1_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1440-Folds_[1]-Tiles_500.data'
inference_files['exp229_fold1_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1460-Folds_[1]-Tiles_500.data'
inference_files['exp229_fold1_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1480-Folds_[1]-Tiles_500.data'
inference_files['exp229_fold1_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp229\Inference\Model_Epoch_1500-Folds_[1]-Tiles_500.data'"""

inference_files['exp230_fold2_epoch1400_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1400-Folds_[2]-Tiles_500.data'
inference_files['exp230_fold2_epoch1420_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1420-Folds_[2]-Tiles_500.data'
inference_files['exp230_fold2_epoch1440_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1440-Folds_[2]-Tiles_500.data'
inference_files['exp230_fold2_epoch1460_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1460-Folds_[2]-Tiles_500.data'
inference_files['exp230_fold2_epoch1480_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1480-Folds_[2]-Tiles_500.data'
inference_files['exp230_fold2_epoch1500_test_500'] = r'C:\Pathnet_results\MIL_general_try4\no_softmax_runs\exp230\Inference\Model_Epoch_1500-Folds_[2]-Tiles_500.data'



infer_type = 'REG'


def auc_for_n_patches(patch_scores, n, all_targets):
    max_n = patch_scores.shape[1]
    n_iter = 10
    auc_array = np.zeros(n_iter)
    for iter in range(n_iter):
        patches = np.random.choice(np.arange(max_n), n, replace=False)
        chosen_patches = patch_scores[:, patches]
        chosen_mean_scores = np.array([np.nanmean(chosen_patches[ii, chosen_patches[ii, :] > 0]) for ii in range(chosen_patches.shape[0])])

        # TODO RanS 4.2.21 - handle slides with nans (less than max_n patches)
        #temp fix - remove slides if all selected patches are nan
        chosen_targets = np.array([all_targets[ii] for ii in range(len(all_targets)) if ~np.isnan(chosen_mean_scores[ii])])
        chosen_mean_scores = np.array([chosen_mean_score for chosen_mean_score in chosen_mean_scores if ~np.isnan(chosen_mean_score)])
        #chosen_targets = np.array([all_targets[patch] for patch in patches])
        auc_array[iter] = roc_auc_score(chosen_targets, chosen_mean_scores)

    auc_res = np.nanmean(auc_array)
    return auc_res


legend_labels = []
roc_auc = []
slide_score_all = []

for ind, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)

    if infer_type == 'REG':
        fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
        num_slides, patch_scores, all_slide_names = inference_data
        save_csv = False
        if save_csv:
            patch_scores_df = pd.DataFrame(patch_scores)
            patch_scores_df.insert(0, "slide_name", all_slide_names)
            patch_scores_df.to_csv(key + '_patch_scores.csv')
        roc_auc.append(auc(fpr, tpr))
        # RanS 18.1.21
        #temp fix RanS 4.2.21
        if patch_scores.ndim == 3:
            patch_scores = np.squeeze(patch_scores[:, ind,:])
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        slide_score_mean = np.array([np.nanmean(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        slide_score_std = np.array([np.nanstd(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])

        slide_score_all.append(slide_score_mean)

        #results per patient RanS 18.3.21
        is_TCGA = True
        if is_TCGA:
            patient_all = [all_slide_names[i][8:12] for i in range(all_slide_names.shape[0])] #only TCGA!
            patch_df = pd.DataFrame({'patient': patient_all, 'scores': slide_score_mean, 'targets': all_targets})
            patient_mean_score_df = patch_df.groupby('patient').mean()
            roc_auc_patient = roc_auc_score(patient_mean_score_df['targets'], patient_mean_score_df['scores'])
            fpr_patient, tpr_patient, thresholds_patient = roc_curve(patient_mean_score_df['targets'],
                                                                     patient_mean_score_df['scores'])

        test_n_patches = False
        if test_n_patches:
            n_list = np.arange(1, 100, 1)
            auc_n = np.zeros(n_list.shape)
            for ind, n in enumerate(n_list):
                auc_n[ind] = auc_for_n_patches(patch_scores, n, all_targets)
            plt.plot(n_list,auc_n)
            plt.xlabel('# of patches')
            plt.ylabel('test AUC score')
            plt.ylim([0,1])
            plt.xlim([1, 100])

    elif infer_type == 'MIL':
        roc_auc1, roc_auc_err, acc, acc_err, bacc, bacc_err, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides = inference_data
        roc_auc.append(roc_auc1)

    EPS = 1e-7
    print(key)
    print('{} / {} correct classifications'.format(int(len(all_labels) - np.abs(np.array(all_targets) - np.array(all_labels)).sum()), len(all_labels)))
    fpr, tpr, _ = roc_curve(all_targets, all_scores)
    roc_auc1 = roc_auc_score(all_targets, all_scores)
    balanced_acc = 100. * ((true_pos + EPS) / (total_pos + EPS) + (true_neg + EPS) / (total_neg + EPS)) / 2
    print('roc_auc:', roc_auc1)
    print('balanced_acc:', balanced_acc)
    print('np.sum(all_labels):', np.sum(all_labels))

    #plt.plot(fpr, tpr)
    #legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 2)) +')')
    if is_TCGA:
        plt.plot(fpr_patient, tpr_patient)
        legend_labels.append(key + ' (patient AUC=' + str(round(roc_auc_patient, 2)) + ')')

#combine several models, RanS 11.4.21
slide_score_mean_all = np.mean(np.array(slide_score_all), axis=0)
combine_all_models = False
if is_TCGA and combine_all_models:
    #patient_all = [all_slide_names[i][8:12] for i in range(all_slide_names.shape[0])]  # only TCGA!
    patch_all_df = pd.DataFrame({'patient': patient_all, 'scores': slide_score_mean_all, 'targets': all_targets})
    patient_mean_score_all_df = patch_all_df.groupby('patient').mean()
    roc_auc_all_patient = roc_auc_score(patient_mean_score_all_df['targets'], patient_mean_score_all_df['scores'])
    fpr_patient_all, tpr_patient_all, thresholds_patient_all = roc_curve(patient_mean_score_all_df['targets'],
                                                             patient_mean_score_all_df['scores'])
    plt.plot(fpr_patient_all, tpr_patient_all)
    legend_labels.append(' (all models combined patient AUC=' + str(round(roc_auc_all_patient, 2)) + ')')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)
plt.show()

