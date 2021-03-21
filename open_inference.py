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

inference_files['1420_10'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1420-Folds_[1]-Tiles_10.data'
inference_files['1425_10'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1425-Folds_[1]-Tiles_10.data'

inference_files['1390_10'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1390-Folds_[1]-Tiles_10.data'
inference_files['1395_10'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1395-Folds_[1]-Tiles_10.data'

inference_files['1390_50'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1390-Folds_[1]-Tiles_50.data'
inference_files['1395_50'] = 'runs/Exp_241-ER-TestFold_1/Inference/Model_Epoch_1395-Folds_[1]-Tiles_50.data'



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
for ind, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)

    if infer_type == 'REG':
        fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides, patch_scores = inference_data
        save_csv = True
        if save_csv:
            patch_scores_df = pd.DataFrame(patch_scores)
            patch_scores_df.to_csv(key + '_patch_scores.csv')
        roc_auc.append(auc(fpr, tpr))
        # RanS 18.1.21
        #temp fix RanS 4.2.21
        if patch_scores.ndim == 3:
            patch_scores = np.squeeze(patch_scores[:, ind,:])
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        #slide_score_std = np.nanstd(patch_scores, axis=1)
        #slide_score_mean = np.nanmean(patch_scores, axis=1)
        slide_score_mean = np.array([np.nanmean(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        slide_score_std = np.array([np.nanstd(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        slide_conf = np.abs(slide_score_mean - 0.5)
        correct_array = all_labels == all_targets
        std_wrong = slide_score_std[~correct_array]
        std_correct = slide_score_std[correct_array]
        conf_wrong = slide_conf[~correct_array]
        conf_correct = slide_conf[correct_array]
        # plt.hist([conf_wrong, conf_correct])
        # plt.legend(['confidence - wrong predictions','confidence - correct predictions'])

        test_n_patches = False
        if test_n_patches:
            n_list = np.arange(1, 500, 1)
            auc_n = np.zeros(n_list.shape)
            for ind, n in enumerate(n_list):
                auc_n[ind] = auc_for_n_patches(patch_scores, n, all_targets)
            plt.plot(n_list,auc_n)
            plt.xlabel('# of patches')
            plt.ylabel('test AUC score')
            plt.ylim([0,1])
            plt.xlim([1, 500])

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

    plt.plot(fpr, tpr)
    legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 2)) +')')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)
plt.show()

