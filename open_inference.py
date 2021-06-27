from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np
import pandas as pd
import os
from inference_loader_input import inference_files, inference_dir, save_csv, patient_level, inference_name

custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))
fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)

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
    with open(os.path.join(inference_dir, inference_files[key]), 'rb') as filehandle:
        inference_data = pickle.load(filehandle)

    if infer_type == 'REG':
        fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
        num_slides, patch_scores, all_slide_names = inference_data

        if save_csv:
            patch_scores_df = pd.DataFrame(patch_scores)
            patch_scores_df.insert(0, "slide_name", all_slide_names)
            patch_scores_df.to_csv(os.path.join(inference_dir, key + '_patch_scores.csv'))
        roc_auc.append(auc(fpr, tpr))
        # RanS 18.1.21
        #temp fix RanS 4.2.21
        if patch_scores.ndim == 3:
            patch_scores = np.squeeze(patch_scores[:, ind,:])
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        slide_score_mean = np.array([np.nanmean(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        slide_score_std = np.array([np.nanstd(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])

        slide_score_all.append(slide_score_mean)

        #results per patient RanS 18.3.21, works only for TCGA, ABCTB
        if patient_level:
            patient_all = []
            for name in all_slide_names:
                if os.path.splitext(name)[-1] == '.svs': #TCGA files
                    patient_all.append(name[8:12])
                elif os.path.splitext(name)[-1] == '.ndpi': #ABCTB files
                    patient_all.append(name[:9])
            #patient_all = [all_slide_names[i][8:12] for i in range(all_slide_names.shape[0])] #only TCGA!
            patch_df = pd.DataFrame({'patient': patient_all, 'scores': slide_score_mean, 'targets': all_targets})
            patient_mean_score_df = patch_df.groupby('patient').mean()
            roc_auc_patient = roc_auc_score(patient_mean_score_df['targets'].astype(int), patient_mean_score_df['scores'])
            fpr_patient, tpr_patient, thresholds_patient = roc_curve(patient_mean_score_df['targets'].astype(int),
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


    if patient_level:
        plt.plot(fpr_patient, tpr_patient)
        legend_labels.append(key + ' (patient AUC=' + str(round(roc_auc_patient, 2)) + ')')
    else:
        plt.plot(fpr, tpr)
        legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 2)) +')')

#combine several models, RanS 11.4.21
slide_score_mean_all = np.mean(np.array(slide_score_all), axis=0)
combine_all_models = False
if patient_level and combine_all_models:
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

if patient_level:
    print('average AUC per patient: ' + str(np.round(np.mean(roc_auc_patient), 2)))
else:
    print('average AUC per slide: ' + str(np.round(np.mean(roc_auc), 2)))
plt.savefig(os.path.join(inference_dir, inference_name + '_inference.png'))
print('finished')
#plt.show()

