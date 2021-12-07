from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np
import pandas as pd
import os
from inference_loader_input import inference_files, inference_dir, save_csv, patient_level, inference_name, dataset

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

if len(inference_files) == 0:
    raise IOError('No inference files found!')

for ind, key in enumerate(inference_files.keys()):
    with open(os.path.join(inference_dir, inference_files[key]), 'rb') as filehandle:
        print(key)
        inference_data = pickle.load(filehandle)

    if key[-8:] == 'test_500':
        key = key[:-9]

    if infer_type == 'REG':
        if len(inference_data) == 14: #current format
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, all_slide_datasets, patch_locs = inference_data
        elif len(inference_data) == 13: #old format, before locations
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, all_slide_datasets = inference_data
        elif len(inference_data) == 12: #old format
            fpr, tpr, all_labels, all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names = inference_data
        elif len(inference_data) == 16: #temp old format with patch locs
            fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, \
            num_slides, patch_scores, all_slide_names, patch_locs, patch_locs_inds, all_slide_size, all_slide_size_ind = inference_data
        else:
            IOError('inference data is of unsupported size!')

        if save_csv:
            patch_scores_df = pd.DataFrame(patch_scores)
            patch_scores_df.insert(0, "slide_name", all_slide_names)
            patch_scores_df.insert(0, "slide_label", all_targets)
            patch_scores_df.to_csv(os.path.join(inference_dir, key + '_patch_scores.csv'))

            try:
                patch_x_df = pd.DataFrame(patch_locs[:, :, 0])
                patch_x_df.insert(0, "slide_name", all_slide_names)
                patch_x_df.to_csv(os.path.join(inference_dir, key + '_x.csv'))

                patch_y_df = pd.DataFrame(patch_locs[:, :, 1])
                patch_y_df.insert(0, "slide_name", all_slide_names)
                patch_y_df.to_csv(os.path.join(inference_dir, key + '_y.csv'))

                slide_size_df = pd.DataFrame(all_slide_size)
                slide_size_df.insert(0, "slide_name", all_slide_names)
                slide_size_df.to_csv(os.path.join(inference_dir, key + '_slide_dimensions.csv'))
            except:
                pass

        roc_auc.append(auc(fpr, tpr))
        # RanS 18.1.21
        #temp fix RanS 4.2.21
        if patch_scores.ndim == 3:
            patch_scores = np.squeeze(patch_scores[:, ind,:])
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        slide_score_mean = np.array([np.nanmean(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])
        slide_score_std = np.array([np.nanstd(patch_scores[ii, patch_scores[ii, :] > 0]) for ii in range(patch_scores.shape[0])])

        slide_score_all.append(slide_score_mean)

        #results per patient
        if patient_level:
            patient_all = []
            if dataset == 'LEUKEMIA':
                slides_data_file = r'C:\ran_data\BoneMarrow\slides_data_LEUKEMIA.xlsx'
                slides_data = pd.read_excel(slides_data_file)
            elif (dataset == 'CAT') or (dataset == 'CARMEL'):
                slides_data_file = r'C:\ran_data\Carmel_Slides_examples\add_ki67_labels\ER100_labels\slides_data_CARMEL_labeled_merged.xlsx'
                slides_data = pd.read_excel(slides_data_file)

            #for name in all_slide_names:
            for name, slide_dataset in zip(all_slide_names, all_slide_datasets):
                if slide_dataset == 'TCGA': #TCGA files
                    patient_all.append(name[8:12])
                elif slide_dataset == 'ABCTB': #ABCTB files
                    patient_all.append(name[:9])
                elif slide_dataset[:-1] == 'CARMEL':  # CARMEL files
                    patient_all.append(slides_data[slides_data['file'] == name]['patient barcode'].item())
                elif dataset == 'LEUKEMIA':
                    patient_all.append(slides_data[slides_data['file'] == name]['PatientID'].item())

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

    #temp RanS - calc BACC for each thresold
    plot_threshold = False
    if plot_threshold:
        bacc1 = np.zeros(20)
        tpr1 = np.zeros(20)
        tnr1 = np.zeros(20)
        threshs = np.arange(0, 1, 0.05)
        for ii, threshold in enumerate(threshs):
            all_preds = (all_scores > threshold).astype(int)
            true_pos1 = np.sum((all_preds == all_targets) & (all_preds == 1))
            true_neg1 = np.sum((all_preds == all_targets) & (all_preds == 0))
            bacc1[ii] = ((true_pos1 + EPS) / (total_pos + EPS) + (true_neg1 + EPS) / (total_neg + EPS)) / 2
            tpr1[ii] = true_pos1/(total_pos + EPS)
            tnr1[ii] = true_neg1 / (total_neg + EPS)
        plt.plot(threshs, tpr1,'r--')
        plt.plot(threshs, tnr1,'g-.')
        plt.plot(threshs, bacc1,'b-')
        plt.xlabel('score threshold')
        plt.legend(['tpr', 'tnr', 'BACC'],loc='lower left')

    calc_p_value = False
    if calc_p_value:
        n_iter = 10000
        rand_roc_auc = np.zeros(n_iter)
        N = len(all_labels)
        for ii in range(n_iter):
            #rand_preds = np.random.binomial(1, 0.79, size=[N, 1])
            rand_scores1 = np.random.permutation(all_scores)
            rand_roc_auc[ii] = roc_auc_score(all_targets, rand_scores1)
        p_value = np.sum(roc_auc1 <= rand_roc_auc)/n_iter

        #per patient
        n_iter = 10000
        rand_roc_auc = np.zeros(n_iter)
        N = len(patient_mean_score_df)
        for ii in range(n_iter):
            # rand_preds = np.random.binomial(1, 0.79, size=[N, 1])
            rand_scores1 = np.random.permutation(patient_mean_score_df['scores'])
            rand_roc_auc[ii] = roc_auc_score(patient_mean_score_df['targets'], rand_scores1)
        p_value_patient = np.sum(roc_auc_patient <= rand_roc_auc) / n_iter


    if patient_level:
        plt.plot(fpr_patient, tpr_patient)
        legend_labels.append(key + ' (patient AUC=' + str(round(roc_auc_patient, 3)) + ')')
    else:
        plt.plot(fpr, tpr)
        legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 3)) +')')

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
    legend_labels.append(' (all models combined patient AUC=' + str(round(roc_auc_all_patient, 3)) + ')')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.legend(legend_labels)

box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)

if patient_level:
    print('average AUC per patient: ' + str(np.round(np.mean(roc_auc_patient), 3)))
    plt.savefig(os.path.join(inference_dir, inference_name + '_inference_patient.png'), bbox_inches="tight")
else:
    print('average AUC per slide: ' + str(np.round(np.mean(roc_auc), 3)))
    plt.savefig(os.path.join(inference_dir, inference_name + '_inference.png'), bbox_inches="tight")
print('finished')
#plt.show()

