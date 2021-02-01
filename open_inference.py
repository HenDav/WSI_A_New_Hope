from sklearn.metrics import auc, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
import pickle
from cycler import cycler
import numpy as np


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
inference_files['ex38_epoch1080_test_2'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1080-Folds_[1]-Tiles_2.data'
inference_files['ex38_epoch1060_test_2'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1060-Folds_[1]-Tiles_2.data'
inference_files['ex38_epoch1040_test_2'] = r'C:\Pathnet_results\MIL_general_try4\exp36,38,42,43\exp38\Inference\Model_Epoch_1040-Folds_[1]-Tiles_2.data'
infer_type = 'REG'

legend_labels = []
roc_auc = []
for _, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)

    if infer_type == 'REG':
        fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides, patch_scores = inference_data
        roc_auc.append(auc(fpr, tpr))
        # RanS 18.1.21
        # all_scores = np.max(patch_scores, axis=1) #maxpool - temp! RanS 20.1.21
        slide_score_std = patch_scores.std(axis=1)
        slide_score_mean = patch_scores.mean(axis=1)
        slide_conf = np.abs(slide_score_mean - 0.5)
        correct_array = all_labels == all_targets
        std_wrong = slide_score_std[~correct_array]
        std_correct = slide_score_std[correct_array]
        conf_wrong = slide_conf[~correct_array]
        conf_correct = slide_conf[correct_array]
        # plt.hist([conf_wrong, conf_correct])
        # plt.legend(['confidence - wrong predictions','confidence - correct predictions'])

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

