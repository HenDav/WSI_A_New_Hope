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

inference_files['test_1'] = 'runs/Exp_79-TestFold_2/Inference/Model_Epoch_11210-Folds_[1]-Tiles_50.data'
inference_files['test_2'] = 'runs/Exp_79-TestFold_2/Inference/Model_Epoch_12000-Folds_[1]-Tiles_50.data'



legend_labels = []
roc_auc = []
for _, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)
    fpr, tpr, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides = inference_data
    roc_auc.append(auc(fpr, tpr))
    #RanS 18.1.21
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

