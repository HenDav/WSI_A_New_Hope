from sklearn.metrics import auc
from matplotlib import pyplot as plt
import pickle
from cycler import cycler


custom_cycler = (cycler(color=['#377eb8', '#ff7f00', '#4daf4a',
                                    '#f781bf', '#a65628', '#984ea3',
                                    '#999999', '#e41a1c', '#dede00']) +
                      cycler(linestyle=['solid', 'dashed', 'dotted',
                                        'dashdot', 'solid', 'dashed',
                                        'dotted', 'dashdot', 'dashed']))
fig1, ax1 = plt.subplots()
ax1.set_prop_cycle(custom_cycler)




inference_files = {}

inference_files['train_50'] = 'Data from gipdeep/runs/141/Inference/Model_Epoch_3555-Folds_[1]-Tiles_50.data'
inference_files['test_50'] = 'Data from gipdeep/runs/141/Inference/Model_Epoch_3555-Folds_[2]-Tiles_50.data'
inference_files['test_500'] = 'Data from gipdeep/runs/141/Inference/Model_Epoch_3555-Folds_[2]-Tiles_500.data'




legend_labels = []
roc_auc = []
for _, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)
    fpr_train, tpr_train, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides = inference_data
    roc_auc.append(auc(fpr_train, tpr_train))

    plt.plot(fpr_train, tpr_train)
    legend_labels.append(key + ' (AUC=' + str(round(roc_auc[-1], 2)) +')')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(b=True)
plt.show()
