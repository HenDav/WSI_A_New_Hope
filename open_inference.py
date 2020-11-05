import numpy as np
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import pickle

inference_files = {}
inference_files['train_500'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[1]-Tiles_500.data'
inference_files['train_1000'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[1]-Tiles_1000.data'
inference_files['test_500'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[2]-Tiles_500.data'
inference_files['test_500_Eval'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[2]-Tiles_500-EVAL_MODE.data'
inference_files['test_1000_Eval'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[2]-Tiles_1000-EVAL_MODE.data'
inference_files['test_1000_Train'] = 'Data from gipdeep/runs/29/Inference/Model_Epoch_921-Folds_[2]-Tiles_1000-TRAIN_MODE.data'


legend_labels = []
roc_auc = []
for _, key in enumerate(inference_files.keys()):
    with open(inference_files[key], 'rb') as filehandle:
        inference_data = pickle.load(filehandle)
    fpr_train, tpr_train, all_labels,  all_targets, all_scores, total_pos, true_pos, total_neg, true_neg, num_slides = inference_data
    roc_auc.append(auc(fpr_train, tpr_train))

    plt.plot(fpr_train, tpr_train)
    legend_labels.append(key)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(legend_labels)
plt.show()
print(roc_auc)

