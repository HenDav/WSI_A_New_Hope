"""
This file contains supplementary functions to support training and inference of multiple models
"""


import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import List
import torch


def write_basic_data_to_writer(writers, experiment_numbers, models, train_types, data_loader_train, data_loader_test):
    for index in range(len(writers)):
        writer = writers[index]
        writer.add_text('Experiment No.', str(experiment_numbers[index]))
        writer.add_text('Train type', train_types[index])
        writer.add_text('Model Name', str(type(models[index])))
        writer.add_text('Data type', data_loader_train.dataset.DataSet)
        writer.add_text('Train Folds', str(data_loader_train.dataset.folds).strip('[]'))
        writer.add_text('Test Folds', str(data_loader_test.dataset.folds).strip('[]'))
        writer.add_text('Transformations', str(data_loader_train.dataset.transform))
        writer.add_text('Receptor Type', str(data_loader_train.dataset.target_kind))


def create_writers(output_dirs):
    writers = []
    for index in range(len(output_dirs)):
        writer_folder = os.path.join(output_dirs[index], 'writer')
        writers.append(SummaryWriter(os.path.join(writer_folder, 'all')))

    return writers


def init_optimizers_adam(models, learning_rate: float = 1e-5, weight_decay: float = 5e-5):
    optimizers = []
    for index in range(len(models)):
        optimizers.append(optim.Adam(models[index].parameters(), lr=learning_rate, weight_decay=weight_decay))

    return optimizers


def init_performance_parameters(num_models):
    fixes = ['train', 'test']

    train_parameters = {}

    train_parameters['epoch'] = 0

    for postfix in fixes:
        # These parameters are the same for all models
        train_parameters['targets_' + postfix] = np.zeros(0)
        train_parameters['total_' + postfix] = 0
        train_parameters['total_pos_' + postfix] = 0
        train_parameters['total_neg_' + postfix] = 0
        train_parameters['time_' + postfix] = 0

        # These parameters are individual for each model and are calculated every mini batch
        train_parameters['correct_pos_' + postfix] = [0] * num_models
        train_parameters['correct_neg_' + postfix] = [0] * num_models
        train_parameters['scores_' + postfix] = [np.zeros(0)] * num_models
        train_parameters['correct_labeling_' + postfix] = [0] * num_models
        train_parameters['loss_' + postfix] = [0] * num_models
        train_parameters['previous_loss_' + postfix] = [1e5] * num_models

    return train_parameters


def write_performance_data(writers: list, performance_parameters: dict, is_train: bool = True):
    postfix: str = 'train' if is_train else 'test'
    prefix: str  = 'Train' if is_train else 'Test'

    epoch = performance_parameters['epoch']

    for index in range(len(writers)):
        # Compute performance parameters
        accuracy = 100 * performance_parameters['correct_labeling_' + postfix][index] / performance_parameters['total_' + postfix]
        bacc = 100 * (performance_parameters['correct_pos_' + postfix][index] / performance_parameters['total_pos_' + postfix] +
                      performance_parameters['correct_neg_' + postfix][index] / performance_parameters['total_neg_' + postfix]
                      ) / 2

        fpr, tpr, _ = roc_curve(performance_parameters['targets_' + postfix], performance_parameters['scores_' + postfix][index])
        roc_auc = auc(fpr, tpr)

        # Write performance parameters
        writers[index].add_scalar(prefix + '/Balanced Accuracy', bacc, epoch)
        writers[index].add_scalar(prefix + '/Roc-Auc', roc_auc, epoch)
        writers[index].add_scalar(prefix + '/Accuracy', accuracy, epoch)
        if is_train:
            writers[index].add_scalar(prefix + '/Loss Per Epoch', performance_parameters['loss_' + postfix][index], epoch)

        # Print performance parameters:
        if is_train:
            if index == 0:
                print('Finished Epoch: {}'.format(epoch))
            print(' Model No. {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train AUC per patch: {:.2f} , Time: {:.0f} m'
                  .format(index,
                          performance_parameters['loss_train'][index],
                          performance_parameters['previous_loss_train'][index] - performance_parameters['loss_train'][index],
                          roc_auc,
                          performance_parameters['time_train']))

            # Update previous_loss entry:
            performance_parameters['previous_loss_train'][index] = performance_parameters['loss_train'][index]
        else:
            print(' * Testing Model {} : Test AUC per patch: {:.2f}'.
                  format(index,
                         roc_auc))


def save_models(models: List[torch.nn.Module], optimizers: list, output_dir: list, epoch: int):
    for index in range(len(models)):
        model, optimizer = models[index], optimizers[index]

        if not os.path.isdir(os.path.join(output_dir[index], 'Model_CheckPoints')):
            os.mkdir(os.path.join(output_dir[index], 'Model_CheckPoints'))

        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        torch.save({'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(output_dir[index], 'Model_CheckPoints', 'model_data_Epoch_' + str(epoch) + '.pt'))
