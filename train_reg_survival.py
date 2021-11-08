import utils
import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import sys
import pandas as pd
from sklearn.utils import resample
import smtplib, ssl
import psutil
from Cox_Loss import Cox_loss
from sksurv.metrics import concordance_index_censored

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='Survival Synthetic', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='Survival_Time', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('--batch_size', default=20, type=int, help='size of batch')
parser.add_argument('--model', default='PreActResNets.PreActResNet50_Ron()', type=str, help='net to use')
#parser.add_argument('--model', default='nets.ResNet50(pretrained=True)', type=str, help='net to use')

parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')

parser.add_argument('--er_eq_pr', action='store_true', help='while training, take only er=pr examples') #RanS 27.6.21

parser.add_argument('--slide_per_block', action='store_true', help='for carmel, take only one slide per block') #RanS 17.8.21



args = parser.parse_args()

EPS = 1e-7


def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'Regular')
        all_writer.add_text('Model type', str(type(model)))
        if args.dataset != 'Survival Synthetic':
            all_writer.add_text('Data type', dloader_train.dataset.DataSet)
            all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
            all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
            all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
            all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    previous_epoch_loss = 1e5

    for e in range(from_epoch, epoch):
        if args.target == 'Survival_Binary':
            total_train, correct_pos_train, correct_neg_train = 0, 0, 0
            total_pos_train, total_neg_train = 0, 0
            true_targets_train, scores_train = np.zeros(0), np.zeros(0)

        all_targets, all_outputs, all_censored, all_cont_targets = [], [], [], []
        train_loss = 0

        #slide_names = []
        print('Epoch {}:'.format(e))

        # RanS 11.7.21
        process = psutil.Process(os.getpid())
        #print('RAM usage:', process.memory_info().rss/1e9, 'GB')

        model.train()
        model.to(DEVICE)

        for batch_idx, minibatch in enumerate(tqdm(dloader_train)):
            time_stamp = batch_idx + e * len(train_loader)

            data = minibatch['Features']
            target = minibatch['Binary Target']
            censored = minibatch['Censored']
            target_cont = minibatch['Target']

            all_targets.extend(target.numpy())
            all_cont_targets.extend(target_cont.numpy())
            all_censored.extend(censored.numpy())

            optimizer.zero_grad()

            if args.target == 'Survival_Binary':

                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)

                loss = criterion(outputs, target)

                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(- outputs[:, 1].detach().cpu().numpy())

                scores_train = np.concatenate((scores_train, outputs[:, 1].detach().cpu().numpy()))
                true_targets_train = np.concatenate((true_targets_train, target.detach().cpu().numpy()))
                total_train += target.size(0)
                total_pos_train += target.eq(1).sum().item()
                total_neg_train += target.eq(0).sum().item()
                correct_pos_train += predicted[target.eq(1)].eq(1).sum().item()
                correct_neg_train += predicted[target.eq(0)].eq(0).sum().item()

            elif args.target == 'Survival_Time':

                data, target_cont = data.to(DEVICE), target_cont.to(DEVICE)
                outputs = model(data)

                loss = criterion(outputs, target_cont, censored)
                outputs = torch.reshape(outputs, [outputs.size(0)])
                all_outputs.extend(outputs.detach().cpu().numpy())

            else:
                Exception('Not implemented')

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            all_writer.add_scalar('Train/Loss per Minibatch', loss, time_stamp)

        # Compute C index
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_cont_targets, all_outputs)
        # Compute AUC:
        if args.target == 'Survival_Binary':
            fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
            roc_auc_train = auc(fpr_train, tpr_train)

        elif args.target == 'Survival_Time':
            # Compute AUC:
            not_censored_indices = np.where(np.array(all_censored) == False)
            relevant_targets = np.array(all_targets)[not_censored_indices]
            relevant_outputs = np.array(all_outputs)[not_censored_indices]
            fpr_train, tpr_train, _ = roc_curve(relevant_targets, - relevant_outputs)
            roc_auc_train = auc(fpr_train, tpr_train)

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train C-index: {:.3f}, Train AUC: {:.3f}'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      c_index,
                      100 * roc_auc_train))

        all_writer.add_scalar('Train/Loss Per Epoch Per Instance', train_loss / len(all_targets), e)
        all_writer.add_scalar('Train/C-index Per Epoch', c_index, e)
        all_writer.add_scalar('Train/AUC Per Epoch', roc_auc_train, e)

        previous_epoch_loss = train_loss

        # Update 'Last Epoch' at run_data.xlsx file:
        utils.run_data(experiment=experiment, epoch=e)

        if e % args.eval_rate == 0:
            check_accuracy(e, all_writer)
            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'tile_size': TILE_SIZE,
                        'tiles_per_bag': 1},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))
            print('saved checkpoint to', args.output_dir) #RanS 23.6.21


    all_writer.close()

    if print_timing:
        time_writer.close()


def check_accuracy(epoch: int, all_writer):
    total_test, correct_pos_test, correct_neg_test = 0, 0, 0
    total_pos_test, total_neg_test = 0, 0
    true_targets_test, scores_test = np.zeros(0), np.zeros(0)

    all_outputs, all_targets, all_censored, all_cont_targets = [], [], [], []
    test_loss = 0

    model.eval()
    model.to(DEVICE)

    with torch.no_grad():
        for idx, minibatch in enumerate(test_loader):

            data = minibatch['Features']
            target = minibatch['Binary Target']
            censored = minibatch['Censored']
            target_cont = minibatch['Target']

            all_targets.extend(target.numpy())
            all_cont_targets.extend(target_cont.numpy())
            all_censored.extend(censored.numpy())

            if args.target == 'Survival_Binary':

                data, target = data.to(device=DEVICE), target.to(device=DEVICE)
                outputs = model(data)

                loss = criterion(outputs, target)

                outputs = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_outputs.extend(- outputs[:, 1].detach().cpu().numpy())

                scores_test = np.concatenate((scores_test, outputs[:, 1].detach().cpu().numpy()))
                true_targets_test = np.concatenate((true_targets_test, target.detach().cpu().numpy()))
                total_test += target.size(0)
                total_pos_test += target.eq(1).sum().item()
                total_neg_test += target.eq(0).sum().item()
                correct_pos_test += predicted[target.eq(1)].eq(1).sum().item()
                correct_neg_test += predicted[target.eq(0)].eq(0).sum().item()

            elif args.target == 'Survival_Time':

                data, target_cont = data.to(device=DEVICE), target_cont.to(device=DEVICE)
                outputs = model(data)

                loss = criterion(outputs, target_cont, censored)
                outputs = torch.reshape(outputs, [outputs.size(0)])
                all_outputs.extend(outputs.detach().cpu().numpy())

            else:
                Exception('Not implemented')

            test_loss += loss.item()

        # Compute C index
        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_cont_targets, all_outputs)
        # Compute AUC:
        if args.target == 'Survival_Binary':
            fpr_test, tpr_test, _ = roc_curve(true_targets_test, scores_test)
            roc_auc_test = auc(fpr_test, tpr_test)

        elif args.target == 'Survival_Time':

            # Compute AUC:
            not_censored_indices = np.where(np.array(all_censored) == False)
            relevant_targets = np.array(all_targets)[not_censored_indices]
            relevant_outputs = np.array(all_outputs)[not_censored_indices]
            fpr_test, tpr_test, _ = roc_curve(relevant_targets, - relevant_outputs)
            roc_auc_test = auc(fpr_test, tpr_test)

        print('Test C-index: {:.3f}, Test AUC: {:.3f}'.format(c_index, 100 * roc_auc_test))

        all_writer.add_scalar('Test/C-index Per Epoch', c_index, epoch)
        all_writer.add_scalar('Test/Loss Per Epoch Per Instance', test_loss / len(all_targets), epoch)
        all_writer.add_scalar('Test/AUC Per Epoch', roc_auc_test, epoch)

    model.train()


########################################################################################################
########################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Tile size definition:
    TILE_SIZE = 128

    if sys.platform == 'linux' or sys.platform == 'win32':
        TILE_SIZE = 256

    # Saving/Loading run meta data to/from file:
    if args.experiment == 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                                     #transformations=args.transformation,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=1,
                                                     DX=args.dx,
                                                     DataSet_name=args.dataset,
                                                     Receptor=args.target,
                                                     num_bags=args.batch_size)

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']
    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE, tiles_per_bag, \
        args.batch_size, args.dx, args.dataset, args.target, args.model, args.mag =\
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], run_data_output['Tile Size'],\
            run_data_output['Tiles Per Bag'], run_data_output['Num Bags'], run_data_output['DX'], run_data_output['Dataset Name'],\
            run_data_output['Receptor'], run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        print('args.dataset:', args.dataset)
        print('args.target:', args.target)
        print('args.args.batch_size:', args.batch_size)
        print('args.output_dir:', args.output_dir)
        print('args.test_fold:', args.test_fold)
        print('args.transform_type:', args.transform_type)
        print('args.dx:', args.dx)

        experiment = args.experiment

    # Get number of available CPUs and compute number of workers:
    cpu_available = utils.get_cpu()
    num_workers = cpu_available

    print('num workers = ', num_workers)

    # Get data:
    if args.target == 'Survival_Binary':
        binary_target = True
    elif args.target == 'Survival_Time':
        binary_target = False

    if args.dataset == 'Survival Synthetic':
        train_dset = datasets.C_Index_Test_Dataset(train=True,
                                                   binary_target=binary_target)
        test_dset = datasets.C_Index_Test_Dataset(train=False,
                                                  binary_target=binary_target)
    else:
        print('NOT Implemented')
        exit()

    sampler = None
    do_shuffle = True

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle,
                              num_workers=num_workers, pin_memory=True, sampler=sampler)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size*2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    if args.dataset != 'Survival Synthetic':
        transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
        utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Load model
    if args.dataset == 'Survival Synthetic':
        if args.target == 'Survival_Time':
            model = nn.Linear(8, 1)
            model.model_name = 'Survival_Synthetic_Continous'
        elif args.target == 'Survival_Binary':
            model = nn.Linear(8, 2)
            model.model_name = 'Survival_Synthetic_Binary'

    else:
        print('NOT Implemented')
        exit()

        '''model = eval(args.model)
    if args.target == 'Survival_Time' and args.dataset != 'Survival Synthetic':
        model.change_num_classes(num_classes=1)  # This will convert the liner (classifier) layer into the beta layer'''

    # Save model data and data-set size to run_data.xlsx file (Only if this is a new run).
    if args.experiment == 0:
        utils.run_data(experiment=experiment, model=model.model_name)
        if args.dataset != 'Survival Synthetic':
            utils.run_data(experiment=experiment, DataSet_size=(train_dset.real_length, test_dset.real_length))
            utils.run_data(experiment=experiment, DataSet_Slide_magnification=train_dset.desired_magnification)

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment != 0:
        print('Loading pre-saved model...')
        if from_epoch == 0: #RanS 25.7.21, load last epoch
            model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                        'Model_CheckPoints',
                                                        'model_data_Last_Epoch.pt'),
                                           map_location='cpu')
            from_epoch = model_data_loaded['epoch'] + 1
        else:
            model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                        'Model_CheckPoints',
                                                        'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
            from_epoch = args.from_epoch + 1
        model.load_state_dict(model_data_loaded['model_state_dict'])

        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, from_epoch))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if DEVICE.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

        # RanS 28.1.21
        # https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
        if args.time:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    if args.experiment != 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)


    if args.target == 'Survival_Time':
        criterion = Cox_loss
    else:
        criterion = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
