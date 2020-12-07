import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from nets import PreActResNet50, ResNet50_2, ResNext_50
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=2, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Epochs to run')
parser.add_argument('-t', dest='transformation', action='store_true', help='Include transformations ?')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='LUNG', help='DataSet to use')

args = parser.parse_args()

"""
def simple_train(model: nn.Module, dloader_train: DataLoader, dloadet_test: DataLoader):

    print('Start Training...')
    for e in range(epoch):
        print('Epoch {}:'.format(e))
        correct_train, num_samples, train_loss = 0, 0, 0

        model.train()
        for idx, (data, target) in enumerate(tqdm(dloader_train)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)
            prob, label, weights = model(data)
            correct_train += (label == target).data.cpu().int().item()
            num_samples += 1

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            loss = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            train_loss += loss.data.cpu().item()
            # criterion = nn.CrossEntropyLoss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracy = 100 * float(correct_train) / num_samples
        print('Finished Epoch: {}, Train Accuracy: {:.2f}% ({}/{}), Loss: {:.2f}'.format(e,
                                                                                         train_accuracy,
                                                                                         correct_train,
                                                                                         num_samples,
                                                                                         train_loss))
        print('Checking post-epoch accuracy over training set...')
        correct_post = 0
        num_sample_post = 0
        with torch.no_grad():
            #model.eval()
            for idx, (data_post, target_post) in enumerate(train_loader):
                data_post, target_post = data_post.to(DEVICE), target_post.to(DEVICE)

                prob_post, label_post, weights_post = model(data_post)
                correct_post += (label_post == target_post).data.cpu().int().item()
                num_sample_post += 1

            accuracy_post_train = 100 * float(correct_post) / num_sample_post
            print('Post train accuracy: {:.2f}% ({} / {})'.format(accuracy_post_train, correct_post, num_sample_post))
"""

def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'Regular')
        all_writer.add_text('Model type', str(type(model)))
        all_writer.add_text('Data type', dloader_train.dataset.DataSet)
        all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        all_writer.add_text('Transformations', str(dloader_train.dataset.transform))

    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    best_train_loss = 1e5
    previous_epoch_loss = 1e5

    # The following 3 lines initialize variables to compute AUC for train dataset.
    best_model = None

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()
        total, correct_pos, correct_neg = 0, 0, 0
        total_pos_train, total_neg_train = 0, 0
        ### true_pos_train, true_neg_train = 0, 0
        true_labels_train, scores_train = np.zeros(0), np.zeros(0)
        correct_labeling, train_loss = 0, 0
        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, (data, target, time_list) in enumerate(tqdm(dloader_train)):
            train_start = time.time()

            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target.squeeze()
            model.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
            true_labels_train = np.concatenate((true_labels_train, target.cpu().detach().numpy()))

            total += target.size(0)
            total_pos_train += target.eq(1).sum().item()
            total_neg_train += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()
            correct_pos += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg += predicted[target.eq(0)].eq(0).sum().item()


            '''
            if target == 1:
                total_pos_train += 1
                true_labels_train[batch_idx] = 1
                if label == 1:
                    true_pos_train += 1
            elif target == 0:
                total_neg_train += 1
                true_labels_train[batch_idx] = 0
                if label == 0:
                    true_neg_train += 1
            '''

            all_writer.add_scalar('Loss', loss.item(), batch_idx + e * len(dloader_train))

            train_time = time.time() - train_start
            if print_timing:
                time_stamp = batch_idx + e * len(dloader_train)
                time_writer.add_scalar('Time/Train (iter) [Sec]', train_time, time_stamp)
                # print('Elapsed time of one train iteration is {:.2f} s'.format(train_time))
                time_list = torch.stack(time_list, 1)
                if len(time_list) == 4:
                    time_writer.add_scalar('Time/Open WSI [Sec]'     , time_list[:, 0].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[:, 1].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]' , time_list[:, 2].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[:, 3].mean().item(), time_stamp)
                else:
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[:, 0].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]', time_list[:, 1].mean().item(), time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[:, 2].mean().item(), time_stamp)

        time_epoch = (time.time() - time_epoch_start) / 60
        if print_timing:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch, e)

        train_acc = 100 * correct_labeling / total

        balanced_acc_train = 100 * (correct_pos / total_pos_train + correct_neg / total_neg_train) / 2

        fpr_train, tpr_train, _ = roc_curve(true_labels_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {}), Time: {:.0f} m'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      train_acc,
                      int(correct_labeling),
                      total,
                      time_epoch))

        previous_epoch_loss = train_loss

        if e % 5 == 0:
            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e, eval_mode=True)
            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e, eval_mode=False)
            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data(experiment=experiment, epoch=e)
        else:
            acc_test, bacc_test = None, None



        # Save model to file:
        if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
            os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))

        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()

        torch.save({'epoch': e,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'acc_test': acc_test,
                    'bacc_test': bacc_test,
                    'tile_size': TILE_SIZE,
                    'tiles_per_bag': 1},
                   os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

        '''
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = e
            best_model = model

    
    # If epochs ended - Save best model:
    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))

    try:
        best_model_state_dict = best_model.module.state_dict()
    except AttributeError:
        best_model_state_dict = best_model.state_dict()
    
    torch.save({'epoch': best_epoch,
                'model_state_dict': best_model_state_dict,
                'best_train_loss': best_train_loss,
                'best_train_acc': best_train_acc,
                'tile_size': TILE_SIZE,
                'tiles_per_bag': 1},
               os.path.join(args.output_dir, 'Model_CheckPoints', 'best_model_Ep_' + str(best_epoch) + '.pt'))
    '''
    all_writer.close()
    time_writer.close()



def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, DEVICE, epoch: int, eval_mode: bool = False):
    num_correct = 0
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0
    true_labels, scores = np.zeros(0), np.zeros(0)

    test_loss, total = 0, 0

    # TODO: eval mode changes the mode of dropout and batchnorm layers. Activate this line only after the model is fully learned
    if eval_mode:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for idx, (data, targets, time_list) in enumerate(data_loader):
            data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)

            model.to(DEVICE)

            out = model(data)

            outputs = torch.nn.functional.softmax(out, dim=1)
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            scores = np.concatenate((scores, outputs[:, 1].cpu().detach().numpy()))
            true_labels = np.concatenate((true_labels, targets.cpu().detach().numpy()))

            total += targets.size(0)
            num_correct += predicted.eq(targets).sum().item()
            total_pos += targets.eq(1).sum().item()
            total_neg += targets.eq(0).sum().item()
            true_pos += predicted[targets.eq(1)].eq(1).sum().item()
            true_neg += predicted[targets.eq(0)].eq(0).sum().item()

        acc = 100 * float(num_correct) / total
        balanced_acc = 100 * (true_pos / total_pos + true_neg / total_neg) / 2

        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)


        # TODO: instead of using the train parameter it is possible to simply check data_loader.dataset.train attribute
        if data_loader.dataset.train:
            writer_string = 'Train_2'
        else:
            if eval_mode:
                writer_string = 'Test (eval mode)'
            else:
                writer_string = 'Test (train mode)'

        """
        writer_test.add_scalar('Accuracy', acc, epoch)
        writer_test.add_scalar('Balanced Accuracy', balanced_acc, epoch)
        writer_test.add_scalar('Roc-Auc', roc_auc, epoch)
        """

        writer_all.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer_all.add_scalar(writer_string + '/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)

        print('{}: Accuracy of {:.2f}% ({} / {}) over Test set'.format('EVAL mode' if eval_mode else 'TRAIN mode', acc, num_correct, total))
    model.train()
    return acc, balanced_acc



##################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()
    # Data type definition:
    ### DATA_TYPE = 'WSI'
    ### MODEL_TYPE = 'REG'

    # Tile size definition:
    TILE_SIZE =128
    timing = True
    data_path = '/Users/wasserman/Developer/All data - outer scope'

    if sys.platform == 'linux':
        TILE_SIZE = 256
        data_path = '/home/womer/project/All Data'

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        args.output_dir, experiment = utils.run_data(test_fold=args.test_fold,
                                                     transformations=args.transformation,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=1,
                                                     DX=args.dx,
                                                     DataSet=args.dataset)
    else:
        args.output_dir, args.test_fold, args.transformation, TILE_SIZE, TILES_PER_BAG, args.dx, args.dataset = utils.run_data(experiment=args.experiment)
        experiment = args.experiment

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()
    #cpu_available = 1

    # Get data:
    train_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      test_fold=args.test_fold,
                                      train=True,
                                      print_timing=timing,
                                      transform=args.transformation,
                                      DX=args.dx)

    test_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                     tile_size=TILE_SIZE,
                                     test_fold=args.test_fold,
                                     train=False,
                                     print_timing=False,
                                     transform=False,
                                     DX=args.dx)


    train_loader = DataLoader(train_dset, batch_size=10, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader  = DataLoader(test_dset, batch_size=50, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)


    # Load model
    #model = ResNext_50()
    model = PreActResNet50()
    # model = ResNet50_2()
    utils.run_data(experiment=experiment, model='PreActResNet50()')


    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment is not 0:
        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

        model.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))

    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)

    if DEVICE.type == 'cuda':
        cudnn.benchmark = True

    if args.experiment is not 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # simple_train(model, train_loader, test_loader)
    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=timing)
