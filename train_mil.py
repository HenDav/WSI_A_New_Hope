import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import model
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('--output_dir', default='runs-output', type=str, help='output directory for TensorBoard data')
parser.add_argument('--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=1, type=int, help='Epochs to run')

args = parser.parse_args()



#def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, print_timing: bool = True):
def train(print_timing: bool = True):
    """
    This function trains the model
    :return:
    """
    print('Start Training...')
    best_train_acc = 0
    best_train_loss = 1e5
    previous_epoch_loss = 1e5

    # The following 3 lines initialize variables to computer AUC for train dataset.
    total_pos_train, total_neg_train = 0, 0
    true_pos_train, true_neg_train = 0, 0
    true_labels_train, scores_train = np.zeros(len(train_loader)), np.zeros(len(train_loader))

    best_model = None

    for e in range(epoch):
        correct_labeling_total, train_loss = 0, 0
        model.train()
        print('Epoch {}:'.format(e))
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            start = time.time()
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)
            prob, label, _ = model(data)

            if target == 1:
                total_pos_train += 1
                if label == 1:
                    true_pos_train += 1
            else:
                total_neg_train += 1
                if target == 0:
                    true_neg_train += 1

            true_labels_train[batch_idx] = target.cpu().detach().numpy()[0][0]
            scores_train[batch_idx] = prob.cpu().detach().numpy()[0][0]

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.data[0], batch_idx + e * len(train_loader))
            # error = 1. - label.eq(target).cpu().float().mean().item()
            # Calculate training accuracy
            correct_labeling = label.eq(target).cpu().float().mean().item()  # TODO: I think the mean() calculation can be removed (each batch consists of only one example
            correct_labeling_total += correct_labeling

            if print_timing:
                end = time.time()
                print('Elapsed time of one train iteration is {:.0f} s'.format(end - start))

        train_accuracy = correct_labeling_total / len(train_loader) * 100

        balanced_acc_train = 100 * (true_pos_train / total_pos_train + true_neg_train / total_neg_train) / 2

        fpr_train, tpr_train, _ = roc_curve(true_labels_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)

        writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        writer.add_scalar('Train/Accuracy', train_accuracy, e)
        print('Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {})'
              .format(e + 1,
                      train_loss.cpu().numpy()[0],
                      previous_epoch_loss - train_loss.cpu().numpy()[0],
                      train_accuracy,
                      correct_labeling_total,
                      len(train_loader)))

        previous_epoch_loss = train_loss.cpu().numpy()[0]

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_acc = train_accuracy
            best_model = model

        model.eval()
        # test_acc = check_accuracy(model, test_loader)
        acc_test, bacc_test = check_accuracy(test_loader, epoch=e, train=False)
        model.train()

        # Save model to file:
        if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
            os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))

        torch.save({'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data[0],
                    'acc_test': acc_test,
                    'bacc_test': bacc_test},
                   os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data.pt'))

    return best_model, best_train_acc, best_train_loss.cpu().numpy()[0]


#def check_accuracy(model: nn.Module, data_loader: DataLoader):
def check_accuracy(data_loader: DataLoader, epoch:int, train: bool = False):
    """
       if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test sett')
    """

    num_correct = 0
    num_samples = 0
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0
    true_labels, scores = np.zeros(len(data_loader)), np.zeros(len(data_loader))

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if target == 1:
                total_pos += 1
            else:
                total_neg += 1

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)
            prob, label, _ = model(data)
            num_correct += (label == target).cpu().int().item()
            num_samples += label.size(0)

            true_labels[idx] = target.cpu().detach().numpy()[0][0]
            scores[idx] = prob.cpu().detach().numpy()[0][0]

            if target == 1 and label == 1:
                true_pos += 1
            elif target == 0 and label == 0:
                true_neg += 1

        acc = 100 * float(num_correct) / num_samples
        balanced_acc = 100 * (true_pos / total_pos + true_neg / total_neg) / 2

        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)


        # TODO: instead of using the train parameter it is possible to simply check data_loader.dataset.train attribute
        if train:
            writer_string = 'Train'
        else:
            writer_string = 'Test'

        writer.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer.add_scalar(writer_string + '/Balanced Accuracy', balanced_acc, epoch)
        writer.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)

        print('Got {} / {} correct ({:.2f}%) over Test set'.format(num_correct, num_samples, acc))
    return acc, balanced_acc




##################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Tile size definition:
    TILE_SIZE = 256

    # Get data:
    train_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE,
                                      num_of_tiles_from_slide=50,
                                      test_fold=args.test_fold,
                                      train=True,
                                      print_timing=False,
                                      transform=None)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=cpu_available, pin_memory=False)

    # TODO: check it it's possible to put all the tiles of a WSI in the test loader - seems like there will a memory problem
    #  with the GPUs. The solution might be to run each batch of tiles through the first section of the net till we have all
    #  the feature vectors and then proceed to the attention weight part for all feature vectors together.

    test_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE,
                                     num_of_tiles_from_slide=50,
                                     test_fold=args.test_fold,
                                     train=True,
                                     print_timing=False,
                                     transform=None)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=True, num_workers=cpu_available, pin_memory=False)

    # Load model
    model = model.GatedAttention(tile_size=TILE_SIZE)
    if DEVICE.type == 'cuda':
        # model = nn.parallel.DistributedDataParallel(model)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            # TODO: check how to work with nn.parallel.DistributedDataParallel
            print('Using {} GPUs'.format(torch.cuda.device_count()))
        cudnn.benchmark = True

    #  optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)  # This is the optimizer from the paper Gil reviewed

    writer = SummaryWriter(args.output_dir)

    epoch = args.epochs
    best_model, best_train_error, best_train_loss = train(print_timing=False)
