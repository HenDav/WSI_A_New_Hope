import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from nets import ResNet50_GatedAttention
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
# parser.add_argument('--output_dir', default='runs-output', type=str, help='output directory for TensorBoard data')
parser.add_argument('--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Epochs to run')
parser.add_argument('-t', '--transformation', default=True, type=bool, help='Include transformations ?')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')

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
    train_writer = SummaryWriter(os.path.join(writer_folder, 'train'))
    test_writer = SummaryWriter(os.path.join(writer_folder, 'test'))

    print('Start Training...')
    best_train_loss = 1e5
    previous_epoch_loss = 1e5

    # The following 3 lines initialize variables to compute AUC for train dataset.
    total_pos_train, total_neg_train = 0, 0
    true_pos_train, true_neg_train = 0, 0
    true_labels_train, scores_train = np.zeros(len(dloader_train)), np.zeros(len(dloader_train))

    best_model = None

    for e in range(from_epoch, epoch + from_epoch):
        correct_labeling, train_loss = 0, 0
        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, (data, target) in enumerate(tqdm(dloader_train)):
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)
            start = time.time()
            prob, label, _ = model(data)

            if target == 1:
                total_pos_train += 1
                true_labels_train[batch_idx] = 1
                if label == 1:
                    true_pos_train += 1
            elif target == 0:
                total_neg_train += 1
                true_labels_train[batch_idx] = 0
                if target == 0:
                    true_neg_train += 1

            scores_train[batch_idx] = prob.cpu().detach().numpy()[0][0]

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_writer.add_scalar('Loss', loss.data[0], batch_idx + e * len(dloader_train))

            # Calculate training accuracy
            correct_labeling += (label == target).data.cpu().int().item()

            if print_timing:
                end = time.time()
                print('Elapsed time of one train iteration is {:.0f} s'.format(end - start))

        train_acc = 100 * correct_labeling / len(dloader_train)

        balanced_acc_train = 100 * (true_pos_train / total_pos_train + true_neg_train / total_neg_train) / 2

        fpr_train, tpr_train, _ = roc_curve(true_labels_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)

        train_writer.add_scalar('Balanced Accuracy', balanced_acc_train, e)
        train_writer.add_scalar('Roc-Auc', roc_auc_train, e)
        train_writer.add_scalar('Loss Per Epoch', train_loss, e)
        train_writer.add_scalar('Accuracy', train_acc, e)

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {})'
              .format(e,
                      train_loss.cpu().numpy()[0],
                      previous_epoch_loss - train_loss.cpu().numpy()[0],
                      train_acc,
                      int(correct_labeling),
                      len(train_loader)))

        previous_epoch_loss = train_loss.cpu().numpy()[0]

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = e
            best_model = model

        if e % 5 == 0:
            acc_test, bacc_test = check_accuracy(model, dloader_test, test_writer, all_writer, DEVICE, e)
        else:
            acc_test, bacc_test = None, None

        # Save model to file:
        if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
            os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))

        torch.save({'epoch': e,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.data[0],
                    'acc_test': acc_test,
                    'bacc_test': bacc_test},
                   os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

    # If epochs ended - Save best model:
    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))
    torch.save({'epoch': best_epoch,
                'model_state_dict': best_model.module.state_dict(),
                'best_train_loss': best_train_loss,
                'best_train_acc': best_train_acc},
               os.path.join(args.output_dir, 'Model_CheckPoints', 'best_model_Ep_' + str(best_epoch) + '.pt'))

    train_writer.close()
    test_writer.close()
    all_writer.close()

    return best_model, best_train_acc, best_train_loss.cpu().numpy()[0]


def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_test, writer_all, DEVICE, epoch: int):
#def check_accuracy(model: nn.Module, data_loader: DataLoader, writer, DEVICE, epoch:int, train_string: str ='', train: bool = False):
#def check_accuracy(data_loader: DataLoader, epoch:int, train: bool = False):
    """
       if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test sett')
    """
    num_correct = 0
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0
    true_labels, scores = np.zeros(len(data_loader)), np.zeros(len(data_loader))

    # TODO: eval mode changes the mode of dropout and batchnorm layers. Activate this line only after the model is fully learned
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if target == 1:
                total_pos += 1
                true_labels[idx] = 1
            elif target == 0:
                total_neg += 1
                true_labels[idx] = 0

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            prob, label, _ = model(data)

            num_correct += (label == target).data.cpu().int().item()

            scores[idx] = prob.cpu().detach().numpy()[0][0]

            """
            print('target={}, label={}'.format(target, label))
            print('target is {} equal to label'.format('not' if (target != label) else ''))
            """
            if target == 1 and label == 1:
                true_pos += 1
            elif target == 0 and label == 0:
                true_neg += 1

        acc = 100 * float(num_correct) / len(data_loader)
        balanced_acc = 100 * (true_pos / total_pos + true_neg / total_neg) / 2

        fpr, tpr, _ = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)


        # TODO: instead of using the train parameter it is possible to simply check data_loader.dataset.train attribute
        if data_loader.dataset.train:
            writer_string = 'Train_2'
        else:
            writer_string = 'Test'

        writer_test.add_scalar('Accuracy', acc, epoch)
        writer_test.add_scalar('Balanced Accuracy', balanced_acc, epoch)
        writer_test.add_scalar('Roc-Auc', roc_auc, epoch)

        writer_all.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer_all.add_scalar(writer_string + '/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)

        print('Accuracy of {:.2f}% ({} / {}) over Test set'.format(acc, num_correct, len(data_loader)))
    model.train()
    return acc, balanced_acc


def infer(model: nn.Module, dloader: DataLoader, DEVICE):
    """
    This function does inference of a slide from ALL the tiles of the slide
    :param model: model to be used for inference
    :param dloader_test:
    :param DEVICE:
    :return:
    """

    print('Starting Inference...')

    model.eval()
    model.infer = True
    torch.no_grad()

    # The following 3 lines initialize variables to compute AUC for train dataset.
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0
    correct_labeling = 0
    true_labels, scores_train = np.zeros(len(dloader)), np.zeros(len(dloader))

    for batch_idx, (data, target) in enumerate(tqdm(dloader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        model.to(DEVICE)
        model.get_features = True

        features = model(data)

        if target == 1:
            total_pos += 1
            true_labels[batch_idx] = 1
            if label == 1:
                true_pos += 1
        elif target == 0:
            total_neg += 1
            true_labels[batch_idx] = 0
            if target == 0:
                true_neg += 1

        scores_train[batch_idx] = prob.cpu().detach().numpy()[0][0]

    torch.enable_grad


##################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Tile size definition:
    TILE_SIZE = 256
    TILES_PER_BAG = 50

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        args.output_dir = utils.run_data(test_fold=args.test_fold,
                                         transformations=args.transformation,
                                         tile_size=TILE_SIZE,
                                         tiles_per_bag=TILES_PER_BAG)
    else:
        args.output_dir, args.test_fold, args.transformation, TILE_SIZE, TILES_PER_BAG = utils.run_data(experiment=args.experiment)

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    """
    # Get data augmentation:
    if args.transformation:
        transform = utils.get_transform()
    else:
        transform = False
    """

    # Get data:
    train_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE,
                                      num_of_tiles_from_slide=TILES_PER_BAG,
                                      test_fold=args.test_fold,
                                      train=True,
                                      print_timing=True,
                                      transform=args.transformation)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=cpu_available, pin_memory=False)

    # TODO: check it it's possible to put all the tiles of a WSI in the test loader - seems like there will a memory problem
    #  with the GPUs. The solution might be to run each batch of tiles through the first section of the net till we have all
    #  the feature vectors and then proceed to the attention weight part for all feature vectors together.

    test_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE,
                                     num_of_tiles_from_slide=TILES_PER_BAG,
                                     test_fold=args.test_fold,
                                     train=False,
                                     print_timing=False,
                                     transform=False)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=cpu_available, pin_memory=False)

    # Load model
    model = ResNet50_GatedAttention() #tile_size=TILE_SIZE)
    model = nn.DataParallel(model)  # TODO: Remove this line after the first runs

    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment is not 0:
        print('Loading pre-saved model')
        model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

        model.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))


    # TODO: Check if the following can be written in such a way that there is only one if statement
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)

    if DEVICE.type == 'cuda':
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            # TODO: check how to work with nn.parallel.DistributedDataParallel
            print('Using {} GPUs'.format(torch.cuda.device_count()))
        cudnn.benchmark = True

    if args.experiment is not 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    # simple_train(model, train_loader, test_loader)
    best_model, best_train_error, best_train_loss = train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=True)
