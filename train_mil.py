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

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('--output_dir', default='runs-output', type=str, help='output directory for TensorBoard data')
parser.add_argument('--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=1, type=int, help='Epochs to run')

args = parser.parse_args()



def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, print_timing: bool = True):
    """
    This function trains the model
    :return:
    """
    print('Start Training...')
    best_train_error = len(dloader_train)
    best_train_loss = 1e5
    previous_epoch_loss = 1e5
    best_model = None
    writer = SummaryWriter(args.output_dir)

    for e in range(epoch):
        train_error, train_loss = 0, 0
        model.train()
        print('Epoch {}:'.format(e + 1))
        for batch_idx, (data, target) in enumerate(tqdm(dloader_train)):
            start = time.time()
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)
            prob, label, weights = model(data)

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss', loss.data[0], batch_idx + e * len(dloader_train))
            # Calculate training error
            error = 1. - label.eq(target).cpu().float().mean().item()
            train_error += error

            if print_timing:
                end = time.time()
                print('Elapsed time of one train iteration is {:.0f} s'.format(end - start))


        writer.add_scalar('Training Loss', train_loss, e)
        writer.add_scalar('Training Error', train_error, e)
        print('Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Error: {:.2f}%'
              .format(e + 1, train_loss.cpu().numpy()[0], previous_epoch_loss - train_loss.cpu().numpy()[0], (train_error / len(dloader_train)) * 100))
        previous_epoch_loss = train_loss.cpu().numpy()[0]
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_error = train_error
            best_model = model

        # Save model to file:
        torch.save({'epoch': e + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                   os.path.join(args.output_dir, 'model_checkpoints.pt'))

        # torch.save(model.state_dict(), os.path.join(args.output_dir, 'model', 'Epoch_' + str(e + 1) + '.pt'))
        model.eval()
        test_acc = check_accuracy(model, dloader_test)
        writer.add_scalar('Test Set Accuracy', test_acc, e)
        model.train()

    return best_model, best_train_error, best_train_loss.cpu().numpy()[0]


def check_accuracy(model: nn.Module, data_loader: DataLoader):
    """
       if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (data, target) in data_loader:
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)
            _, label, _ = model(data)
            num_correct += (label == target).cpu().int().item()
            num_samples += label.size(0)
        acc = float(num_correct) / num_samples
        print('Got {} / {} correct = {:.2f} % over Test set'.format(num_correct, num_samples, 100 * acc))
    return acc




##################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Tile size definition:
    TILE_SIZE = 256

    # Get data:
    #train_dset = utils.PreSavedTiles_MILdataset(train=True)

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

    # This test data set is written such that it'll be the same as the train dataset
    test_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE,
                                     num_of_tiles_from_slide=50,
                                     print_timing=False,
                                     test_fold=args.test_fold,
                                     train=True,
                                     transform=None)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=cpu_available, pin_memory=False)

    # Model upload:
    infer = False
    if not infer:
        net = model.GatedAttention(tile_size=TILE_SIZE)
        if DEVICE.type == 'cuda':
            # net = nn.parallel.DistributedDataParallel(net)
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
                # TODO: check how to work with nn.parallel.DistributedDataParallel
                print('Using {} GPUs'.format(torch.cuda.device_count()))
            cudnn.benchmark = True

        #  optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer = optim.Adam(net.parameters(), lr=1e-5, weight_decay=5e-5)  # This is the optimizer from the paper Gil reviewed
        # optimizer = optim.Adadelta(net.parameters())

        epoch = args.epochs
        best_model, best_train_error, best_train_loss = train(net, train_loader, test_loader, print_timing=False)
