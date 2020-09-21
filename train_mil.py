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
    writer = SummaryWriter('runs-output')

    for e in range(epoch):
        train_error, train_loss = 0, 0
        model.train()
        print('Epoch {}:'.format(e))
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
              .format(e + 1, train_loss.cpu().numpy()[0], previous_epoch_loss - train_loss.cpu().numpy()[0],(train_error / len(dloader_train)) * 100))
        previous_epoch_loss = train_loss.cpu().numpy()[0]
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_error = train_error
            best_model = model



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
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for (data, target) in data_loader:
            if target == 1:
                total_pos += 1
            else:
                total_neg += 1

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)
            _, label, _ = model(data)
            num_correct += (label == target).cpu().int().item()
            num_samples += label.size(0)

            if target == 1 and label == 1:
                true_pos += 1
            elif target == 0 and label == 0:
                true_neg += 1

        acc = float(num_correct) / num_samples
        balanced_acc = 100 * (true_pos / total_pos + true_neg / total_neg) / 2
        wri
        print('Got {} / {} correct = {:.2f} % over Test set'.format(num_correct, num_samples, 100 * acc))
    return acc




##################################################################################################


# Device definition:
DEVICE = utils.device_gpu_cpu()

# Get number of available CPUs:
cpu_available = utils.get_cpu()

# Tile size definition:
TILE_SIZE = 256

# Get data:
#train_dset = utils.PreSavedTiles_MILdataset(train=True)

train_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE, num_of_tiles_from_slide=50, train=True, print_timing=False, transform=utils.get_transform())
train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=cpu_available, pin_memory=False)

# TODO: check it it's possible to put all the tiles of a WSI in the test loader
test_dset = utils.WSI_MILdataset(tile_size=TILE_SIZE, num_of_tiles_from_slide=200, print_timing=False, train=False)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.Adadelta(net.parameters())

    epoch = 10
    best_model, best_train_error, best_train_loss = train(net, train_loader, test_loader, print_timing=False)
