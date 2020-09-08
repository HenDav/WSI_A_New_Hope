import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import model
from tqdm import tqdm
import time

def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, print_every: int = 100):
    """
    This function trains the model
    :return:
    """
    print('Start Training...')
    best_train_error = len(dloader_train)
    best_train_loss = 1e5
    best_model = None

    for e in range(epoch):
        train_error, train_loss = 0, 0
        model.train()
        print('Epoch {}:'.format(e))
        for batch_idx, (data, target) in enumerate(tqdm(dloader_train)):
            start = time.time()
            data, target = data.to(DEVICE), target.to(DEVICE)
            prob, label, weights = model(data)

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training error
            error = 1. - label.eq(target).cpu().float().mean().item()
            train_error += error

            end = time.time()
            print('Elapsed time of one train iteration is {:.0f} seconds'.format(end - start))


        print('Epoch {}: Train Loss = {:.2f}, Train Error = {:.2f}%'.format(e, train_loss, train_error / len(dloader_train)))

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_error = train_error
            best_model = model



        model.eval()
        test_acc = check_accuracy(model, dloader_test)
        model.train()
        print('Epoch: {}, Loss: {:.2f}, Train error: {:.0f}'.format(e, train_loss.cpu().numpy()[0], train_error))

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
        for data, target, _ in data_loader:
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            _, label, _ = model(data)
            num_correct += (label == target).cpu().int().item()
            num_samples += label.size(0)
        acc = float(num_correct) / num_samples
        print('Got {} / {} correct {:.2f} %'.format(num_correct, num_samples, 100 * acc))
    return acc




##################################################################################################


# Device definition:
DEVICE = utils.device_gpu_cpu()

if DEVICE == 'cuda':
    net = nn.parallel.DistributedDataParallel(net)
    # net = nn.DataParallel(net)
    cudnn.benchmark = True


# Get data:
train_dset = utils.WSI_MILdataset(train=True)
train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=20, pin_memory=False)

test_dset = utils.WSI_MILdataset(train=False)
test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=20, pin_memory=False)

# Model upload:
infer = False
if not infer:
    net = model.GatedAttention(tile_size=256)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    # optimizer = optim.Adadelta(net.parameters())

    epoch = 5
    best_model, best_train_error, best_train_loss = train(net, train_loader, test_loader)
