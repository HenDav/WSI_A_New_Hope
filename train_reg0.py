import utils
import PreActResNets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import nets
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys
import datasets

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=4, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Epochs to run')
parser.add_argument('-tt', '--transform_type', type=str, default='flip', help='keyword for transform type')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', type=str, default='ER', help='DataSet to use')


args = parser.parse_args()

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
        all_writer.add_text('Model type', model.model_name)
        all_writer.add_text('Data type', dloader_train.dataset.DataSet)
        all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
        all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    previous_epoch_loss = 1e5

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()
        total, correct_pos, correct_neg = 0, 0, 0
        total_pos_train, total_neg_train = 0, 0
        true_targets_train, scores_train = np.zeros(0), np.zeros(0)
        correct_labeling, train_loss = 0, 0

        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, (data, target, time_list, image_file, basic_tiles) in enumerate(tqdm(dloader_train)):
        #for batch_idx, (data, target, time_list) in enumerate(tqdm(dloader_train)):
            train_start = time.time()
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target.squeeze()
            model.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            #print(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
            true_targets_train = np.concatenate((true_targets_train, target.cpu().detach().numpy()))

            total += target.size(0)
            total_pos_train += target.eq(1).sum().item()
            total_neg_train += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()
            correct_pos += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg += predicted[target.eq(0)].eq(0).sum().item()

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

        balanced_acc_train = 100 * (correct_pos / (total_pos_train + 1e-7) + correct_neg / (total_neg_train + 1e-7)) / 2

        fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
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

        if e % 10 == 0:
            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e)
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

    all_writer.close()
    if print_timing:
        time_writer.close()



#def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, DEVICE, epoch: int, eval_mode: bool = False):
def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, DEVICE, epoch: int):

    total_test, correct_pos_test, correct_neg_test = 0, 0, 0
    total_pos_test, total_neg_test = 0, 0
    true_labels_test, scores_test = np.zeros(0), np.zeros(0)
    correct_labeling_test, loss_test = 0, 0

    model.eval()

    with torch.no_grad():
        #for idx, (data, targets, time_list) in enumerate(data_loader):
        for idx, (data, target, time_list, image_file, basic_tiles) in enumerate(data_loader):

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            out = model(data)

            outputs = torch.nn.functional.softmax(out, dim=1)
            target = target.squeeze()
            loss = criterion(outputs, target)

            loss_test += loss.item()
            _, predicted = outputs.max(1)
            scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))
            true_labels_test = np.concatenate((true_labels_test, target.cpu().detach().numpy()))

            total_test += target.size(0)
            correct_labeling_test += predicted.eq(target).sum().item()
            total_pos_test += target.eq(1).sum().item()
            total_neg_test += target.eq(0).sum().item()
            correct_pos_test += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg_test += predicted[target.eq(0)].eq(0).sum().item()

        acc = 100 * float(correct_labeling_test) / total_test
        balanced_acc = 100 * (correct_pos_test / (total_pos_test + 1e-7) + correct_neg_test / (total_neg_test + 1e-7)) / 2

        fpr, tpr, _ = roc_curve(true_labels_test, scores_test)
        roc_auc = auc(fpr, tpr)

        writer_string = 'Test (eval mode)'
        writer_all.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer_all.add_scalar(writer_string + '/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)

        print('{}: Accuracy of {:.2f}% ({} / {}) over Test set'.format('EVAL mode', acc, correct_labeling_test, total_test))
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
    # data_path = '/Users/wasserman/Developer/All data - outer scope'

    if sys.platform == 'linux':
        TILE_SIZE = 256
        data_path = '/home/womer/project/All Data'

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        args.output_dir, experiment = utils.run_data(test_fold=args.test_fold,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=1,
                                                     num_bags=10,
                                                     DX=args.dx,
                                                     DataSet=args.dataset,
                                                     Receptor=args.target,
                                                     MultiSlide=False)
    else:
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE,\
            TILES_PER_BAG, args.dx, args.dataset, args.target, _ = utils.run_data(experiment=args.experiment)
        experiment = args.experiment

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Get data:
    ''' 
    train_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      target_kind=args.target,
                                      test_fold=args.test_fold,
                                      train=True,
                                      print_timing=args.time,
                                      transform=args.transformation,
                                      DX=args.dx)

    test_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                     tile_size=TILE_SIZE,
                                     target_kind=args.target,
                                     test_fold=args.test_fold,
                                     train=False,
                                     print_timing=False,
                                     transform=False,
                                     DX=args.dx)
    '''
    train_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                         tile_size=TILE_SIZE,
                                         target_kind=args.target,
                                         test_fold=args.test_fold,
                                         train=True,
                                         print_timing=args.time,
                                         transform_type=args.transform_type,
                                         n_patches=10
                                         )
    test_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                        tile_size=TILE_SIZE,
                                        target_kind=args.target,
                                        test_fold=args.test_fold,
                                        train=False,
                                        print_timing=args.time,
                                        transform_type='none',
                                        n_patches=1
                                        )


    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader  = DataLoader(test_dset, batch_size=50, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Load model
    # model = nets.PreActResNet50()
    # model = nets.ResNet50_GN()
    # model = nets.ResNet_18()
    # model = nets.ResNet_34()
    # model = nets.ResNet50()
    # model = PreActResNets.preactresnet50_Omer()
    # model = PreActResNets.preactresnet50()
    model = PreActResNets.PreActResNet50_Ron()


    '''model_params = sum(p.numel() for p in model.parameters())
    print(model_params)'''

    utils.run_data(experiment=experiment, model=model.model_name)


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
    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
