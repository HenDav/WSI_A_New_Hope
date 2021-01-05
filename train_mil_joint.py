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

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=2, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=1, type=int, help='Epochs to run')
#parser.add_argument('-t', dest='transformation', action='store_true', help='Include transformations ?')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ms', dest='multi_slides', action='store_true', help='Use more than one slide in each bag')
parser.add_argument('-ds', '--dataset', type=str, default='HEROHE', help='DataSet to use')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')

parser.add_argument('--target', default='Her2', type=str, help='label: Her2/ER/PR/EGFR/PDL1/RedSquares') # RanS 7.12.20
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty') # RanS 7.12.20
parser.add_argument('--balanced_sampling', action='store_true', help='balanced_sampling') # RanS 7.12.20, TODO
parser.add_argument('--transform_type', default='flip', type=str, help='type of patch augmentation (string)') # RanS 7.12.20
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') # RanS 8.12.20
parser.add_argument('--model', default='resnet50_gn', type=str, help='resnet50_gn / receptornet') # RanS 15.12.20
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error') #RanS 16.12.20
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs') #RanS 16.12.20
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter') # RanS 28.12.20
parser.add_argument('--bag_size_test', default=50, type=int, help='# of samples in test bags (inference)') # RanS 29.12.20
args = parser.parse_args()
eps = 1e-7


def norm_img(img):
    img -= img.min()
    img /= img.max()
    return img


def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))
    image_writer = SummaryWriter(os.path.join(writer_folder, 'image'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'MIL')
        all_writer.add_text('Model type', str(type(model)))
        all_writer.add_text('Data type', dloader_train.dataset.DataSet)
        all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
        all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))


    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    best_train_loss = 1e5
    previous_epoch_loss = 1e5
    best_model = None

    '''    
    # The following part saves the random slides on file for further debugging
    if os.path.isfile('random_slides.xlsx'):
        random_slides = pd.read_excel('random_slides.xlsx')
    else:
        random_slides = pd.DataFrame()
    ###############################################
    '''

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()
        if e == 0:
            data_list = []
            # slide_random_list = []

        # The following 3 lines initialize variables to compute AUC for train dataset.
        total_pos_train, total_neg_train = 0, 0
        true_pos_train, true_neg_train = 0, 0
        targets_train, scores_train = np.zeros(len(dloader_train), dtype=np.int8), np.zeros(len(dloader_train))
        correct_labeling, train_loss = 0, 0

        print('Epoch {}:'.format(e))
        model.train()
        model.infer = False
        for batch_idx, (data, target, time_list, image_file, basic_tiles) in enumerate(tqdm(dloader_train)):
            train_start = time.time()


            if e == 0:
                data_dict = { 'File Name':  image_file,
                              'Target': target.cpu().detach().item()
                              }
                data_list.append(data_dict)

            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)

            optimizer.zero_grad()
            prob, label, _ = model(data)

            targets_train[batch_idx] = target.cpu().detach().item()
            total_pos_train += target.eq(1).item()
            total_neg_train += target.eq(0).item()

            if target == 1 and label == 1:
                true_pos_train += 1
            elif target == 0 and label == 0:
                true_neg_train += 1

            scores_train[batch_idx] = prob.cpu().detach().numpy()[0][0]

            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            all_writer.add_scalar('Loss', loss.data[0], batch_idx + e * len(dloader_train))

            # Calculate training accuracy
            correct_labeling += label.eq(target).cpu().detach().int().item()

            train_time = time.time() - train_start
            if print_timing:
                time_stamp = batch_idx + e * len(dloader_train)
                time_writer.add_scalar('Time/Train (iter) [Sec]', train_time, time_stamp)
                # print('Elapsed time of one train iteration is {:.2f} s'.format(train_time))
                if len(time_list) == 4:
                    time_writer.add_scalar('Time/Open WSI [Sec]'     , time_list[0], time_stamp)
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[1], time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]' , time_list[2], time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[3], time_stamp)
                else:
                    time_writer.add_scalar('Time/Avg to Extract Tile [Sec]', time_list[0], time_stamp)
                    time_writer.add_scalar('Time/Augmentation [Sec]', time_list[1], time_stamp)
                    time_writer.add_scalar('Time/Total To Collect Data [Sec]', time_list[2], time_stamp)

        time_epoch = (time.time() - time_epoch_start) / 60
        if print_timing:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch, e)

        train_acc = 100 * correct_labeling / len(dloader_train)

        #balanced_acc_train = 100 * (true_pos_train / total_pos_train + true_neg_train / total_neg_train) / 2
        balanced_acc_train = 100. * ((true_pos_train + eps) / (total_pos_train + eps) + (true_neg_train + eps) / (total_neg_train + eps)) / 2

        fpr_train, tpr_train, _ = roc_curve(targets_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)

        #print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {}), Time: {:.0f} m'
        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train AUC: {:.2f} , Time: {:.0f} m'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      #train_acc,
                      roc_auc_train,
                      #int(correct_labeling),
                      #len(train_loader),
                      time_epoch))

        previous_epoch_loss = train_loss

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = e
            best_model = model

        #if e % 5 == 0:
        if e % args.eval_rate == 0:
            if e % 20 == 0 and args.images:
                temp = False
                if temp:
                    from mpl_toolkits.axes_grid1 import ImageGrid
                    import matplotlib.pyplot as plt
                    fig1, fig2 = plt.figure(), plt.figure()
                    fig1.set_size_inches(32, 18)
                    fig2.set_size_inches(32, 18)

                    grid1 = ImageGrid(fig1, 111, nrows_ncols=(2, 5), axes_pad=0)
                    grid2 = ImageGrid(fig2, 111, nrows_ncols=(2, 5), axes_pad=0)

                    qq = data.squeeze().detach().cpu().numpy()
                    for ii in range(10):
                        img1 = np.squeeze(data[:, ii, :, :, :])
                        grid1[ii].imshow(np.transpose(img1, axes=(1, 2, 0)))

                        img2 = np.squeeze(qq[ii, :, :, :])
                        grid2[ii].imshow(np.transpose(img2, axes=(1, 2, 0)))

                    fig1.suptitle('Train Images/After Transforms', fontsize=14)
                    fig1.suptitle('data.squeeze().detach().cpu().numpy()', fontsize=14)

                    plt.show()
                    ######################################################

                image_writer.add_images('Train Images/Before Transforms', basic_tiles.squeeze().detach().cpu().numpy(),
                                       global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms', data.squeeze().detach().cpu().numpy(),
                                       global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms (De-Normalized)',
                                       norm_img(data.squeeze().detach().cpu().numpy()), global_step=e,
                                       dataformats='NCHW')

            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, image_writer, DEVICE, e, eval_mode=True)
            ### acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, image_writer, DEVICE, e, eval_mode=False)
            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data(experiment=experiment, epoch=e)


            # Save model to file:
            #RanS 31.12.20 - save every eval_rate epochs
            if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
                os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))

            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()

            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.data[0],
                        'acc_test': acc_test,
                        'bacc_test': bacc_test,
                        'tile_size': TILE_SIZE,
                        'tiles_per_bag': TILES_PER_BAG},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

        else:
            acc_test, bacc_test = None, None

        if e == 0:
            pd.DataFrame(data_list).to_excel('validate_data.xlsx')
            print('Saved validation data')

    all_writer.close()
    if print_timing:
        time_writer.close()
    '''
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
                'tiles_per_bag': TILES_PER_BAG},
               os.path.join(args.output_dir, 'Model_CheckPoints', 'best_model_Ep_' + str(best_epoch) + '.pt'))

    '''



def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, image_writer, DEVICE, epoch: int, eval_mode: bool = False):
    num_correct = 0
    total_pos, total_neg = 0, 0
    true_pos, true_neg = 0, 0
    targets_test, scores, preds = np.zeros(len(data_loader)), np.zeros(len(data_loader)), np.zeros(len(data_loader))

    if eval_mode:
        model.eval()
    else:
        model.train()
    with torch.no_grad():
        for idx, (data, target, time_list, _, basic_tiles) in enumerate(data_loader):
            if epoch % 20 == 0 and args.images:
                image_writer.add_images('Test Images/Before Transforms', basic_tiles.squeeze().detach().cpu().numpy(),
                                     global_step=epoch, dataformats='NCHW')
                image_writer.add_images('Test Images/After Transforms', data.squeeze().detach().cpu().numpy(),
                                     global_step=epoch, dataformats='NCHW')
                image_writer.add_images('Test Images/After Transforms (De-Normalized)',
                                     norm_img(data.squeeze().detach().cpu().numpy()), global_step=epoch, dataformats='NCHW')

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            prob, label, _ = model(data)

            targets_test[idx] = target.cpu().detach().item()
            total_pos += target.eq(1).item()
            total_neg += target.eq(0).item()

            if target == 1 and label == 1:
                true_pos += 1
            elif target == 0 and label == 0:
                true_neg += 1

            num_correct += label.eq(target).cpu().detach().int().item()
            scores[idx] = prob.cpu().detach().item()
            preds[idx] = label.cpu().detach().item()

        if data_loader.dataset.train:
            writer_string = 'Train_2'
        else:
            if eval_mode:
                writer_string = 'Test (eval mode)'
            else:
                writer_string = 'Test (train mode)'

        ###########################################################################

        if not args.bootstrap:
            acc = 100 * float(num_correct) / len(data_loader)
            bacc = 100. * ((true_pos + eps) / (total_pos + eps) + (true_neg + eps) / (total_neg + eps)) / 2
            roc_auc = np.nan
            if not all(targets_test==targets_test[0]): #more than one label
                fpr, tpr, _ = roc_curve(targets_test, scores)
                roc_auc = auc(fpr, tpr)
        else: #bootstrap, RanS 16.12.20
            n_iterations = 100
            # run bootstrap
            roc_auc_array, acc_array, bacc_array = np.empty(n_iterations), np.empty(n_iterations), np.empty(n_iterations)
            roc_auc_array[:], acc_array[:], bacc_array[:]  = np.nan, np.nan, np.nan

            for ii in range(n_iterations):
                #resample bags, each bag is a sample
                scores_resampled, preds_resampled, targets_resampled = resample(scores, preds, targets_test)
                fpr, tpr, _ = roc_curve(targets_resampled, scores_resampled)

                num_correct_i = np.sum(preds_resampled==targets_resampled)
                true_pos_i = np.sum(targets_resampled+preds_resampled==2)
                total_pos_i = np.sum(targets_resampled==0)
                true_neg_i = np.sum(targets_resampled+preds_resampled==0)
                total_neg_i = np.sum(targets_resampled==1)
                acc_array[ii] = 100 * float(num_correct_i) / len(data_loader)
                bacc_array[ii] = 100. * ((true_pos_i + eps) / (total_pos_i + eps) + (true_neg_i + eps) / (total_neg_i + eps)) / 2
                if not all(targets_resampled == targets_resampled[0]):  # more than one label
                    roc_auc_array[ii] = roc_auc_score(targets_resampled, scores_resampled)

            roc_auc = np.nanmean(roc_auc_array)
            roc_auc_err = np.nanstd(roc_auc_array)
            acc = np.nanmean(acc_array)
            acc_err = np.nanstd(acc_array)
            bacc = np.nanmean(bacc_array)
            bacc_err = np.nanstd(bacc_array)
            writer_all.add_scalar(writer_string + '/Roc-Auc error', roc_auc_err, epoch)
            writer_all.add_scalar(writer_string + '/Accuracy error', acc_err, epoch)
            writer_all.add_scalar(writer_string + '/Balanced Accuracy error', bacc_err, epoch)

        writer_all.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer_all.add_scalar(writer_string + '/Balanced Accuracy', bacc, epoch)
        writer_all.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)

        #print('{}: Accuracy of {:.2f}% ({} / {}) over Test set'.format('EVAL mode' if eval_mode else 'TRAIN mode', acc, num_correct, len(data_loader)))
        print('{}: AUC of {:.2f} over Test set'.format('EVAL mode' if eval_mode else 'TRAIN mode', roc_auc))

    model.train()
    '''
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
    '''
    return acc, bacc


if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()
    # Data type definition:
    DATA_TYPE = 'WSI'

    # Tile size definition:
    TILE_SIZE =128
    TILES_PER_BAG = 10
    # data_path = '/Users/wasserman/Developer/All data - outer scope'

    if sys.platform == 'linux':
        TILE_SIZE = 256
        TILES_PER_BAG = 50
    if sys.platform == 'win32':
        TILE_SIZE = 256

    # Saving/Loading run meta data to/from file:
    if args.experiment == 0:
        args.output_dir, experiment = utils.run_data(test_fold=args.test_fold,
                                                     #transformations=args.transformation,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=TILES_PER_BAG,
                                                     DX=args.dx,
                                                     DataSet=args.dataset,
                                                     Receptor=args.target,
                                                     MultiSlide=args.multi_slides)
    else:
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE, TILES_PER_BAG, args.dx, \
        args.dataset, args.target, args.multi_slides = utils.run_data(experiment=args.experiment)
        experiment = args.experiment
        # args.output_dir, args.test_fold, args.transformation, TILE_SIZE, TILES_PER_BAG, args.dx,\

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Get data:
    if not args.multi_slides:
        train_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                          tile_size=TILE_SIZE,
                                          bag_size=TILES_PER_BAG,
                                          target_kind=args.target,
                                          test_fold=args.test_fold,
                                          train=True,
                                          print_timing=args.time,
                                          #transform=args.transformation,
                                          transform_type=args.transform_type,
                                          DX=args.dx,
                                          get_images=args.images,
                                          c_param=args.c_param)

        test_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                         tile_size=TILE_SIZE,
                                         #bag_size=TILES_PER_BAG,
                                         bag_size=args.bag_size_test, #RanS 29.12.20
                                         target_kind=args.target,
                                         test_fold=args.test_fold,
                                         train=False,
                                         print_timing=False,
                                         #transform=False,
                                         transform_type='none',
                                         DX=args.dx,
                                         get_images=args.images)
    else:
        train_dset = datasets.WSI_MIL3_dataset(DataSet=args.dataset,
                                            tile_size=TILE_SIZE,
                                            bag_size=TILES_PER_BAG,
                                            target_kind=args.target,
                                            TPS=10,
                                            test_fold=args.test_fold,
                                            train=True,
                                            print_timing=args.time,
                                            #transform=args.transformation,
                                            transform_type=args.transform_type,
                                            DX=args.dx,
                                            c_param=args.c_param)

        test_dset = datasets.WSI_MIL3_dataset(DataSet=args.dataset,
                                           tile_size=TILE_SIZE,
                                           #bag_size=TILES_PER_BAG,
                                           bag_size=args.bag_size_test, #RanS 29.12.20
                                           target_kind=args.target,
                                           TPS=10,
                                           test_fold=args.test_fold,
                                           train=False,
                                           print_timing=False,
                                           #transform=False,
                                           transform_type='none',
                                           DX=args.dx)

    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader  = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Load model
    model = utils.get_model(args.model)

    utils.run_data(experiment=experiment, model=model.model_name)

    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment != 0:
        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

        model.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if DEVICE.type == 'cuda':
        cudnn.benchmark = True

    if args.experiment != 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
