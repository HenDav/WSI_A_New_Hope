import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from nets import PreActResNet50, ResNet50_2, ResNext_50, ResNet50_GN, resnet50_with_3FC
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import sys
import pandas as pd

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=2, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Epochs to run')
#parser.add_argument('-t', dest='transformation', action='store_true', help='Include transformations ?')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='LUNG', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('--target', default='Her2', type=str, help='label: Her2/ER/PR/EGFR/PDL1') # RanS 7.12.20
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time') # RanS 7.12.20
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time') # RanS 7.12.20
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty') # RanS 7.12.20
parser.add_argument('--balanced_sampling', action='store_true', help='balanced_sampling') # RanS 7.12.20
parser.add_argument('--transform_type', default='flip', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)') # RanS 7.12.20
parser.add_argument('--batch_size', default=10, type=int, help='size of batch') # RanS 8.12.20
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') # RanS 8.12.20
parser.add_argument('--model', default='resnet50_gn', type=str, help='preact_resnet50 / resnet50 / resnet50_3FC / resnet50_gn') # RanS 15.12.20
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error') #RanS 16.12.20
args = parser.parse_args()
eps = 1e-7

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
        all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

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
        slide_names = []
        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, (data, target, time_list, f_names) in enumerate(tqdm(dloader_train)):
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

            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
            true_labels_train = np.concatenate((true_labels_train, target.cpu().detach().numpy()))
            slide_names.extend(slide_names_batch)

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

        #balanced_acc_train = 100 * (correct_pos / total_pos_train + correct_neg / total_neg_train) / 2
        balanced_acc_train = 100. * ((correct_pos + eps) / (total_pos_train + eps) + (correct_neg + eps) / (total_neg_train + eps)) / 2

        roc_auc_train = np.nan
        if not all(true_labels_train==true_labels_train[0]): #more than one label
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

        if (e % 20 == 0) or args.model == 'resnet50_3FC': #RanS 15.12.20, pretrained networks converge fast
            # RanS 8.12.20, perform slide inference
            patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores_train, 'labels': true_labels_train})
            slide_mean_score_df = patch_df.groupby('slide').mean()
            roc_auc_slide = np.nan
            if not all(slide_mean_score_df['labels']==slide_mean_score_df['labels'][0]): #more than one label
                roc_auc_slide = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])
            all_writer.add_scalar('Train/slide AUC', roc_auc_slide, e)

            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e, eval_mode=True)
            # acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e, eval_mode=False)
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
    slide_names = []

    test_loss, total = 0, 0

    # TODO: eval mode changes the mode of dropout and batchnorm layers. Activate this line only after the model is fully learned
    if eval_mode:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for idx, (data, targets, time_list, f_names) in enumerate(data_loader):
            data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)

            model.to(DEVICE)

            out = model(data)

            outputs = torch.nn.functional.softmax(out, dim=1)
            targets = targets.squeeze()
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            scores = np.concatenate((scores, outputs[:, 1].cpu().detach().numpy()))
            true_labels = np.concatenate((true_labels, targets.cpu().detach().numpy()))
            slide_names.extend(slide_names_batch)

            total += targets.size(0)
            num_correct += predicted.eq(targets).sum().item()
            total_pos += targets.eq(1).sum().item()
            total_neg += targets.eq(0).sum().item()
            true_pos += predicted[targets.eq(1)].eq(1).sum().item()
            true_neg += predicted[targets.eq(0)].eq(0).sum().item()

        acc = 100 * float(num_correct) / total
        #balanced_acc = 100 * (true_pos / total_pos + true_neg / total_neg) / 2
        balanced_acc = 100. * ((true_pos + eps) / (total_pos + eps) + (true_neg + eps) / (total_neg + eps)) / 2

        # TODO: instead of using the train parameter it is possible to simply check data_loader.dataset.train attribute
        if data_loader.dataset.train:
            writer_string = 'Train_2'
        else:
            if eval_mode:
                writer_string = 'Test (eval mode)'
            else:
                writer_string = 'Test (train mode)'

        if not args.bootstrap:
            roc_auc = np.nan
            if not all(true_labels==true_labels[0]): #more than one label
                fpr, tpr, _ = roc_curve(true_labels, scores)
                roc_auc = auc(fpr, tpr)

            #RanS 8.12.20, perform slide inference
            patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores, 'labels': true_labels})
            slide_mean_score_df = patch_df.groupby('slide').mean()
            roc_auc_slide = np.nan
            if not all(slide_mean_score_df['labels']==slide_mean_score_df['labels'][0]): #more than one label
                roc_auc_slide = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])
        else: #bootstrap, RanS 16.12.20

            from sklearn.utils import resample
            # load dataset
            # configure bootstrap
            n_iterations = 100

            # run bootstrap
            roc_auc_array = np.empty(n_iterations)
            slide_roc_auc_array = np.empty(n_iterations)
            roc_auc_array[:], slide_roc_auc_array[:] = np.nan, np.nan
            for ii in range(n_iterations):
                #slide_resampled, scores_resampled, labels_resampled = resample(slide_names, scores, true_labels)
                #fpr, tpr, _ = roc_curve(labels_resampled, scores_resampled)
                #patch_df = pd.DataFrame({'slide': slide_resampled, 'scores': scores_resampled, 'labels': labels_resampled})

                #patch_df = pd.DataFrame({'slide': slide_names, 'scores': scores, 'labels': true_labels})
                slide_names = np.array(slide_names)
                slide_choice = resample(np.unique(np.array(slide_names)))
                slide_resampled = np.concatenate([slide_names[slide_names==slide] for slide in slide_choice])
                scores_resampled = np.concatenate([scores[slide_names == slide] for slide in slide_choice])
                labels_resampled = np.concatenate([true_labels[slide_names == slide] for slide in slide_choice])
                patch_df = pd.DataFrame({'slide': slide_resampled, 'scores': scores_resampled, 'labels': labels_resampled})

                fpr, tpr, _ = roc_curve(labels_resampled, scores_resampled)
                if not all(labels_resampled == labels_resampled[0]): #more than one label
                    roc_auc_array[ii] = roc_auc_score(labels_resampled, scores_resampled)

                slide_mean_score_df = patch_df.groupby('slide').mean()
                if not all(slide_mean_score_df['labels'] == slide_mean_score_df['labels'][0]):  # more than one label
                    slide_roc_auc_array[ii] = roc_auc_score(slide_mean_score_df['labels'], slide_mean_score_df['scores'])
            roc_auc = np.nanmean(roc_auc_array)
            roc_auc_slide = np.nanmean(slide_roc_auc_array)
            roc_auc_std = np.nanstd(roc_auc_array)
            roc_auc_slide_std = np.nanstd(slide_roc_auc_array)
            writer_all.add_scalar(writer_string + '/Roc-Auc error', roc_auc_std, epoch)
            writer_all.add_scalar(writer_string + '/slide AUC error', roc_auc_slide_std, epoch)

        writer_all.add_scalar(writer_string + '/Accuracy', acc, epoch)
        writer_all.add_scalar(writer_string + '/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar(writer_string + '/Roc-Auc', roc_auc, epoch)
        writer_all.add_scalar(writer_string + '/slide AUC', roc_auc_slide, epoch)

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
    timing = False

    if sys.platform == 'linux' or sys.platform == 'win32':
        TILE_SIZE = 256

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        args.output_dir, experiment = utils.run_data(test_fold=args.test_fold,
                                                     #transformations=args.transformation,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=1,
                                                     DX=args.dx,
                                                     DataSet=args.dataset)
    else:
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE,\
            TILES_PER_BAG, args.dx, args.dataset, args.target = utils.run_data(experiment=args.experiment)
        experiment = args.experiment

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()
    #cpu_available = 1

    # Get data:
    train_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      target_kind=args.target,
                                      test_fold=args.test_fold,
                                      train=True,
                                      print_timing=args.time,
                                      #transform=args.transformation,
                                      DX=args.dx,
                                      transform_type=args.transform_type,
                                      n_patches=args.n_patches_train)

    test_dset = utils.WSI_REGdataset(DataSet=args.dataset,
                                     tile_size=TILE_SIZE,
                                     target_kind=args.target,
                                     test_fold=args.test_fold,
                                     train=False,
                                     print_timing=False,
                                     #transform=False,
                                     transform_type='none',
                                     DX=args.dx,
                                     n_patches=args.n_patches_test)


    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader  = DataLoader(test_dset, batch_size=args.batch_size*2, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)


    # Load model
    # RanS 14.12.20
    if args.model == 'resnet50_3FC':
        model = resnet50_with_3FC()
    elif args.model == 'preact_resnet50':
        #model = ResNext_50()
        model = PreActResNet50()
        # model = ResNet50_2()
    elif args.model == 'resnet50_gn':
        model = ResNet50_GN()
    else:
        print('model not defined!')

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)

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
    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
