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
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-tt', '--transform_type', type=str, default='rvf', help='keyword for transform type')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-diffslides', dest='different_slides', action='store_true', help='Use more than one slide in each bag')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-tar', '--target', type=str, default='ER', help='DataSet to use')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-nb', '--num_bags', type=int, default=2, help='Number of bags in each minibatch')
parser.add_argument('-tpb', '--tiles_per_bag', type=int, default=8, help='Tiles Per Bag')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') # RanS 8.12.20
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty') # RanS 7.12.20
parser.add_argument('--model', default='nets_mil.MIL_PreActResNet50_Ron_MultiBag()', type=str, help='net to use')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-slide_reps', '--slide_repetitions', type=int, default=1, help='Slide repetitions per epoch')

#parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time') # RanS 7.12.20
#parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time') # RanS 7.12.20

#parser.add_argument('--batch_size', default=18, type=int, help='size of batch')  # RanS 8.12.20
#parser.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')  # RanS 7.12.20
#parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error') #RanS 16.12.20

args = parser.parse_args()

EPS = 1e-7


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
    if args.images:
        image_writer = SummaryWriter(os.path.join(writer_folder, 'image'))
    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'MIL')
        all_writer.add_text('Model type', str(type(model)))
        all_writer.add_text('Data type', dloader_train.dataset.DataSet)
        all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
        all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

    print()
    print('Training will be conducted with {} bags and {} tiles per bag in each MiniBatch'.format(args.num_bags, args.tiles_per_bag))
    print('Start Training...')
    previous_epoch_loss = 1e5

    '''    
    # The following part saves the random slides on file for further debugging
    if os.path.isfile('random_slides.xlsx'):
        random_slides = pd.read_excel('random_slides.xlsx')
    else:
        random_slides = pd.DataFrame()
    '''

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()
        if e == 0:
            data_list = []
            # slide_random_list = []

        # The following 3 lines initialize variables to compute AUC for train dataset.
        total_train, correct_pos_train, correct_neg_train = 0, 0, 0
        total_pos_train, total_neg_train = 0, 0
        true_targets_train, scores_train = np.zeros(0), np.zeros(0)
        correct_labeling, train_loss = 0, 0

        print('Epoch {}:'.format(e))
        model.train()
        for batch_idx, (data, target, time_list, image_file, basic_tiles) in enumerate(tqdm(dloader_train)):
            train_start = time.time()
            '''
            if args.images:
                step = batch_idx + e * 1000
                image_writer.add_images('Train Images/Before Transforms', basic_tiles.squeeze().detach().cpu().numpy(),
                                    global_step=step, dataformats='NCHW')
            '''

            '''
            # The following section is responsible for saving the random slides for it's iteration - For debbugging purposes 
            slide_dict = {'Epoch': e,
                          'Main Slide index': idxx.cpu().detach().numpy()[0],
                          'random Slides index': slides_idx_other}
            slide_random_list.append(slide_dict)
            '''
            '''
            if e == 0:
                data_dict = { 'File Name':  image_file,
                              'Target': target.cpu().detach().numpy()
                              }
                data_list.append(data_dict)
            '''
            '''this_num_bags, _, _, _, _ = Data.shape
            data = torch.reshape(Data, (this_num_bags * TILES_PER_BAG, 3, TILE_SIZE, TILE_SIZE))'''

            data, target = data.to(DEVICE), target.to(DEVICE).squeeze(1)
            optimizer.zero_grad()
            model.to(DEVICE)

            outputs, weights = model(data)
            weights = weights.cpu().detach().numpy()

            #target_diag = torch.diag(target)
            '''neg_log_likelihood = -1. * (target * torch.log(scores) + (1. - target) * torch.log(1. - scores))  # negative log bernoulli'''
            '''neg_log_likelihood = -1. * (
                    torch.log(scores) * target_diag + torch.log(1. - scores) * torch.diag(1. - target))'''

            loss = criterion(outputs, target)
            train_loss += loss.item()

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            loss.backward()
            optimizer.step()

            #scores_train = np.concatenate((scores_train, scores.cpu().detach().numpy().reshape(-1)))

            scores_train = np.concatenate((scores_train, outputs[:, 1].cpu().detach().numpy()))
            true_targets_train = np.concatenate((true_targets_train, target.cpu().detach().numpy()))

            total_train += target.size(0)
            total_pos_train += target.eq(1).sum().item()
            total_neg_train += target.eq(0).sum().item()
            correct_labeling += predicted.eq(target).sum().item()

            correct_pos_train += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg_train += predicted[target.eq(0)].eq(0).sum().item()

            '''
            targets_train[batch_idx * num_bags : (batch_idx + 1) * this_num_bags] = target.cpu().detach().numpy().reshape(2)
            total_pos_train += target.eq(1).numpy().sum()
            total_neg_train += target.eq(0).numpy().sum()

            true_labels = target.eq(label)
            for label_idx, correctness in enumerate(true_labels):
                if correctness == True:
                    if label[label_idx] == 1:
                        true_pos_train += 1
                    elif label[label_idx] == 0:
                        true_neg_train += 1

            scores_train[batch_idx * num_bags : (batch_idx + 1) * num_bags] = prob.cpu().detach().numpy().reshape(2)
            '''
            '''
            prob = torch.clamp(prob, min=1e-5, max=1. - 1e-5)
            neg_log_likelihood = -1. * (target * torch.log(prob) + (1. - target) * torch.log(1. - prob))  # negative log bernoulli
            loss = neg_log_likelihood
            train_loss += loss.item()
            '''

            # Calculate training accuracy
            all_writer.add_scalar('Loss', loss.item(), batch_idx + e * len(dloader_train))

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

        '''
        random_slides = random_slides.append(pd.DataFrame(slide_random_list))
        random_slides.to_excel('random_slides.xlsx')
        '''
        time_epoch = (time.time() - time_epoch_start) / 60
        if print_timing:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch, e)

        train_acc = 100 * correct_labeling / total_train
        balanced_acc_train = 100 * (correct_pos_train / (total_pos_train + EPS) + correct_neg_train / (total_neg_train + EPS)) / 2

        fpr_train, tpr_train, _ = roc_curve(true_targets_train, scores_train)
        roc_auc_train = auc(fpr_train, tpr_train)

        all_writer.add_scalar('Train/Balanced Accuracy', balanced_acc_train, e)
        all_writer.add_scalar('Train/Roc-Auc', roc_auc_train, e)
        all_writer.add_scalar('Train/Loss Per Epoch', train_loss, e)
        all_writer.add_scalar('Train/Accuracy', train_acc, e)
        all_writer.add_scalar('Train/Weights mean Total (per bag)', np.mean(np.sum(weights, axis=1)), e)
        all_writer.add_scalar('Train/Weights mean Variance (per bag)', np.mean(np.var(weights, axis=1)), e)

        '''print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {}), Time: {:.0f} m'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      train_acc,
                      int(correct_labeling),
                      len(train_loader.dataset),
                      time_epoch))'''

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train AUC per patch: {:.2f} , Time: {:.0f} m'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      roc_auc_train,
                      time_epoch))

        previous_epoch_loss = train_loss

        '''
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = e
            best_model = model
        '''

        if e % args.eval_rate == 0:
            acc_test, bacc_test = check_accuracy(model, dloader_test, all_writer, DEVICE, e)
            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data(experiment=experiment, epoch=e)

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
                        'tiles_per_bag': args.tiles_per_bag},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

            if e % 20 == 0 and args.images:
                image_writer.add_images('Train Images/Before Transforms', basic_tiles.squeeze().detach().cpu().numpy(),
                                        global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms', data.squeeze().detach().cpu().numpy(),
                                        global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms (De-Normalized)',
                                        norm_img(data.squeeze().detach().cpu().numpy()), global_step=e,
                                        dataformats='NCHW')
        else:
            acc_test, bacc_test = None, None

        '''if e == 0:
            pd.DataFrame(data_list).to_excel('validate_data.xlsx')
            print('Saved validation data')'''

    all_writer.close()
    if print_timing:
        time_writer.close()
    if args.images:
        image_writer.close()

def check_accuracy(model: nn.Module, data_loader: DataLoader, writer_all, DEVICE, epoch: int):
    test_loss, total_test = 0, 0
    correct_labeling_test = 0
    total_pos_test, total_neg_test = 0, 0
    correct_pos_test, correct_neg_test = 0, 0
    targets_test, scores_test = np.zeros(0), np.zeros(0)

    model.eval()

    with torch.no_grad():
        for idx, (data, target, time_list, image_file_name, basic_tiles) in enumerate(data_loader):
            data, target = data.to(device=DEVICE), target.to(device=DEVICE).squeeze(1)
            model.to(DEVICE)

            outputs, weights = model(data)

            outputs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            scores_test = np.concatenate((scores_test, outputs[:, 1].cpu().detach().numpy()))
            targets_test = np.concatenate((targets_test, target.cpu().detach().numpy()))

            total_test += target.size(0)
            correct_labeling_test += predicted.eq(target).sum().item()
            total_pos_test += target.eq(1).sum().item()
            total_neg_test += target.eq(0).sum().item()
            correct_pos_test += predicted[target.eq(1)].eq(1).sum().item()
            correct_neg_test += predicted[target.eq(0)].eq(0).sum().item()

        acc = 100 * float(correct_labeling_test) / total_test
        balanced_acc = 100 * (correct_pos_test / (total_pos_test + EPS) + correct_neg_test / (total_neg_test + EPS)) / 2

        fpr, tpr, _ = roc_curve(targets_test, scores_test)
        roc_auc = auc(fpr, tpr)

        writer_all.add_scalar('Test/Accuracy', acc, epoch)
        writer_all.add_scalar('Test/Balanced Accuracy', balanced_acc, epoch)
        writer_all.add_scalar('Test/Roc-Auc', roc_auc, epoch)

        '''print('Accuracy of {:.2f}% ({} / {}) over Test set'.format(acc, correct_labeling_test, len(data_loader.dataset)))'''
        print('Tile AUC of {:.2f} over Test set'.format(roc_auc))

    model.train()
    '''
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
    '''
    return acc, balanced_acc



##################################################################################################


if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Data type definition:
    DATA_TYPE = 'WSI'

    # Tile size definition:
    TILE_SIZE = 128
    #TILES_PER_BAG = args.tiles_per_bag
    #num_bags = args.num_bags

    if sys.platform == 'linux' or sys.platform == 'win32':
        TILE_SIZE = 256
        #TILES_PER_BAG = args.tiles_per_bag
        # data_path = '/home/womer/project/All Data'

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=args.tiles_per_bag,
                                                     num_bags=args.num_bags,
                                                     DX=args.dx,
                                                     DataSet_name=args.dataset,
                                                     Receptor=args.target,
                                                     MultiSlide=True,
                                                     DataSet_Slide_magnification=args.mag)

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']
    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transformation, TILE_SIZE, args.tiles_per_bag, args.num_bags, args.dx, \
        args.dataset, args.target, is_MultiSlide, args.model, args.mag =\
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], run_data_output['Tile Size'],\
            run_data_output['Tiles Per Bag'], run_data_output['Num Bags'], run_data_output['DX'], run_data_output['Dataset Name'],\
            run_data_output['Receptor'], run_data_output['MultiSlide'], run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        experiment = args.experiment

    # Get data:
    train_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                         tile_size=TILE_SIZE,
                                         bag_size=args.tiles_per_bag,
                                         target_kind=args.target,
                                         test_fold=args.test_fold,
                                         train=True,
                                         print_timing=args.time,
                                         transform_type=args.transform_type,
                                         DX=args.dx,
                                         get_images=args.images,
                                         desired_slide_magnification=args.mag,
                                         slide_repetitions=args.slide_repetitions)

    test_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                        tile_size=TILE_SIZE,
                                        bag_size=args.tiles_per_bag,
                                        target_kind=args.target,
                                        test_fold=args.test_fold,
                                        train=False,
                                        print_timing=False,
                                        transform_type='none',
                                        DX=args.dx,
                                        get_images=args.images,
                                        desired_slide_magnification=args.mag)


    train_loader = DataLoader(train_dset, batch_size=args.num_bags, shuffle=True, num_workers=cpu_available, pin_memory=True)
    if args.tiles_per_bag == 1:
        # In case there is only 1 tile per bag than were working in the REG paradigm and NOT MIL.
        # In that case we'll test each slide by itself.
        test_loader = DataLoader(test_dset, batch_size=args.num_bags, shuffle=False, num_workers=cpu_available, pin_memory=True)
    else:
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Saving code files, args and main file name (this file) to Code directory within the run files.
    utils.save_code_files(args, train_dset)

    # Load model
    model = eval(args.model)
    if model.model_name == 'nets_mil.MIL_PreActResNet50_Ron_MultiBag()':
        model.tiles_per_bag = args.tiles_per_bag

    # Save model data and data-set size to run_data.xlsx file (Only if this is a new run).
    if args.experiment == 0:
        utils.run_data(experiment=experiment, model=model.model_name)
        utils.run_data(experiment=experiment, DataSet_size=(train_dset.real_length, test_dset.real_length))

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

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

    if args.experiment is not 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
