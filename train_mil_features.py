import utils
import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import nets_mil
from PreActResNets import PreActResNet50_Ron
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from sklearn.metrics import roc_curve, auc
import numpy as np
import sys
import pandas as pd
import copy

parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
#parser.add_argument('-tt', '--transform_type', type=str, default='none', help='keyword for transform type')
#parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
#parser.add_argument('-diffslides', dest='different_slides', action='store_true', help='Use more than one slide in each bag')
#parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
#parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-ds', '--dataset', type=str, default='CARMEL_40', help='DataSet to use')
parser.add_argument('-tar', '--target', type=str, default='ER', help='Target to train for')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-ppt', dest='per_patient_training', action='store_true', help='will the data be taken per patient (or per slides) ?')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-nb', '--num_bags', type=int, default=50, help='Number of bags in each minibatch')
parser.add_argument('-tpb', '--tiles_per_bag', type=int, default=100, help='Tiles Per Bag')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate') # RanS 8.12.20
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty') # RanS 7.12.20
parser.add_argument('--model', default='nets_mil.MIL_Feature_Attention_MultiBag()', type=str, help='net to use')
#parser.add_argument('--model', default='nets_mil.MIL_Feature_3_Attention_MultiBag()', type=str, help='net to use')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
#parser.add_argument('-slide_reps', '--slide_repetitions', type=int, default=1, help='Slide repetitions per epoch')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('-llf', dest='last_layer_freeze', action='store_true', help='get last layer and freeze it ?')
parser.add_argument('-dl', '--data_limit', type=int, default=None, help='Data Limit to a specified number of feature tiles')
parser.add_argument('-repData', dest='repeating_data', action='store_false', help='sample data with repeat ?')
parser.add_argument('-conly', dest='carmel_only', action='store_true', help='Use ONLY CARMEL slides  ?')
parser.add_argument('-remark', '--remark', type=str, default='', nargs=argparse.REMAINDER, help='option to add remark for the run')

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
        #all_writer.add_text('Data type', dloader_train.dataset.DataSet)
        #all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
        #all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
        #all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
        #all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

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
        for batch_idx, minibatch in enumerate(tqdm(dloader_train)):
            labels = minibatch['labels']
            target = minibatch['targets']
            data = minibatch['features']

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

            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            model.to(DEVICE)

            outputs, weights, _ = model(x=None, H=data)

            if str(outputs.min().item()) == 'nan':
                print('slides:', minibatch['slide name'])
                print('features', data.shape)
                print('feature 1:', data[0].min().item(), data[0].max().item())
                print('feature 2:', data[1].min().item(), data[1].max().item())
                print('num tiles:', minibatch['num tiles'])
                print('feature1', data[0])
                print('feature2', data[1])

                exit()

            weights = weights.cpu().detach().numpy()



            DividedSlides_Flag = True if len(data.shape) == 3 else False

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

            #print(outputs[:, 1].cpu().detach().numpy().shape, outputs[:, 1].cpu().detach().numpy().min(), outputs[:, 1].cpu().detach().numpy().max())
            #print(outputs[:, 1].cpu().detach().numpy())
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
                        'tiles_per_bag': args.tiles_per_bag},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))

            '''
            if e % 20 == 0 and args.images:
                image_writer.add_images('Train Images/Before Transforms', basic_tiles.squeeze().detach().cpu().numpy(),
                                        global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms', data.squeeze().detach().cpu().numpy(),
                                        global_step=e, dataformats='NCHW')
                image_writer.add_images('Train Images/After Transforms (De-Normalized)',
                                        norm_img(data.squeeze().detach().cpu().numpy()), global_step=e,
                                        dataformats='NCHW')
            '''
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
        for idx, minibatch_val in enumerate(data_loader):
            target = minibatch_val['targets']
            data = minibatch_val['features']

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            outputs, weights, _ = model(x=None, H=data)

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
        print('Slide AUC of {:.2f} over Test set'.format(roc_auc))

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
    DATA_TYPE = 'Features'

    '''
    # These lines are for debugging:
    if sys.platform == 'darwin':
        args.last_layer_freeze = True
        args.per_patient_training = True
        args.data_limit = 500
    '''

    if sys.platform == 'darwin':
        if args.dataset == 'TCGA_ABCTB':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_293-ER-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_293-TestFold_1/Test'
                    ''
                elif args.test_fold == 2:
                    Dataset_name = r'FEATURES: Exp_299-ER-TestFold_2'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Test'

                basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/ran_293/model_data_Epoch_1000.pt'
                traind_model = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/features/338 - freezed last layer/model_data_Epoch_500.pt'

            elif args.target == 'PR':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_309-PR-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Fold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/PR/Fold_1/Test'

            elif args.target == 'Her2':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_308-Her2-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Fold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/Her2/Fold_1/Test'

        elif args.dataset == 'CAT':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_355-ER-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_355-TestFold_1/Test'
                    basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'

        elif args.dataset == 'CARMEL':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_358-ER-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_358-TestFold_1/Test'
                    basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_358-TF_1/model_data_Epoch_1000.pt'

        elif args.dataset == 'CARMEL_40':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_381-ER-TestFold_1'
                    train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Train'
                    test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Ran_Exp_381-TestFold_1/Test'
                    basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_381-TF_1/model_data_Epoch_1200.pt'


    elif sys.platform == 'linux':
        if args.dataset == 'TCGA_ABCTB':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_293-ER-TestFold_1'
                    train_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_293-ER-TestFold_1/Inference/train_inference_w_features'
                    test_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_293-ER-TestFold_1/Inference/test_inference_w_features'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_293-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'

                elif args.test_fold == 2:
                    Dataset_name = r'FEATURES: Exp_299-ER-TestFold_2'
                    train_data_dir = r'/home/womer/project/All Data/Ran_Features/299/Train'
                    test_data_dir = r'/home/womer/project/All Data/Ran_Features/299/Test'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_299-ER-TestFold_2/Model_CheckPoints/model_data_Epoch_1000.pt'

            elif args.target == 'PR':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_309-PR-TestFold_1'
                    train_data_dir = r'/home/womer/project/All Data/Ran_Features/PR/Fold_1/Train'
                    test_data_dir = r'/home/womer/project/All Data/Ran_Features/PR/Fold_1/Test'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_309-PR-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'

            elif args.target == 'Her2':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_308-Her2-TestFold_1'
                    train_data_dir = r'/home/womer/project/All Data/Ran_Features/Her2/Fold_1/Train'
                    test_data_dir = r'/home/womer/project/All Data/Ran_Features/Her2/Fold_1/Test'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_308-Her2-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'

        elif args.dataset == 'CAT':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_355-ER-TestFold_1'
                    train_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_355-ER-TestFold_1/Inference/train_w_features'
                    test_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_355-ER-TestFold_1/Inference/test_w_features'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_355-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'

        elif args.dataset == 'CARMEL':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_358-ER-TestFold_1'
                    train_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_358-ER-TestFold_1/Inference/train_w_features'
                    test_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_358-ER-TestFold_1/Inference/test_w_features'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_358-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1000.pt'

        elif args.dataset == 'CARMEL_40':
            if args.target == 'ER':
                if args.test_fold == 1:
                    Dataset_name = r'FEATURES: Exp_381-ER-TestFold_1'
                    train_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_381-ER-TestFold_1/Inference/train_w_features'
                    test_data_dir = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_381-ER-TestFold_1/Inference/test_w_features'
                    basic_model_location = r'/home/rschley/code/WSI_MIL/general_try4/runs/Exp_381-ER-TestFold_1/Model_CheckPoints/model_data_Epoch_1200.pt'

    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                          transform_type=None,
                                          tile_size=0,
                                          tiles_per_bag=args.tiles_per_bag,
                                          num_bags=args.num_bags,
                                          DX=None,
                                          DataSet_name=Dataset_name,
                                          is_per_patient=args.per_patient_training,
                                          is_last_layer_freeze=args.last_layer_freeze,
                                          is_repeating_data=args.repeating_data,
                                          Receptor=args.target + '_Features',
                                          MultiSlide=True,
                                          DataSet_Slide_magnification=0,
                                          data_limit=args.data_limit,
                                          carmel_only=args.carmel_only,
                                          Remark=' '.join(args.remark))

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
    train_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                              data_location=train_data_dir,
                                              is_per_patient=args.per_patient_training,
                                              is_repeating_tiles=args.repeating_data,
                                              bag_size=args.tiles_per_bag,
                                              target=args.target,
                                              is_train=True,
                                              data_limit=args.data_limit,
                                              test_fold=args.test_fold,
                                              carmel_only=args.carmel_only)

    test_dset = datasets.Features_MILdataset(dataset=args.dataset,
                                             data_location=test_data_dir,
                                             is_per_patient=args.per_patient_training,
                                             bag_size=args.tiles_per_bag,
                                             target=args.target,
                                             is_train=False,
                                             test_fold=args.test_fold,
                                             carmel_only=args.carmel_only)

    train_loader = DataLoader(train_dset, batch_size=args.num_bags, shuffle=True, num_workers=cpu_available, pin_memory=True)
    test_loader = DataLoader(test_dset, batch_size=args.num_bags, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Load model
    model = eval(args.model)
    if args.experiment != 0:  # In case we continue from an already trained model, than load the previous model and optimizer data:
        print('Loading pre-saved model...')
        model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'),
                                       map_location='cpu')

        model.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))

    elif args.last_layer_freeze:  # This part will load the last linear layer from the REG model into the last layer (classifier part) of the attention module
        print('Copying and freezeing last layer from model \"{}\"'.format(basic_model_location))
        basic_model_data = torch.load(basic_model_location, map_location='cpu')['model_state_dict']
        basic_model = PreActResNet50_Ron()
        basic_model.load_state_dict(basic_model_data)

        last_linear_layer_data = copy.deepcopy(basic_model.linear.state_dict())
        model.classifier.load_state_dict(last_linear_layer_data)

        for p in model.classifier.parameters():  # This part will freeze the classifier part so it won't change during training
            p.requires_grad = False

    if model.model_name in ['nets_mil.MIL_Feature_Attention_MultiBag()',
                            'nets_mil.MIL_Feature_2_Attention_MultiBag()',
                            'nets_mil.MIL_Feature_3_Attention_MultiBag()']:
        model.tiles_per_bag = args.tiles_per_bag

    # Save model data and DataSet size (and some other dataset data) to run_data.xlsx file (Only if this is a new run).
    if args.experiment == 0:
        utils.run_data(experiment=experiment, model=model.model_name)
        utils.run_data(experiment=experiment, DataSet_size=(len(train_dset), len(test_dset)))

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

    epoch = args.epochs
    from_epoch = args.from_epoch

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if DEVICE.type == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)
    print('Training No. {} has concluded successfully after {} Epochs'.format(experiment, args.epochs))