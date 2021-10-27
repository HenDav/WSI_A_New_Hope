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
import smtplib, ssl
import psutil
from Cox_Loss import Cox_loss
from sksurv.metrics import concordance_index_censored

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='Survival Synthetic', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='Survival Time', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('--batch_size', default=18, type=int, help='size of batch')
parser.add_argument('--model', default='PreActResNets.PreActResNet50_Ron()', type=str, help='net to use')
#parser.add_argument('--model', default='nets.ResNet50(pretrained=True)', type=str, help='net to use')
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--loan', action='store_true', help='Localized Annotation for strongly supervised training') #RanS 17.6.21
parser.add_argument('--er_eq_pr', action='store_true', help='while training, take only er=pr examples') #RanS 27.6.21
parser.add_argument('--focal', action='store_true', help='use focal loss with gamma=2') #RanS 18.7.21
parser.add_argument('--slide_per_block', action='store_true', help='for carmel, take only one slide per block') #RanS 17.8.21
parser.add_argument('-baldat', '--balanced_dataset', dest='balanced_dataset', action='store_true', help='take same # of positive and negative patients from each dataset')  # RanS 5.9.21


args = parser.parse_args()

EPS = 1e-7


def train(model: nn.Module, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, optimizer, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """
    if not os.path.isdir(os.path.join(args.output_dir, 'Model_CheckPoints')):
        os.mkdir(os.path.join(args.output_dir, 'Model_CheckPoints'))
    writer_folder = os.path.join(args.output_dir, 'writer')
    all_writer = SummaryWriter(os.path.join(writer_folder, 'all'))

    if from_epoch == 0:
        all_writer.add_text('Experiment No.', str(experiment))
        all_writer.add_text('Train type', 'Regular')
        all_writer.add_text('Model type', str(type(model)))
        if args.dataset != 'Survival Synthetic':
            all_writer.add_text('Data type', dloader_train.dataset.DataSet)
            all_writer.add_text('Train Folds', str(dloader_train.dataset.folds).strip('[]'))
            all_writer.add_text('Test Folds', str(dloader_test.dataset.folds).strip('[]'))
            all_writer.add_text('Transformations', str(dloader_train.dataset.transform))
            all_writer.add_text('Receptor Type', str(dloader_train.dataset.target_kind))

    if print_timing:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))

    print('Start Training...')
    previous_epoch_loss = 1e5

    for e in range(from_epoch, epoch):
        outputs_train = np.zeros(0)
        all_targets, all_outputs, all_censored = [], [], []
        train_loss = 0

        slide_names = []
        print('Epoch {}:'.format(e))

        # RanS 11.7.21
        process = psutil.Process(os.getpid())
        #print('RAM usage:', process.memory_info().rss/1e9, 'GB')

        model.train()

        for batch_idx, minibatch in enumerate(tqdm(dloader_train)):
            time_stamp = batch_idx + e * len(train_loader)
            data = minibatch['Features']
            target = minibatch['Target']
            censored = minibatch['Censored']
            all_targets.extend(target.numpy())
            all_censored.extend(censored.numpy())
            data, target = data.to(DEVICE), target.to(DEVICE)
            model.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(data)
            outputs = torch.reshape(outputs, [outputs.size(0)])
            all_outputs.extend(outputs.detach().cpu().numpy())

            loss = criterion(outputs, target, censored)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            all_writer.add_scalar('Train/Loss per Minibatch', loss, time_stamp)

            if DEVICE.type == 'cuda' and print_timing:
                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                #print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
                all_writer.add_scalar('GPU/gpu', res.gpu, batch_idx + e * len(dloader_train))
                all_writer.add_scalar('GPU/gpu-mem', res.memory, batch_idx + e * len(dloader_train))

        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_targets, all_outputs)
        all_writer.add_scalar('Train/Loss Per Epoch Per Instance', train_loss / len(all_targets), e)
        all_writer.add_scalar('Train/Accuracy', c_index, e)
        #print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train Accuracy: {:.2f}% ({} / {}), Time: {:.0f} m'

        print('Finished Epoch: {}, Loss: {:.2f}, Loss Delta: {:.3f}, Train C-index per patch: {:.3f}'
              .format(e,
                      train_loss,
                      previous_epoch_loss - train_loss,
                      c_index))
        previous_epoch_loss = train_loss

        # Update 'Last Epoch' at run_data.xlsx file:
        utils.run_data(experiment=experiment, epoch=e)

        # Save model to file:
        try:
            model_state_dict = model.module.state_dict()
        except AttributeError:
            model_state_dict = model.state_dict()
        torch.save({'epoch': e,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tile_size': TILE_SIZE,
                    'tiles_per_bag': 1},
                   os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Last_Epoch.pt'))

        if e % args.eval_rate == 0:
            c_index_test = check_accuracy(e, all_writer)
            # Save model to file:
            try:
                model_state_dict = model.module.state_dict()
            except AttributeError:
                model_state_dict = model.state_dict()
            torch.save({'epoch': e,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        'tile_size': TILE_SIZE,
                        'tiles_per_bag': 1},
                       os.path.join(args.output_dir, 'Model_CheckPoints', 'model_data_Epoch_' + str(e) + '.pt'))
            print('saved checkpoint to', args.output_dir) #RanS 23.6.21


    all_writer.close()

    if print_timing:
        time_writer.close()


def check_accuracy(epoch: int, all_writer):

    test_loss = 0
    all_outputs, all_targets, all_censored = [], [], []
    slide_names = []

    model.eval()

    with torch.no_grad():
        for idx, minibatch in enumerate(test_loader):
            data = minibatch['Features']
            target = minibatch['Target']
            censored = minibatch['Censored']
            all_targets.extend(target.numpy())
            all_censored.extend(censored.numpy())

            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            model.to(DEVICE)

            outputs = model(data)
            outputs = torch.reshape(outputs, [outputs.size(0)])
            all_outputs.extend(outputs.detach().cpu().numpy())

            loss = criterion(outputs, target, censored)
            test_loss += loss.item()

        c_index, num_concordant_pairs, num_discordant_pairs, _, _ = concordance_index_censored(np.invert(all_censored), all_targets, all_outputs)
        print('Test C-index: {:.3f}'.format(c_index))
        all_writer.add_scalar('Test/C-index Per Epoch', c_index, epoch)
        all_writer.add_scalar('Test/Loss Per Epoch Per Instance', test_loss / len(all_targets), epoch)


    model.train()
    return c_index

########################################################################################################
########################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Tile size definition:
    TILE_SIZE = 128

    if sys.platform == 'linux' or sys.platform == 'win32':
        TILE_SIZE = 256

    # Saving/Loading run meta data to/from file:
    if args.experiment == 0:
        run_data_results = utils.run_data(test_fold=args.test_fold,
                                                     #transformations=args.transformation,
                                                     transform_type=args.transform_type,
                                                     tile_size=TILE_SIZE,
                                                     tiles_per_bag=1,
                                                     DX=args.dx,
                                                     DataSet_name=args.dataset,
                                                     Receptor=args.target,
                                                     num_bags=args.batch_size)

        args.output_dir, experiment = run_data_results['Location'], run_data_results['Experiment']
    else:
        run_data_output = utils.run_data(experiment=args.experiment)
        args.output_dir, args.test_fold, args.transform_type, TILE_SIZE, tiles_per_bag, \
        args.batch_size, args.dx, args.dataset, args.target, args.model, args.mag =\
            run_data_output['Location'], run_data_output['Test Fold'], run_data_output['Transformations'], run_data_output['Tile Size'],\
            run_data_output['Tiles Per Bag'], run_data_output['Num Bags'], run_data_output['DX'], run_data_output['Dataset Name'],\
            run_data_output['Receptor'], run_data_output['Model Name'], run_data_output['Desired Slide Magnification']

        print('args.dataset:', args.dataset)
        print('args.target:', args.target)
        print('args.args.batch_size:', args.batch_size)
        print('args.output_dir:', args.output_dir)
        print('args.test_fold:', args.test_fold)
        print('args.transform_type:', args.transform_type)
        print('args.dx:', args.dx)

        experiment = args.experiment

    # Get number of available CPUs and compute number of workers:
    cpu_available = utils.get_cpu()
    num_workers = cpu_available
    #num_workers = cpu_available * 2 #temp RanS 9.8.21
    #num_workers = cpu_available//2  # temp RanS 9.8.21
    #num_workers = 4 #temp RanS 24.3.21

    if sys.platform == 'win32':
        num_workers = 0 #temp RanS 3.5.21

    print('num workers = ', num_workers)

    # Get data:
    if args.dataset == 'Survival Synthetic':
        train_dset = datasets.C_Index_Test_Dataset(train=True)
        test_dset = datasets.C_Index_Test_Dataset(train=False)
    else:
        train_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                             tile_size=TILE_SIZE,
                                             target_kind=args.target,
                                             test_fold=args.test_fold,
                                             train=True,
                                             print_timing=args.time,
                                             transform_type=args.transform_type,
                                             n_tiles=args.n_patches_train,
                                             color_param=args.c_param,
                                             get_images=args.images,
                                             desired_slide_magnification=args.mag,
                                             DX=args.dx,
                                             loan=args.loan,
                                             er_eq_pr=args.er_eq_pr,
                                             slide_per_block=args.slide_per_block,
                                             balanced_dataset=args.balanced_dataset
                                             )
        test_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                            tile_size=TILE_SIZE,
                                            target_kind=args.target,
                                            test_fold=args.test_fold,
                                            train=False,
                                            print_timing=False,
                                            transform_type='none',
                                            n_tiles=args.n_patches_test,
                                            get_images=args.images,
                                            desired_slide_magnification=args.mag,
                                            DX=args.dx,
                                            loan=args.loan,
                                            er_eq_pr=args.er_eq_pr
                                            )
    sampler = None
    do_shuffle = True
    if args.balanced_sampling:
        labels = pd.DataFrame(train_dset.target * train_dset.factor)
        n_pos = np.sum(labels == 'Positive').item()
        n_neg = np.sum(labels == 'Negative').item()
        weights = pd.DataFrame(np.zeros(len(train_dset)))
        weights[np.array(labels == 'Positive')] = 1 / n_pos
        weights[np.array(labels == 'Negative')] = 1 / n_neg
        do_shuffle = False  # the sampler shuffles
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(), num_samples=len(train_dset))

    args.batch_size = 20

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle,
                              num_workers=num_workers, pin_memory=True, sampler=sampler)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size*2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # RanS 20.6.21
    if args.loan:
        train_labels_df = pd.DataFrame({'slide_name': train_loader.dataset.image_file_names, 'label': train_loader.dataset.target})
        test_labels_df = pd.DataFrame({'slide_name': test_loader.dataset.image_file_names, 'label': test_loader.dataset.target})

    # Save transformation data to 'run_data.xlsx'
    if args.dataset != 'Survival Synthetic':
        transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
        utils.run_data(experiment=experiment, transformation_string=transformation_string)

    # Load model
    if args.dataset == 'Survival Synthetic':
        model = nn.Linear(8, 1)
        model.model_name = 'Survival_Synthetic'

    else:
        model = eval(args.model)
    if args.target == 'Survival Time' and args.dataset != 'Survival Synthetic':
        model.change_num_classes(num_classes=1)  # This will convert the liner (classifier) layer into the beta layer

    # Save model data and data-set size to run_data.xlsx file (Only if this is a new run).
    if args.experiment == 0 and args.dataset != 'Survival Synthetic':
        utils.run_data(experiment=experiment, model=model.model_name)
        utils.run_data(experiment=experiment, DataSet_size=(train_dset.real_length, test_dset.real_length))
        utils.run_data(experiment=experiment, DataSet_Slide_magnification=train_dset.desired_magnification)

        # Saving code files, args and main file name (this file) to Code directory within the run files.
        utils.save_code_files(args, train_dset)

    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment != 0:
        print('Loading pre-saved model...')
        if from_epoch == 0: #RanS 25.7.21, load last epoch
            model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                        'Model_CheckPoints',
                                                        'model_data_Last_Epoch.pt'),
                                           map_location='cpu')
            from_epoch = model_data_loaded['epoch'] + 1
        else:
            model_data_loaded = torch.load(os.path.join(args.output_dir,
                                                        'Model_CheckPoints',
                                                        'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')
            from_epoch = args.from_epoch + 1
        model.load_state_dict(model_data_loaded['model_state_dict'])

        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, from_epoch))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if DEVICE.type == 'cuda':
        model = torch.nn.DataParallel(model) #DataParallel, RanS 1.8.21
        cudnn.benchmark = True

        # RanS 28.1.21
        # https://forums.fast.ai/t/show-gpu-utilization-metrics-inside-training-loop-without-subprocess-call/26594
        if args.time:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    if args.experiment != 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

    if args.focal:
        criterion = utils.FocalLoss(gamma=2)  # RanS 18.7.21
        criterion.to(DEVICE) #RanS 20.7.21
    elif args.target == 'Survival Time':
        criterion = Cox_loss
    else:
        criterion = nn.CrossEntropyLoss()

    train(model, train_loader, test_loader, DEVICE=DEVICE, optimizer=optimizer, print_timing=args.time)

    #finished training, send email if possible
    if os.path.isfile('mail_cfg.txt'):
        with open("mail_cfg.txt", "r") as f:
            text = f.readlines()
            receiver_email = text[0][:-1]
            password = text[1]

        port = 465  # For SSL
        sender_email = "gipmed.python@gmail.com"

        message = 'Subject: finished running experiment ' + str(experiment)

        # Create a secure SSL context
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
            print('email sent to ' + receiver_email)
