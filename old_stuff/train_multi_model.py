import utils
import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch
import nets
from tqdm import tqdm
import time
import argparse
import numpy as np
import sys
from old_stuff import utils_multi_ops
from typing import List


parser = argparse.ArgumentParser(description='WSI_MIL Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=5, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=2, type=int, help='Epochs to run')
parser.add_argument('-tt', '--transform_type', type=str, default='flip', help='keyword for transform type')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-diffslides', dest='different_slides', action='store_true', help='Use more than one slide in each bag')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-tar', '--target', type=str, default='ER', help='DataSet to use')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-nb', '--num_bags', type=int, default=5, help='Number of bags in each minibatch')
parser.add_argument('-tpb', '--tiles_per_bag', type=int, default=1, help='Tiles Per Bag')
parser.add_argument('--eval_rate', type=int, default=1, help='Evaluate validation set every # epochs')

args = parser.parse_args()


def norm_img(img):
    img -= img.min()
    img /= img.max()
    return img



def train(models, optimizers, dloader_train: DataLoader, dloader_test: DataLoader, DEVICE, print_timing: bool=False):
    """
    This function trains the model
    :return:
    """

    print()
    print('Training will be conducted with {} bags per MiniBatch'.format(num_bags))
    print('Start Training...')

    for e in range(from_epoch, epoch + from_epoch):
        time_epoch_start = time.time()

        performance_parameters = utils_multi_ops.init_performance_parameters(len(models))

        performance_parameters['epoch'] = e

        print('Epoch {}:'.format(e))
        for model in models:
            model.train()

        for batch_idx, (data, target, time_list, image_file, basic_tiles) in enumerate(tqdm(dloader_train)):
            train_start = time.time()

            target = target.squeeze(1)
            data, target = data.to(DEVICE), target.to(DEVICE)

            performance_parameters['targets_train'] = np.concatenate((performance_parameters['targets_train'],
                                                                      target.cpu().detach().numpy()))
            performance_parameters['total_train'] += target.size(0)
            performance_parameters['total_pos_train'] += target.eq(1).sum().item()
            performance_parameters['total_neg_train'] += target.eq(0).sum().item()

            for index in range(len(models)):
                optimizer, model = optimizers[index], models[index]
                optimizer.zero_grad()
                model.to(DEVICE)
                scores = model(data)
                loss = criterion(scores, target)
                loss.backward()
                optimizer.step()

                performance_parameters['loss_train'][index] += loss.item()
                _, predicted = scores.max(1)

                performance_parameters['scores_train'][index] = np.concatenate((performance_parameters['scores_train'][index],
                                                                                scores[:, 1].cpu().detach().numpy()))
                performance_parameters['correct_labeling_train'][index] += predicted.eq(target).sum().item()
                performance_parameters['correct_pos_train'][index] += predicted[target.eq(1)].eq(1).sum().item()
                performance_parameters['correct_neg_train'][index] += predicted[target.eq(0)].eq(0).sum().item()
                all_writers[index].add_scalar('Loss', loss.item(), batch_idx + e * len(dloader_train))

        time_epoch = (time.time() - time_epoch_start) / 60
        performance_parameters['time_train'] = time_epoch
        '''
        if args.time:
            time_writer.add_scalar('Time/Full Epoch [min]', time_epoch, e)
        '''

        utils_multi_ops.write_performance_data(all_writers, performance_parameters)

        if e % args.eval_rate == 0:
            check_accuracy(models, all_writers, dloader_test, DEVICE)


            # Update 'Last Epoch' at run_data.xlsx file:
            utils.run_data_multi_model(experiments=experiments, epoch=e)

            # Save model to file:
            utils_multi_ops.save_models(models, optimizers, args.output_dir, e)


def check_accuracy(models: List[nn.Module], writers: list, data_loader: DataLoader, DEVICE):

    for model in models:
        model.eval()

    with torch.no_grad():
        for idx, (data, target, time_list, image_file_name, basic_tiles) in enumerate(data_loader):
            target = target.squeeze(1)
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            performance_parameters['targets_test'] = np.concatenate((performance_parameters['targets_test'],
                                                                     target.cpu().detach().numpy()))
            performance_parameters['total_test'] += target.size(0)
            performance_parameters['total_pos_test'] += target.eq(1).sum().item()
            performance_parameters['total_neg_test'] += target.eq(0).sum().item()

            for index in range(len(models)):
                model = models[index]
                model.to(DEVICE)
                scores = model(data)
                _, predicted = scores.max(1)

                performance_parameters['scores_test'][index] = np.concatenate((performance_parameters['scores_test'][index],
                                                                               scores[:, 1].cpu().detach().numpy()))

                performance_parameters['correct_labeling_test'][index] += predicted.eq(target).sum().item()
                performance_parameters['correct_pos_test'][index] += predicted[target.eq(1)].eq(1).sum().item()
                performance_parameters['correct_neg_test'][index] += predicted[target.eq(0)].eq(0).sum().item()

    utils_multi_ops.write_performance_data(writers, performance_parameters, is_train=False)

    for model in models:
        model.train()

##################################################################################################

if __name__ == '__main__':
    # Device definition:
    DEVICE = utils.device_gpu_cpu()

    # Tile size definition:
    TILE_SIZE = 128
    TILES_PER_BAG = args.tiles_per_bag
    num_bags = args.num_bags

    if sys.platform == 'linux':
        TILE_SIZE = 256


    # Saving/Loading run meta data to/from file:
    if args.experiment is 0:

        ##############    Create model    ##############
        models, train_types = [], []
        models.append(nets.ResNet50(num_classes=2))
        train_types.append('')
        '''models.append(nets.ResNet_50_NO_downsample())
        train_types.append('')
        models.append(PreActResNets.PreActResNet50_Ron())
        train_types.append('')'''

        NUM_EXPERIMENTS = len(models)
        args.output_dir, experiments = [], []
        for index in range(NUM_EXPERIMENTS):
            output_dir, experiment = utils.run_data(test_fold=args.test_fold,
                                                    transform_type=args.transform_type,
                                                    tile_size=TILE_SIZE,
                                                    tiles_per_bag=args.tiles_per_bag,
                                                    num_bags=args.num_bags,
                                                    DX=args.dx,
                                                    DataSet_name=args.dataset,
                                                    Receptor=args.target,
                                                    MultiSlide=False)

            args.output_dir.append(output_dir)
            experiments.append(experiment)

    # Get number of available CPUs:
    cpu_available = utils.get_cpu()

    # Get data:
    train_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                         tile_size=TILE_SIZE,
                                         bag_size=TILES_PER_BAG,
                                         target_kind=args.target,
                                         test_fold=args.test_fold,
                                         train=True,
                                         print_timing=args.time,
                                         transform_type=args.transform_type,
                                         DX=args.dx,
                                         get_images=args.images)

    test_dset = datasets.WSI_MILdataset(DataSet=args.dataset,
                                        tile_size=TILE_SIZE,
                                        bag_size=TILES_PER_BAG,
                                        target_kind=args.target,
                                        test_fold=args.test_fold,
                                        train=False,
                                        print_timing=False,
                                        transform_type='none',
                                        DX=args.dx,
                                        get_images=args.images)


    train_loader = DataLoader(train_dset, batch_size=args.num_bags, shuffle=True, num_workers=cpu_available, pin_memory=True)
    if args.tiles_per_bag == 1:
        test_loader = DataLoader(test_dset, batch_size=args.num_bags, shuffle=False, num_workers=cpu_available, pin_memory=True)
    else:
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=cpu_available, pin_memory=True)

    # Save transformation data to 'run_data.xlsx'
    transformation_string = ', '.join([str(train_dset.transform.transforms[i]) for i in range(len(train_dset.transform.transforms))])
    utils.run_data_multi_model(experiments=experiments, transformation_string=transformation_string)

    utils.run_data_multi_model(experiments=experiments, models=models)

    epoch = args.epochs
    from_epoch = args.from_epoch

    # In case we continue from an already trained model, than load the previous model and optimizer data:
    if args.experiment is not 0:
        print('Loading pre-saved model...')
        # TODO: modify this to multi model
        pass
        '''
        model_data_loaded = torch.load(os.path.join(args.output_dir_model_1,
                                                    'Model_CheckPoints',
                                                    'model_data_Epoch_' + str(args.from_epoch) + '.pt'), map_location='cpu')

        model_1.load_state_dict(model_data_loaded['model_state_dict'])

        from_epoch = args.from_epoch + 1
        print()
        print('Resuming training of Experiment {} from Epoch {}'.format(args.experiment, args.from_epoch))'''

    optimizers = utils_multi_ops.init_optimizers_adam(models)

    if DEVICE.type == 'cuda':
        cudnn.benchmark = True
    '''
    if args.experiment is not 0:
        optimizer.load_state_dict(model_data_loaded['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)
    '''
    criterion = nn.CrossEntropyLoss()

    # Initialized SummaryWriters:
    all_writers = utils_multi_ops.create_writers(args.output_dir)
    utils_multi_ops.write_basic_data_to_writer(all_writers, experiments, models, train_types, train_loader, test_loader)
    performance_parameters = utils_multi_ops.init_performance_parameters(len(models))

    '''
    if args.images:
        image_writer = SummaryWriter(os.path.join(writer_folder, 'image'))
    if args.time:
        time_writer = SummaryWriter(os.path.join(writer_folder, 'time'))
    '''

    train(models, optimizers, train_loader, test_loader, DEVICE=DEVICE, print_timing=args.time)

    # Close SummaryWriters:
    for writer in all_writers:
        writer.close()
