import utils
from torch.utils.data import DataLoader
import torch
import datasets
import numpy as np
from sklearn.metrics import roc_curve
import os
import sys, platform
import argparse
from tqdm import tqdm
import pickle
import resnet_v2
from collections import OrderedDict
import smtplib, ssl
import nets, PreActResNets, resnet_v2
import torchvision

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=[241], help='Use models from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=[1395, 1390], help='Use this epoch models for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=10, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, nargs="+", default=1, help=' folds to infer')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches') #RanS 8.2.21
parser.add_argument('-mp', '--model_path', type=str, default='', help='fixed path of rons model') #RanS 16.3.21 r'/home/rschley/Pathnet/results/fold_1_ER_large/checkpoint/ckpt_epoch_1467.pth'
parser.add_argument('--save_features', action='store_true', help='save features') #RanS 1.7.21
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides') #RanS 3.8.21, override run_data
parser.add_argument('--resume', type=int, default=0, help='resume a failed feature extraction') #RanS 5.10.21
parser.add_argument('--patch_dir', type=str, default='', help='patch locations directory, for use with predecided patches') #RanS 24.10.21
parser.add_argument('-sd', '--subdir', type=str, default='', help='output sub-dir') #RanS 6.12.21
args = parser.parse_args()

args.folds = list(map(int, args.folds[0])) #RanS 14.6.21

# If args.experiment contains 1 number than all epochs are from the same experiments, BUT if it is bigger than 1 than all
# the length of args.experiment should be equal to args.from_epoch
if len(args.experiment) > 1:
    if len(args.experiment) != len(args.from_epoch):
        raise Exception("number of from_epoch(-fe) should be equal to number of experiment(-ex)")
    else:
        different_experiments = True
        Output_Dirs = []
else:
    different_experiments = False

DEVICE = utils.device_gpu_cpu()
data_path = ''

print('Loading pre-saved models:')
models = []
dx = False

#decide which epochs to save features from - if model_path is used, take it.
# #else, if there only one epoch, take it. otherwise take epoch 1000
if args.save_features:
    if args.model_path != '':
        feature_epoch_ind = len(args.from_epoch)
    elif len(args.from_epoch) > 1:
        if sys.platform == 'win32':
            feature_epoch_ind = (args.from_epoch).index(16)
        else:
            feature_epoch_ind = (args.from_epoch).index(1000)
    elif len(args.from_epoch) == 1:
        feature_epoch_ind = 0

for counter in range(len(args.from_epoch)):
    epoch = args.from_epoch[counter]
    experiment = args.experiment[counter] if different_experiments else args.experiment[0]

    print('  Exp. {} and Epoch {}'.format(experiment, epoch))
    # Basic meta data will be taken from the first model (ONLY if all inferences are done from the same experiment)
    if counter == 0:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, TILE_SIZE, dx, args.target, model_name, args.mag =\
            run_data_output['Location'], run_data_output['Tile Size'], run_data_output['DX'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']
        if different_experiments:
            Output_Dirs.append(output_dir)
        fix_data_path = True
    elif counter > 0 and different_experiments:
        run_data_output = utils.run_data(experiment=experiment)
        output_dir, dx, target, model_name, args.mag =\
            run_data_output['Location'], run_data_output['DX'], run_data_output['Receptor'],\
            run_data_output['Model Name'], run_data_output['Desired Slide Magnification']
        Output_Dirs.append(output_dir)
        fix_data_path = True

    if fix_data_path:
        # we need to make some root modifications according to the computer we're running at.
        if sys.platform == 'linux':
            data_path = ''
        elif sys.platform == 'win32':
            output_dir = output_dir.replace(r'/', '\\')
            data_path = os.getcwd()

        fix_data_path = False

        # Verifying that the target receptor is not changed:
        if counter > 1 and args.target != target:
            raise Exception("Target Receptor is changed between models - DataSet cannot support this action")


    # loading basic model type
    model = eval(model_name)
    # loading model parameters from the specific epoch
    model_data_loaded = torch.load(os.path.join(data_path, output_dir, 'Model_CheckPoints',
                                                'model_data_Epoch_' + str(epoch) + '.pt'), map_location='cpu')
    model.load_state_dict(model_data_loaded['model_state_dict'])
    model.eval()
    models.append(model)

#RanS 3.8.21, override run_data dx if args.dx is true
if args.dx:
    dx = args.dx

TILE_SIZE = 128
tiles_per_iter = 20
if sys.platform == 'linux':
    TILE_SIZE = 256
    tiles_per_iter = 150
    if platform.node() in ['gipdeep4', 'gipdeep5', 'gipdeep6']:
        tiles_per_iter = 100
elif sys.platform == 'win32':
    TILE_SIZE = 256

#RanS 16.3.21, support ron's model as well
if args.model_path != '':
    if os.path.exists(args.model_path):
        args.from_epoch.append('rons_model')
        model = resnet_v2.PreActResNet50()
        model_data_loaded = torch.load(os.path.join(args.model_path), map_location='cpu')

        try:
            model.load_state_dict(model_data_loaded['net'])
        except:
            state_dict = model_data_loaded['net']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        #RanS 27.10.21, use pretrained model
        args.from_epoch.append(args.model_path.split('.')[-1])
        model = eval(args.model_path)
        model.fc = torch.nn.Identity()
        tiles_per_iter = 100
    model.eval()
    models.append(model)

if args.save_features:
    print('features will be taken from model ', str(args.from_epoch[feature_epoch_ind]))

#slide_num = 0
#if args.resume > 0 and args.save_features: #RanS 5.10.21
slide_num = args.resume

inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=tiles_per_iter,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles,
                                  desired_slide_magnification=args.mag,
                                  dx=dx,
                                  resume_slide=slide_num,
                                  patch_dir=args.patch_dir
                                  )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
#NUM_SLIDES = len(inf_dset.valid_slide_indices)
NUM_SLIDES = len(inf_dset.image_file_names) #RanS 24.5.21, valid_slide_indices always counts non-dx slides
NUM_SLIDES_SAVE = 50
print('NUM_SLIDES: ', str(NUM_SLIDES))

all_targets = []
all_scores, all_labels = np.zeros((NUM_SLIDES, NUM_MODELS)), np.zeros((NUM_SLIDES, NUM_MODELS))
patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))
#patch_locs_all = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles, 2)) #RanS 17.10.21
patch_locs_all = np.empty((NUM_SLIDES, args.num_tiles, 2)) #RanS 17.10.21
#patch_locs_all = np.empty((7665, args.num_tiles, 2)) #temp RanS 10.11.21
if args.save_features:
    #features_all = np.empty((NUM_SLIDES_SAVE, NUM_MODELS, args.num_tiles, 512))
    features_all = np.empty((NUM_SLIDES_SAVE, 1, args.num_tiles, 512)) #RanS 30.9.21
    features_all[:] = np.nan
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
all_slide_datasets = np.zeros(NUM_SLIDES, dtype=object)
patch_scores[:] = np.nan
patch_locs_all[:] = np.nan

# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
correct_pos = [0 for ii in range(NUM_MODELS)] # RanS 12.7.21
correct_neg = [0 for ii in range(NUM_MODELS)] # RanS 12.7.21

if args.resume:
    # load the inference state
    resume_file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir,
                                    'Exp_' + str(args.experiment[0])
                                    + '-Folds_' + str(args.folds) + '_' + str(
                                        args.target) + '-Tiles_' + str(
                                        args.num_tiles) + '_resume_slide_num_' + str(slide_num) + '.data')
    with open(resume_file_name, 'rb') as filehandle:
        resume_data = pickle.load(filehandle)
    all_labels, all_targets, all_scores, total_pos, correct_pos, total_neg, \
    correct_neg, patch_scores, all_slide_names, all_slide_datasets, NUM_SLIDES, patch_locs_all = resume_data
    #correct_neg, patch_scores, all_slide_names, all_slide_datasets, NUM_SLIDES = resume_data  # temp RanS 10.11.21

else:
    resume_file_name = 0

if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference')):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference'))

if not os.path.isdir(os.path.join(data_path, output_dir, 'Inference', args.subdir)):
    os.mkdir(os.path.join(data_path, output_dir, 'Inference', args.subdir))

print('slide_num0 = ', str(slide_num)) #temp
with torch.no_grad():
    for batch_idx, MiniBatch_Dict in enumerate(tqdm(inf_loader)):
        #print('slide num = ', str(inf_loader.dataset.slide_num), 'batch_num = ', str(batch_idx)) #temp
        #if args.resume and (inf_loader.dataset.slide_num < slide_num):
        #    print('skip') #temp
        #    continue

        # Unpacking the data:
        data = MiniBatch_Dict['Data']
        target = MiniBatch_Dict['Label']
        time_list = MiniBatch_Dict['Time List']
        last_batch = MiniBatch_Dict['Is Last Batch']
        slide_file = MiniBatch_Dict['Slide Filename']
        slide_dataset = MiniBatch_Dict['Slide DataSet']
        patch_locs = MiniBatch_Dict['Patch Loc']
        #patch_loc_inds = MiniBatch_Dict['Patch loc index']
        #slide_size_ind = MiniBatch_Dict['Slide Index Size']
        #slide_size = MiniBatch_Dict['Slide Size']

        if new_slide:
            #n_tiles = inf_loader.dataset.num_tiles[slide_num]  # RanS 1.7.21
            n_tiles = inf_loader.dataset.num_tiles[slide_num - args.resume]  # RanS 6.10.21
            #scores_0, scores_1 = [np.zeros(0)] * NUM_MODELS, [np.zeros(0)] * NUM_MODELS
            #scores_0, scores_1 = [np.zeros(n_tiles)] * NUM_MODELS, [np.zeros(n_tiles)] * NUM_MODELS #RanS 1.7.21
            scores_0 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)] # RanS 12.7.21
            scores_1 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)]  # RanS 12.7.21
            patch_locs_1_slide = np.zeros((n_tiles, 2))  # RanS 10.8.21
            #patch_locs_inds_1 = [np.zeros((n_tiles, 2)) for ii in range(NUM_MODELS)]  # RanS 10.8.21
            if args.save_features:
                #feature_arr = [np.zeros((n_tiles, 512))] * NUM_MODELS #RanS 1.7.21
                #feature_arr = [np.zeros((n_tiles, 512)) for ii in range(NUM_MODELS)]  # RanS 1.7.21
                feature_arr = [np.zeros((n_tiles, 512))]  # RanS 30.9.21
            target_current = target
            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)
        data, target = data.to(DEVICE), target.to(DEVICE)

        #patch_locs_1_slide[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = patch_locs  # RanS 10.8.21
        patch_locs_1_slide[slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data),:] = np.array(patch_locs)  # RanS 19.10.21

        for index, model in enumerate(models):
            model.to(DEVICE)

            if model._get_name() == 'PreActResNet_Ron':
                scores, features = model(data)
            else:
                #use resnet only for features, dump scores RanS 27.10.21
                features = model(data)
                scores = torch.zeros((len(data), 2))

            scores = torch.nn.functional.softmax(scores, dim=1) #RanS 11.3.21

            scores_0[index][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data)] = scores[:, 0].cpu().detach().numpy()
            scores_1[index][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data)] = scores[:, 1].cpu().detach().numpy()
            #patch_locs_1[index][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = patch_locs  # RanS 10.8.21, cancelled

            if args.save_features:
                if index == feature_epoch_ind:
                    feature_arr[0][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data), :] = features.cpu().detach().numpy()

        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target.cpu().numpy()[0][0])
            if target == 1:
                total_pos += 1
            else:
                total_neg += 1

            if args.save_features:
                features_all[slide_num % NUM_SLIDES_SAVE, 0, :len(feature_arr[0])] = feature_arr[0]

            patch_locs_all[slide_num, :len(patch_locs_1_slide), :] = patch_locs_1_slide #RanS 17.10.21

            for model_num in range(NUM_MODELS):
                current_slide_tile_scores = np.vstack((scores_0[model_num], scores_1[model_num]))

                predicted = current_slide_tile_scores.mean(1).argmax()
                patch_scores[slide_num, model_num, :len(scores_1[model_num])] = scores_1[model_num]

                #patch_locs_all[slide_num, model_num, :len(patch_locs_1[model_num]), :] = patch_locs_1[model_num] #cancelled
                all_scores[slide_num, model_num] = scores_1[model_num].mean()
                all_labels[slide_num, model_num] = predicted
                all_slide_names[slide_num] = slide_file[0]
                all_slide_datasets[slide_num] = slide_dataset[0]

                if target == 1 and predicted == 1:
                    correct_pos[model_num] += 1
                elif target == 0 and predicted == 0:
                    correct_neg[model_num] += 1

            slide_num += 1

            # RanS 6.7.21, save features every NUM_SLIDES_SAVE slides
            if slide_num % NUM_SLIDES_SAVE == 0:
                #save the inference state
                prev_resume_file_name = resume_file_name
                resume_file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir,
                                                 'Exp_' + str(args.experiment[0])
                                                 + '-Folds_' + str(args.folds) + '_' + str(
                                                     args.target) + '-Tiles_' + str(
                                                     args.num_tiles) + '_resume_slide_num_' + str(slide_num) + '.data')
                resume_data = [all_labels, all_targets, all_scores,
                                  total_pos, correct_pos, total_neg, correct_neg,
                                  patch_scores, all_slide_names, all_slide_datasets, NUM_SLIDES, patch_locs_all]

                with open(resume_file_name, 'wb') as filehandle:
                    pickle.dump(resume_data, filehandle)
                #delete previous resume file
                if os.path.isfile(prev_resume_file_name):
                    os.remove(prev_resume_file_name)

                #save features
                if args.save_features:
                    #for model_num in range(NUM_MODELS):
                    feature_file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir,
                                                     'Model_Epoch_' + str(args.from_epoch[feature_epoch_ind])
                                                     + '-Folds_' + str(args.folds) + '_' + str(
                                                         args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_' + str(slide_num) + '.data')
                    inference_data = [all_labels[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind],
                                      all_targets[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      all_scores[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind],
                                      np.squeeze(patch_scores[slide_num-NUM_SLIDES_SAVE:slide_num, feature_epoch_ind, :]),
                                      all_slide_names[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      features_all,
                                      all_slide_datasets[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      patch_locs_all[slide_num-NUM_SLIDES_SAVE:slide_num]]
                    with open(feature_file_name, 'wb') as filehandle:
                        pickle.dump(inference_data, filehandle)
                    print('saved output for ', str(slide_num), ' slides')
                    features_all = np.empty((NUM_SLIDES_SAVE, 1, args.num_tiles, 512))
                    features_all[:] = np.nan

#save features for last slides
if args.save_features and slide_num % NUM_SLIDES_SAVE != 0:
    #for model_num in range(NUM_MODELS):
    feature_file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir,
                                     'Model_Epoch_' + str(args.from_epoch[feature_epoch_ind])
                                     + '-Folds_' + str(args.folds) + '_' + str(
                                         args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_last.data')
    last_save = slide_num // NUM_SLIDES_SAVE * NUM_SLIDES_SAVE
    inference_data = [all_labels[last_save:slide_num, feature_epoch_ind],
                      all_targets[last_save:slide_num],
                      all_scores[last_save:slide_num, feature_epoch_ind],
                      np.squeeze(patch_scores[last_save:slide_num, feature_epoch_ind, :]),
                      all_slide_names[last_save:slide_num],
                      features_all[:slide_num-last_save],
                      all_slide_datasets[last_save:slide_num],
                      patch_locs_all[last_save:slide_num]]
    with open(feature_file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)
    print('saved output for ', str(slide_num), ' slides')

for model_num in range(NUM_MODELS):
    if different_experiments:
        output_dir = Output_Dirs[model_num]

    fpr, tpr, _ = roc_curve(all_targets, all_scores[:, model_num])

    # Save roc_curve to file:
    file_name = os.path.join(data_path, output_dir, 'Inference', args.subdir, 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '_' + str(args.target) + '-Tiles_' + str(args.num_tiles) + '.data')
    inference_data = [fpr, tpr, all_labels[:, model_num], all_targets, all_scores[:, model_num],
                      total_pos, correct_pos[model_num], total_neg, correct_neg[model_num], NUM_SLIDES,
                      np.squeeze(patch_scores[:, model_num, :]), all_slide_names, all_slide_datasets,
                      np.squeeze(patch_locs_all)]
                      #np.squeeze(patch_scores[:, model_num, :]), all_slide_names, np.squeeze(patch_locs_all[:, model_num, :, :]), np.squeeze(patch_locs_inds_all[:, model_num, :, :]), all_slide_size, all_slide_size_ind]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)

    experiment = args.experiment[model_num] if different_experiments else args.experiment[0]
    print('For model from Experiment {} and Epoch {}: {} / {} correct classifications'
          .format(experiment,
                  args.from_epoch[model_num],
                  int(len(all_labels[:, model_num]) - np.abs(np.array(all_targets) - np.array(all_labels[:, model_num])).sum()),
                  len(all_labels[:, model_num])))
print('Done !')

#delete last resume file
if os.path.isfile(resume_file_name):
    os.remove(resume_file_name)

# finished training, send email if possible
if os.path.isfile('mail_cfg.txt'):
    with open("mail_cfg.txt", "r") as f:
        text = f.readlines()
        receiver_email = text[0][:-1]
        password = text[1]

    port = 465  # For SSL
    sender_email = "gipmed.python@gmail.com"

    message = 'Subject: finished inference for experiment ' + str(args.experiment)

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        print('email sent to ' + receiver_email)
