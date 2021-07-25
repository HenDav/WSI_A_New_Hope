import utils
from torch.utils.data import DataLoader
import torch
import datasets
import numpy as np
from sklearn.metrics import roc_curve
import os
import sys
import argparse
from tqdm import tqdm
import pickle
import resnet_v2
from collections import OrderedDict
import smtplib, ssl

parser = argparse.ArgumentParser(description='WSI_REG Slide inference')
parser.add_argument('-ex', '--experiment', nargs='+', type=int, default=[241], help='Use models from this experiment')
parser.add_argument('-fe', '--from_epoch', nargs='+', type=int, default=[1395, 1390], help='Use this epoch models for inference')
parser.add_argument('-nt', '--num_tiles', type=int, default=10, help='Number of tiles to use')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-f', '--folds', type=list, nargs="+", default=[1], help=' folds to infer')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches') #RanS 8.2.21
parser.add_argument('-mp', '--model_path', type=str, default='', help='fixed path of rons model') #RanS 16.3.21 r'/home/rschley/Pathnet/results/fold_1_ER_large/checkpoint/ckpt_epoch_1467.pth'
parser.add_argument('--save_features', action='store_true', help='save features') #RanS 1.7.21
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

for counter in range(len(args.from_epoch)):
    epoch = args.from_epoch[counter]
    experiment = args.experiment[counter] if different_experiments else args.experiment[0]

    print('  Exp. {} and Epoch {}'.format(experiment, epoch))
    # Basic meta data will be taken from the first model (ONLY if all inferences are done from the same experiment)
    if counter == 0:
        output_dir, _, _, TILE_SIZE, _, _, dx, _, args.target, _, model_name, args.mag = utils.run_data(experiment=experiment)
        if different_experiments:
            Output_Dirs.append(output_dir)
        fix_data_path = True
    elif counter > 0 and different_experiments:
        output_dir, _, _, _, _, _, dx, _, target, _, model_name = utils.run_data(experiment=experiment)
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

#RanS 16.3.21, support ron's model as well
if args.model_path != '':
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
    model.eval()
    models.append(model)

TILE_SIZE = 128
tiles_per_iter = 20
if sys.platform == 'linux':
    TILE_SIZE = 256
    tiles_per_iter = 150
elif sys.platform == 'win32':
    TILE_SIZE = 256

#tiles_per_iter = 1 #temp RanS 22.7.21

inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                  tile_size=TILE_SIZE,
                                  tiles_per_iter=tiles_per_iter,
                                  target_kind=args.target,
                                  folds=args.folds,
                                  num_tiles=args.num_tiles,
                                  desired_slide_magnification=args.mag,
                                  dx=dx
                                  )
inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

new_slide = True

NUM_MODELS = len(models)
#NUM_SLIDES = len(inf_dset.valid_slide_indices)
NUM_SLIDES = len(inf_dset.image_file_names) #RanS 24.5.21, valid_slide_indices always counts non-dx slides
NUM_SLIDES_SAVE = 50
print('NUM_SLIDES: ', str(NUM_SLIDES)) #temp RanS 24.5.21

all_targets = []
all_scores, all_labels = np.zeros((NUM_SLIDES, NUM_MODELS)), np.zeros((NUM_SLIDES, NUM_MODELS))
patch_scores = np.empty((NUM_SLIDES, NUM_MODELS, args.num_tiles))
features_all = np.empty((NUM_SLIDES_SAVE, NUM_MODELS, args.num_tiles, 512))
all_slide_names = np.zeros(NUM_SLIDES, dtype=object)
patch_scores[:] = np.nan
features_all[:] = np.nan
slide_num = 0
# The following 2 lines initialize variables to compute AUC for train dataset.
total_pos, total_neg = 0, 0
#correct_pos, correct_neg = [0] * NUM_MODELS, [0] * NUM_MODELS
correct_pos = [0 for ii in range(NUM_MODELS)] # RanS 12.7.21
correct_neg = [0 for ii in range(NUM_MODELS)] # RanS 12.7.21

with torch.no_grad():
    for batch_idx, (data, target, time_list, last_batch, _, slide_file, patient) in enumerate(tqdm(inf_loader)):
        if new_slide:
            n_tiles = inf_loader.dataset.num_tiles[slide_num]  # RanS 1.7.21
            #scores_0, scores_1 = [np.zeros(0)] * NUM_MODELS, [np.zeros(0)] * NUM_MODELS
            #scores_0, scores_1 = [np.zeros(n_tiles)] * NUM_MODELS, [np.zeros(n_tiles)] * NUM_MODELS #RanS 1.7.21
            scores_0 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)] # RanS 12.7.21
            scores_1 = [np.zeros(n_tiles) for ii in range(NUM_MODELS)]  # RanS 12.7.21
            if args.save_features:
                #feature_arr = [np.zeros((n_tiles, 512))] * NUM_MODELS #RanS 1.7.21
                feature_arr = [np.zeros((n_tiles, 512)) for ii in range(NUM_MODELS)]  # RanS 1.7.21
            target_current = target
            slide_batch_num = 0
            new_slide = False

        data = data.squeeze(0)
        #print("data.shape: ", str(data.shape)) #temp RanS 22.7.21
        data, target = data.to(DEVICE), target.to(DEVICE)

        for index, model in enumerate(models):
            model.to(DEVICE)

            scores, features = model(data)

            scores = torch.nn.functional.softmax(scores, dim=1) #RanS 11.3.21

            #scores_0[index] = np.concatenate((scores_0[index], scores[:, 0].cpu().detach().numpy()))
            #scores_1[index] = np.concatenate((scores_1[index], scores[:, 1].cpu().detach().numpy()))
            scores_0[index][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data)] = scores[:, 0].cpu().detach().numpy() #RanS 1.7.21
            scores_1[index][slide_batch_num * tiles_per_iter: slide_batch_num * tiles_per_iter + len(data)] = scores[:, 1].cpu().detach().numpy() # RanS 1.7.21
            if args.save_features:
                feature_arr[index][slide_batch_num*tiles_per_iter: slide_batch_num*tiles_per_iter + len(data), :] = features.cpu().detach().numpy() #RanS 1.7.21

        slide_batch_num += 1

        if last_batch:
            new_slide = True

            all_targets.append(target.cpu().numpy()[0][0])
            if target == 1:
                total_pos += 1
            else:
                total_neg += 1

            for model_num in range(NUM_MODELS):
                current_slide_tile_scores = np.vstack((scores_0[model_num], scores_1[model_num]))

                predicted = current_slide_tile_scores.mean(1).argmax()
                #print('len(scores_1[model_num]):', len(scores_1[model_num])) #temp
                patch_scores[slide_num, model_num, :len(scores_1[model_num])] = scores_1[model_num]
                if args.save_features:
                    features_all[slide_num % NUM_SLIDES_SAVE, model_num, :len(feature_arr[model_num])] = feature_arr[model_num] #RanS 1.7.21
                all_scores[slide_num, model_num] = scores_1[model_num].mean()
                all_labels[slide_num, model_num] = predicted
                #all_slide_names[slide_num] = os.path.basename(slide_file[0])
                all_slide_names[slide_num] = slide_file[0] #RanS 5.5.21

                if target == 1 and predicted == 1:
                    correct_pos[model_num] += 1
                elif target == 0 and predicted == 0:
                    correct_neg[model_num] += 1

            slide_num += 1

            # RanS 6.7.21, save features every NUM_SLIDES_SAVE slides
            if args.save_features and slide_num % NUM_SLIDES_SAVE == 0:
                for model_num in range(NUM_MODELS):
                    feature_file_name = os.path.join(data_path, output_dir, '',
                                                     'Model_Epoch_' + str(args.from_epoch[model_num])
                                                     + '-Folds_' + str(args.folds) + '_' + str(
                                                         args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_' + str(slide_num) + '.data')
                    inference_data = [all_labels[slide_num-NUM_SLIDES_SAVE:slide_num, model_num],
                                      all_targets[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      all_scores[slide_num-NUM_SLIDES_SAVE:slide_num, model_num],
                                      np.squeeze(patch_scores[slide_num-NUM_SLIDES_SAVE:slide_num, model_num, :]),
                                      all_slide_names[slide_num-NUM_SLIDES_SAVE:slide_num],
                                      features_all]
                    with open(feature_file_name, 'wb') as filehandle:
                        pickle.dump(inference_data, filehandle)
                print('saved output for ', str(slide_num), ' slides')
                features_all = np.empty((NUM_SLIDES_SAVE, NUM_MODELS, args.num_tiles, 512))
                features_all[:] = np.nan

#save features for last slides
if args.save_features and slide_num % NUM_SLIDES_SAVE != 0:
    for model_num in range(NUM_MODELS):
        feature_file_name = os.path.join(data_path, output_dir, '',
                                         'Model_Epoch_' + str(args.from_epoch[model_num])
                                         + '-Folds_' + str(args.folds) + '_' + str(
                                             args.target) + '-Tiles_' + str(args.num_tiles) + '_features_slides_last.data')
        last_save = slide_num // NUM_SLIDES_SAVE * NUM_SLIDES_SAVE
        inference_data = [all_labels[last_save:slide_num, model_num],
                          all_targets[last_save:slide_num],
                          all_scores[last_save:slide_num, model_num],
                          np.squeeze(patch_scores[last_save:slide_num, model_num, :]),
                          all_slide_names[last_save:slide_num],
                          features_all[:slide_num-last_save]]
        with open(feature_file_name, 'wb') as filehandle:
            pickle.dump(inference_data, filehandle)
        print('saved output for ', str(slide_num), ' slides')


for model_num in range(NUM_MODELS):
    if different_experiments:
        output_dir = Output_Dirs[model_num]

    fpr, tpr, _ = roc_curve(all_targets, all_scores[:, model_num])

    # Save roc_curve to file:
    if not os.path.isdir(os.path.join(data_path, output_dir, '')):
        os.mkdir(os.path.join(data_path, output_dir, ''))

    file_name = os.path.join(data_path, output_dir, '', 'Model_Epoch_' + str(args.from_epoch[model_num])
                             + '-Folds_' + str(args.folds) + '_' + str(args.target) + '-Tiles_' + str(args.num_tiles) + '.data')
    inference_data = [fpr, tpr, all_labels[:, model_num], all_targets, all_scores[:, model_num],
                      total_pos, correct_pos[model_num], total_neg, correct_neg[model_num], len(inf_dset),
                      np.squeeze(patch_scores[:, model_num, :]), all_slide_names]

    with open(file_name, 'wb') as filehandle:
        pickle.dump(inference_data, filehandle)

    experiment = args.experiment[model_num] if different_experiments else args.experiment[0]
    print('For model from Experiment {} and Epoch {}: {} / {} correct classifications'
          .format(experiment,
                  args.from_epoch[model_num],
                  int(len(all_labels[:, model_num]) - np.abs(np.array(all_targets) - np.array(all_labels[:, model_num])).sum()),
                  len(all_labels[:, model_num])))
print('Done !')

# finished training, send email if possible
if os.path.isfile('mail_cfg.txt'):
    with open("mail_cfg.txt", "r") as f:
        text = f.readlines()
        receiver_email = text[0][:-1]
        password = text[1]

    port = 465  # For SSL
    sender_email = "gipmed.python@gmail.com"

    message = 'Subject: finished running experiment ' + str(args.experiment)

    # Create a secure SSL context
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
        print('email sent to ' + receiver_email)
