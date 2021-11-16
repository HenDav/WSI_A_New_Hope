import datasets
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from scipy.io import savemat, loadmat

compute_highest_and_lowest = True
if compute_highest_and_lowest:
    data_dir_dict = utils.get_RegModel_Features_location_dict(train_DataSet='CAT with Location', target='ResNet34', test_fold=1)
    original_dset = datasets.Features_MILdataset(dataset='CAT',
                                                 data_location=data_dir_dict['TestSet Location'],
                                                 is_per_patient=False,
                                                 is_all_tiles=True,
                                                 target='ER',
                                                 is_train=False,
                                                 carmel_only=True,
                                                 test_fold=1)


    data_loader = DataLoader(original_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    all_slide_names = []
    all_tile_targets = []
    all_tile_scores = []
    all_tile_locations = []
    all_tile_features = []

    for batch_idx, minibatch in enumerate(tqdm(data_loader)):
        slide_scores = minibatch['scores']
        tile_scores = minibatch['tile scores']
        slide_names = np.array(minibatch['slide name'])
        labels = minibatch['labels']
        targets = minibatch['targets']
        tile_locations = minibatch['tile locations']
        tile_features = minibatch['features'].squeeze(0)


        all_slide_names.extend([slide_names.item()] * tile_scores.size(1))
        all_tile_scores.extend(tile_scores.tolist()[0])
        all_tile_locations.extend(tile_locations.tolist()[0])
        all_tile_targets.extend([targets.item()] * tile_scores.size(1))
        all_tile_features.extend(tile_features.tolist())


    # Converting to numpy arrays:
    all_slide_names = np.array(all_slide_names)
    all_tile_locations = np.array(all_tile_locations)
    all_tile_scores = np.array(all_tile_scores)
    all_tile_targets = np.array(all_tile_targets)
    all_tile_features = np.array(all_tile_features)

    # Sorting by tile scores:
    sort_idx = np.argsort(all_tile_scores)
    sorted_slide_names = all_slide_names[sort_idx]
    sorted_tile_scores = all_tile_scores[sort_idx]
    sorted_tile_locations = all_tile_locations[sort_idx]
    sorted_tile_targets = all_tile_targets[sort_idx]
    sorted_tile_features = all_tile_features[sort_idx]

    # Picking top 10% and bottom 3%:
    num_top_10_percent = int(90 * len(all_slide_names)/100)
    num_bottom_3_percent = int(3 * len(all_slide_names)/100)

    highest_10_percent_slide_names = sorted_slide_names[num_top_10_percent:]
    highest_10_percent_tile_scores = sorted_tile_scores[num_top_10_percent:]
    highest_10_percent_tile_locations = sorted_tile_locations[num_top_10_percent:]
    highest_10_percent_tile_targets = sorted_tile_targets[num_top_10_percent:]
    highest_10_percent_tile_features = sorted_tile_features[num_top_10_percent:]

    lowest_3_percent_slide_names = sorted_slide_names[:num_bottom_3_percent]
    lowest_3_percent_tile_scores = sorted_tile_scores[:num_bottom_3_percent]
    lowest_3_percent_tile_locations = sorted_tile_locations[:num_bottom_3_percent]
    lowest_3_percent_tile_targets = sorted_tile_targets[:num_bottom_3_percent]
    lowest_3_percent_tile_features = sorted_tile_features[:num_bottom_3_percent]

    # Saving the data:
    save_as_multiple_files = False
    if save_as_multiple_files:
        num_tiles_per_file = 8000

        start_marker = 0
        end_marker = start_marker + num_tiles_per_file

        for file_num in range(round(len(highest_10_percent_slide_names) / num_tiles_per_file)):
            if file_num == round(len(highest_10_percent_slide_names) / num_tiles_per_file) - 1:
                end_marker = -1

            highest = {'Slide_Names': highest_10_percent_slide_names[start_marker : end_marker],
                       'Tile_Scores': highest_10_percent_tile_scores[start_marker : end_marker],
                       'Tile_Locations': highest_10_percent_tile_locations[start_marker : end_marker],
                       'Tile_Targets': highest_10_percent_tile_targets[start_marker : end_marker],
                       'Features': highest_10_percent_tile_features[start_marker : end_marker]
                       }

            start_marker = end_marker
            end_marker += num_tiles_per_file
            savemat('/Users/wasserman/Developer/WSI_MIL/Data For Gil/highest_10_file_' + str(file_num) + '.mat', highest)


        start_marker = 0
        end_marker = start_marker + num_tiles_per_file
        for file_num in range(round(len(lowest_3_percent_slide_names) / num_tiles_per_file)):
            if file_num == round(len(lowest_3_percent_slide_names) / num_tiles_per_file) - 1:
                end_marker = -1

            lowest = {'Slide_Names': lowest_3_percent_slide_names[start_marker : end_marker],
                      'Tile_Scores': lowest_3_percent_tile_scores[start_marker : end_marker],
                      'Tile_Locations': lowest_3_percent_tile_locations[start_marker : end_marker],
                      'Tile_Targets': lowest_3_percent_tile_targets[start_marker : end_marker],
                      'Features': lowest_3_percent_tile_features[start_marker : end_marker]
                      }

            start_marker = end_marker
            end_marker += num_tiles_per_file
            savemat('/Users/wasserman/Developer/WSI_MIL/Data For Gil/lowest_3_file_' + str(file_num) + '.mat', lowest)

    else:
        highest = {'Slide_Names': highest_10_percent_slide_names,
                   'Tile_Scores': highest_10_percent_tile_scores,
                   'Tile_Locations': highest_10_percent_tile_locations,
                   'Tile_Targets': highest_10_percent_tile_targets,
                   'Features': highest_10_percent_tile_features
                   }

        lowest = {'Slide_Names': lowest_3_percent_slide_names,
                  'Tile_Scores': lowest_3_percent_tile_scores,
                  'Tile_Locations': lowest_3_percent_tile_locations,
                  'Tile_Targets': lowest_3_percent_tile_targets,
                  'Features': lowest_3_percent_tile_features
                  }

        savemat('/Users/wasserman/Developer/WSI_MIL/Data For Gil/highest_10.mat', highest)
        savemat('/Users/wasserman/Developer/WSI_MIL/Data For Gil/lowest_3.mat', lowest)

else:
    highest = loadmat(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/highest_10.mat')
    lowest = loadmat(r'/Users/wasserman/Developer/WSI_MIL/Data For Gil/lowest_3.mat')


print('Done')