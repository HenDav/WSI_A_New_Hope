import utils
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import PreActResNets
import nets_mil
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
from scipy.io import savemat


def get_AvgValues_from_Diff_Heatmaps(small_heatmaps):
    if len(small_heatmaps) != 1025:
        raise Exception('small_heatmaps list should contain 1025 heatmaps')
    all_average_maps = []
    for heatmap_idx in range(1, 1025):
        diff_heatmap = small_heatmaps[0] - small_heatmaps[heatmap_idx]
        diff = np.zeros((258, 258))
        diff[1:257, 1:257] = diff_heatmap
        diff_0_1 = np.where(diff != 0, 1, diff)
        # Now we'll go over each row and look for rows who's values are all zeroes and that they are adjacent to rows which
        # have non zero values - these rows mark the borders of the areas.
        # Then we'll do the same for columns

        beginning_row, ending_row, beginning_col, ending_col = [], [], [], []
        for row_idx in range(1, 258):
            if diff_0_1[row_idx, :].sum() > 0 and diff_0_1[row_idx - 1, :].sum() == 0:
                beginning_row.append(row_idx)
            elif diff_0_1[row_idx, :].sum() == 0 and diff_0_1[row_idx - 1, :].sum() > 0:
                ending_row.append(row_idx)
            else:  # Both rows are equal to 0
                continue
        for col_idx in range(1, 258):
            if diff_0_1[:, col_idx].sum() > 0 and diff_0_1[:, col_idx - 1].sum() == 0:
                beginning_col.append(col_idx)
            elif diff_0_1[:, col_idx].sum() == 0 and diff_0_1[:, col_idx - 1].sum() > 0:
                ending_col.append(col_idx)
            else:  # Both rows are equal to 0
                continue

        average_values = np.zeros((8, 8))  # We have an 8X8 array of non zero values
        # we need to calculate the average of each block which has non zero values
        for row_idx in range(8):
            for col_idx in range(8):
                average_values[row_idx, col_idx] = diff[beginning_row[row_idx]:ending_row[row_idx], beginning_col[col_idx]:ending_col[col_idx]].mean()

        all_average_maps.append(average_values)

    return all_average_maps


def connect_all_average_maps(average_maps):
    # At first I'll create the final heatmap by using the first small heatmap and inserting 31 zeros between each row and column
    final_heatmap = np.zeros((39, 39))
    final_heatmap[:8, :8] = average_maps[0]
    for add_row_idx in range(1, 32):
        final_heatmap = np.insert(final_heatmap, range(add_row_idx, 8 * add_row_idx, add_row_idx), 0, axis=0)
    for add_col_idx in range(1, 32):
        final_heatmap = np.insert(final_heatmap, range(add_col_idx, 8 * add_col_idx, add_col_idx), 0, axis=1)

    # Now I'll add the other small_heatmaps data.
    for idx in range(1, 1024):
        map = average_maps[idx]




        for add_row_idx in range(1, 32):
            map = np.insert(map, range(add_row_idx, 8 * add_row_idx, add_row_idx), 0, axis=0)
        for add_col_idx in range(1, 32):
            map = np.insert(map, range(add_col_idx, 8 * add_col_idx, add_col_idx), 0, axis=1)







def get_cutout_scores(tile, basic_model, MIL_model):
    basic_model.to(DEVICE)
    MIL_model.to(DEVICE)
    embedded_squares_in_tiles = Embed_Square(tile)
    if tile.shape[1] == 256:
        scores_REG_image, weights_MIL_image = np.zeros((32, 32)), np.zeros((32, 32))
    else:
        all_small_heat_maps = []

    with torch.no_grad():
        for idx, tile in enumerate(tqdm(embedded_squares_in_tiles)):
            tile = tile.to(DEVICE)
            basic_model_outputs = basic_model(tile)
            if tile.shape[2] == 2048:
                all_small_heat_maps.append(basic_model_outputs['Small Heat Map'].squeeze().numpy())
                if idx == 1:
                    pass

                continue
                '''# Save to file:
                if not os.path.isdir('/Users/wasserman/Developer/WSI_MIL/Heatmaps/CutOut/' + slide.split('.')[0]):
                    Path('/Users/wasserman/Developer/WSI_MIL/Heatmaps/CutOut/' + slide.split('.')[0]).mkdir(parents=True, exist_ok=True)

                pd.DataFrame(small_heatmap).to_excel('/Users/wasserman/Developer/WSI_MIL/Heatmaps/CutOut/' + slide.split('.')[0] + str(tile_num) + '/heatmap_tile_' + str(idx) + '.xlsx')'''

            _, _, weight_before_sftmx = mil_model(x=None, H=basic_model_outputs['Features'])
            score = torch.nn.functional.softmax(basic_model_outputs['Scores'], dim=1)[0][1].item()

            scores_REG_image[idx // 32, idx % 32] = score
            weights_MIL_image[idx // 32, idx % 32] = weight_before_sftmx

        scores_DF = pd.DataFrame(scores_REG_image).to_excel('/Users/wasserman/Developer/WSI_MIL/Heatmaps/cutout_scores.xlsx')
        weights_DF = pd.DataFrame(weights_MIL_image).to_excel('/Users/wasserman/Developer/WSI_MIL/Heatmaps/cutout_weights.xlsx')

    return {'CutOut Scores': scores_REG_image,
            'CutOut Weights': weights_MIL_image}


def get_tile_movements(initial_location: dict = None,
                       slidename: str = None,
                       model = None):

    one_slide_dset = datasets.One_Full_Slide_Inference_Dataset(DataSet='TCGA',
                                                               tile_size=2048,
                                                               slidename=slidename)

    delta_pixel = one_slide_dset.delta_pixel
    # Create locations:
    locations = []
    for row in range(0, 8):
        Row = initial_location['Row'] + row * delta_pixel
        for col in range(0, 8):
            Col = initial_location['Col'] + col * delta_pixel
            locations.append((Row, Col))

    slide_tile_data = one_slide_dset.__getitem__(location=locations)

    # We can pass the tiles through the model:
    model.eval()
    model.to(DEVICE)
    model.is_HeatMap = True
    score_heatmaps = []
    tile_scores = []

    for idx, tile in enumerate(tqdm(slide_tile_data['Data'])):
        model_output = model(tile)
        score_heatmaps.append(model_output['Small Heat Map'])
        tile_scores.append(model_output['Scores'])

        pd.DataFrame(model_output['Small Heat Map'].squeeze().numpy()).to_excel(
            '/Users/wasserman/Developer/WSI_MIL/Heatmaps/score_heatmap_tile_ER_Pos_2048_1_pixel_movement_' + str(idx) + '.xlsx')

    return {'Score Heatmaps': score_heatmaps,
            'Tile Scores': tile_scores}

# Data type definition:
DATA_TYPE = 'Features'
target = 'ER'
test_fold = 1

model_locations = {'ABCTB_TCGA': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/ran_293/model_data_Epoch_1000.pt',
                   'Carmel': r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/Carmel_338/model_data_Epoch_1000.pt'}

if target == 'ER':
    if test_fold == 1:
        Dataset_name = r'FEATURES: Exp_293-ER-TestFold_1'
        train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Fold_1/Train'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/Fold_1/Test'
        #basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/ran_293/model_data_Epoch_1000.pt'
        traind_model = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/features/338 - freezed last layer/model_data_Epoch_500.pt'
    elif test_fold == 2:
        Dataset_name = r'FEATURES: Exp_299-ER-TestFold_2'
        train_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Train'
        test_data_dir = r'/Users/wasserman/Developer/WSI_MIL/All Data/Features/ER/ran_299-Fold_2/Test'



need_to_find_slides_to_work_on = False
# Get data:
if need_to_find_slides_to_work_on:
    test_dset = datasets.Features_MILdataset(data_location=test_data_dir,
                                             is_per_patient=False,
                                             bag_size=500,
                                             target=target,
                                             is_train=False,
                                             test_fold=test_fold)


    test_loader = DataLoader(test_dset, batch_size=100, shuffle=False, num_workers=0, pin_memory=True)

    all_positive_slide_names = []
    all_positive_scores = []
    all_positive_tile_scores = []

    all_negative_slide_names = []
    all_negative_scores = []
    all_negative_tile_scores = []

    for batch_idx, minibatch in enumerate(tqdm(test_loader)):
        slide_scores = minibatch['scores']
        tile_scores = minibatch['tile scores']
        slide_names = np.array(minibatch['slide name'])
        labels = minibatch['labels']
        targets = minibatch['targets']

        all_positive_slide_names.extend(slide_names[targets == 1])
        all_positive_scores.extend(slide_scores[targets == 1])
        all_positive_tile_scores.extend(tile_scores[targets == 1])

        all_negative_slide_names.extend(slide_names[targets == 0])
        all_negative_scores.extend(slide_scores[targets == 0])
        all_negative_tile_scores.extend(tile_scores[targets == 0])

    all_positive_slide_names, all_negative_slide_names = np.array(all_positive_slide_names), np.array(all_negative_slide_names)
    all_positive_scores, all_negative_scores = np.array(all_positive_scores), np.array(all_negative_scores)
    sorted_scores, sorted_scores_negative = list(np.sort(all_positive_scores)), list(np.sort(all_negative_scores))
    sorted_scores.reverse()

    highest_score_slides = []
    highest_scores = []
    tile_scores_for_slides = []
    # Sorting the positive slides
    for idx, score in enumerate(sorted_scores):
        highest_scores.append(score)
        idx = np.where(all_positive_scores == score)[0][0]
        highest_score_slides.append(all_positive_slide_names[idx])
        tile_scores_for_slides.append(all_positive_tile_scores[idx])

    slide_name = highest_score_slides[0]
    slide_08 = highest_score_slides[530]
    slide_07 = highest_score_slides[645]

    # Sorting the NEGATIVE slides:
    lowest_score_slides = []
    lowest_scores = []
    lowest_tile_scores_for_slides = []

    for idx, score in enumerate(sorted_scores_negative):
        lowest_scores.append(score)
        idx = np.where(all_negative_scores == score)[0][0]
        lowest_score_slides.append(all_negative_slide_names[idx])
        lowest_tile_scores_for_slides.append(all_negative_tile_scores[idx])

    # Picking some negative slides:
    slide_name_NEGative = lowest_score_slides[7]
    slide_score = lowest_scores[7]
else:
    slide_name_POS = 'TCGA-A8-A099-01Z-00-DX1.B19C28B5-FEBC-49B4-A60E-E6B85BB00DD7.svs'
    slide_name_NEG = 'TCGA-AR-A1AI-01Z-00-DX1.5EF2A589-4284-45CF-BF0C-169E3A85530C.svs'
    slide_name_CARMEL_model_355 = ['18-2728_1_1_a.mrxs', '18-2728_1_1_e.mrxs', '19-5229_2_1_e.mrxs']
    slide_name_CARMEL_model_338 = ['18-7361_1_7_b.mrxs', '18-3507_1_1_k.mrxs', '19-6074_1_1_e.mrxs', '19-9915_1_1_e.mrxs']

    slide_locations = {}
    slide_locations['18-2728_1_1_a.mrxs'] = [{'TOP': 144000, 'LEFT': 24000},
                                             {'TOP': 151000, 'LEFT': 24000},
                                             {'TOP': 158000, 'LEFT': 25000},
                                             {'TOP': 163500, 'LEFT': 40500}]
    slide_locations['18-2728_1_1_e.mrxs'] = [{'TOP': 148000, 'LEFT': 21000},
                                             {'TOP': 155000, 'LEFT': 22500},
                                             {'TOP': 162000, 'LEFT': 22500},
                                             {'TOP': 167000, 'LEFT': 37000}]
    slide_locations['19-5229_2_1_e.mrxs'] = [{'TOP': 148000, 'LEFT': 2500},
                                             {'TOP': 162000, 'LEFT': 4000},
                                             {'TOP': 146000, 'LEFT': 16000},
                                             {'TOP': 154000, 'LEFT': 17500},
                                             {'TOP': 134000, 'LEFT': 26500},
                                             {'TOP': 181000, 'LEFT': 31000},
                                             {'TOP': 151000, 'LEFT': 47000},
                                             {'TOP': 167000, 'LEFT': 46000},
                                             {'TOP': 175000, 'LEFT': 50000}]
    slide_locations['18-7361_1_7_b.mrxs'] = [{'TOP': 61000, 'LEFT': 39000},
                                             {'TOP': 118000, 'LEFT': 54000},
                                             {'TOP': 68000, 'LEFT': 7500},
                                             {'TOP': 64000, 'LEFT': 12000},
                                             {'TOP': 114000, 'LEFT': 37000}]
    slide_locations['18-3507_1_1_k.mrxs'] = [{'TOP': 82000, 'LEFT': 28000},
                                             {'TOP': 90000, 'LEFT': 31000},
                                             {'TOP': 82000, 'LEFT': 45500},
                                             {'TOP': 90000, 'LEFT': 47000},
                                             {'TOP': 98000, 'LEFT': 47000},
                                             {'TOP': 98000, 'LEFT': 57000},
                                             {'TOP': 111000, 'LEFT': 40000}]
    slide_locations['19-6074_1_1_e.mrxs'] = [{'TOP': 42000, 'LEFT': 42000},
                                             {'TOP': 45000, 'LEFT': 50000},
                                             {'TOP': 49000, 'LEFT': 58000},
                                             {'TOP': 52000, 'LEFT': 45000},
                                             {'TOP': 56000, 'LEFT': 53000},
                                             {'TOP': 69000, 'LEFT': 44000}]
    slide_locations['TCGA-A8-A099-01Z-00-DX1.B19C28B5-FEBC-49B4-A60E-E6B85BB00DD7.svs'] = [{'TOP': 20000, 'LEFT': 20000},
                                                                                           {'TOP': 20000, 'LEFT': 8000}]
    slide_locations['TCGA-AR-A1AI-01Z-00-DX1.5EF2A589-4284-45CF-BF0C-169E3A85530C.svs'] = [{'TOP': 33000, 'LEFT': 49000}]

# Slide categorization to models:
Model_ABCTB_TCGA_ER_Fold_1 = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/ran_293/model_data_Epoch_1000.pt'
Model_CAT_ER_Fold_1 = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'
Model_CARMEL_ER_Fold_5 = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CARMEL_338-TF_5/model_data_Epoch_1000.pt'

Slide_to_Model = {'TCGA-A8-A099-01Z-00-DX1.B19C28B5-FEBC-49B4-A60E-E6B85BB00DD7.svs': Model_ABCTB_TCGA_ER_Fold_1,
                  'TCGA-AR-A1AI-01Z-00-DX1.5EF2A589-4284-45CF-BF0C-169E3A85530C.svs': Model_ABCTB_TCGA_ER_Fold_1,
                  '18-2728_1_1_a.mrxs': Model_CAT_ER_Fold_1,
                  '18-2728_1_1_e.mrxs': Model_CAT_ER_Fold_1,
                  '19-5229_2_1_e.mrxs': Model_CAT_ER_Fold_1,
                  '18-3507_1_1_k.mrxs': Model_CARMEL_ER_Fold_5,
                  '18-7361_1_7_b.mrxs': Model_CARMEL_ER_Fold_5,
                  '19-6074_1_1_e.mrxs': Model_CARMEL_ER_Fold_5,
                  '19-9915_1_1_e.mrxs': Model_CARMEL_ER_Fold_5}

# Second part: computing tile scores to find a few tiles with high scores
inf_dset = datasets.Full_Slide_Inference_Dataset(DataSet='TCGA',
                                                 tile_size=256,
                                                 tiles_per_iter=1,
                                                 target_kind='ER',
                                                 folds=[1, 5],
                                                 desired_slide_magnification=10,
                                                 num_background_tiles=0)

inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

tile_locations = []
tile_scores = []
tile_data = []
original_tile_data = []
is_tissue = []
small_heat_maps_all = []
large_heat_maps_all = []
features_all = []
large_image_for_MIL_weights_all = []

data_256_all = []
original_256_all = []
small_heat_maps_256_all = []
large_heat_maps_256_all = []
features_256_all = []
tile_scores_256 = []

# Choose slide and tile location:
slide = slide_name_POS
tile_num = 0

location = (slide_locations[slide][tile_num]['TOP'], slide_locations[slide][tile_num]['LEFT'])

# Load Model:
'''
if inf_dset.DataSet == 'TCGA':
    basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/ran_293/model_data_Epoch_1000.pt'
elif inf_dset.DataSet == 'CARMEL':
    basic_model_location = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/runs/Ran_models/ER/CAT_355_TF_1/model_data_Epoch_1000.pt'
'''
basic_model_location = Slide_to_Model[slide]
model = PreActResNets.PreActResNet50_Ron()
model_data_loaded = torch.load(basic_model_location, map_location='cpu')
model.load_state_dict(model_data_loaded['model_state_dict'])
model.eval()

DEVICE = utils.device_gpu_cpu()

model.to(DEVICE)

with torch.no_grad():
    #for batch_idx, minibatch in enumerate(tqdm(inf_loader)):
    for batch_idx in tqdm(range(len(inf_dset))):
        minibatch = inf_dset.__getitem__(batch_idx, location=[location], tile_size=2048)
        #slide_filename = minibatch['Slide Filename'][0].split('/')[-1]
        slide_filename = minibatch['Slide Filename'].split('/')[-1]
        if slide_filename != slide:
            continue

        get_64_tile_movement_heatmaps = False
        if get_64_tile_movement_heatmaps:
            desired_idx = batch_idx
            # Computing 64 tiles, 1 pixel movements:
            """ 
            Negative Slide: TCGA-AR-A1AI-01Z-00-DX1.5EF2A589-4284-45CF-BF0C-169E3A85530C.svs, Location: initial_location={'Row': 33000, 'Col': 49000}
            Positive Slide: TCGA-A8-A099-01Z-00-DX1.B19C28B5-FEBC-49B4-A60E-E6B85BB00DD7.svs, Location: 
            Positive Slide:  
            """
            initial_location_Negative = {'Row': 33000, 'Col': 49000}
            initial_location_Positive = {'Row': 20000, 'Col': 20000}
            initial_location_Positive_1 = {'Row': 20000, 'Col': 8000}
            heatmaps = get_tile_movements(initial_location=initial_location_Negative,
                                          slidename=slide_filename,
                                          model=model)


        data = minibatch['Data']
        if data.shape[3] == 1024:
            pix = 0
            data_256 = data[:, :, :, :256, pix:256 + pix]

        label = minibatch['Label']
        is_last_batch = minibatch['Is Last Batch']
        initial_num_tiles = minibatch['Initial Num Tiles']
        equivalent_grid = minibatch['Equivalent Grid']
        equivalent_grid_size = minibatch['Equivalent Grid Size']
        level_0_locations = minibatch['Level 0 Locations']
        original_tiles = minibatch['Original Data']
        is_tissue_tile = minibatch['Is Tissue Tiles']

        tile_data.extend(data)
        #original_tile_data.extend(original_tiles)
        original_tile_data.extend([original_tiles])
        is_tissue.extend(is_tissue_tile)
        #is_tissue.extend(is_tissue_tile[0])
        data = data.to(DEVICE)
        model.is_HeatMap = True

        model_output = model(data)
        ###########
        # Model output extraction:
        data_4_gil = model_output['Data 4 Gil']
        scores = model_output['Scores']
        heatmap = model_output['Large Heat Map']
        small_heatmap = model_output['Small Heat Map']
        large_image_for_MIL_weights_all.append(model_output['Large Image for MIL Weights Without Averaging Sliding Window'])

        if (small_heatmap.mean().item() - (scores[0][1] - scores[0][0]).item()) / small_heatmap.mean().item() >= 1e-5:
            raise Exception('Heatmap mean is not equal to score difference')

        scores = torch.nn.functional.softmax(scores, dim=1)
        tile_scores.append(scores[:, 1].item())
        tile_locations.extend(level_0_locations)

        small_heat_maps_all.append(small_heatmap)
        large_heat_maps_all.append(heatmap)
        features_all.append(model_output['Features'])

        '''
        # Save data for gil:
        data_4_gil['original_tile'] = original_tile_data[data_idx].numpy()
        savemat('/Users/wasserman/Developer/WSI_MIL/data4gil.mat', data_4_gil)
        ######################
        '''


        # If we use a small tile than we'll calculate the top left 256 pixels corner and compare.
        if data.shape[3] == 1024:
            data_256 = data_256.to(DEVICE)
            model_output_256 = model(data_256)

            original_256 = original_tiles[:, :, :256, :256]

            data_256_all.extend(data_256)
            original_256_all.extend(original_256)
            small_heat_maps_256_all.append(model_output_256['Small Heat Map'])
            large_heat_maps_256_all.append(model_output_256['Large Heat Map'])
            features_256_all.append(model_output_256['Features'])

            scores = model_output_256['Scores']
            scores = torch.nn.functional.softmax(scores, dim=1)
            tile_scores_256.append(scores[:, 1].item())

        if batch_idx == 0:
            break

extract_MIL_weights = True
if extract_MIL_weights:
    # Computing tile weight using MIL:
    mil_model = eval('nets_mil.MIL_Feature_Attention_MultiBag()')
    mil_model_data_loaded = torch.load(os.path.join('/Users/wasserman/Developer/WSI_MIL/runs/Exp_347-ER-TestFold_1/Model_CheckPoints/',
                                                'model_data_Epoch_' + str(500) + '.pt'), map_location='cpu')
    mil_model.load_state_dict(mil_model_data_loaded['model_state_dict'])
    mil_model.to(DEVICE)
    mil_model.infer = True
    mil_model.eval()

    weights_all = []
    mil_weights_image_all = []
    outputs_all = []

    if data.shape[3] == 1024:
        weights_256_all = []
        outputs_256_all = []

    with torch.no_grad():
        for batch_idx, features in enumerate(features_all):
            features.to(DEVICE)
            outputs, weights_after_sfmx, weights_befor_sfmx = mil_model(x=None, H=features)

            weights_all.append(weights_befor_sfmx.item())
            outputs_all.append(outputs)

            # Compute MIL weights per Pixel:
            image_512_features = large_image_for_MIL_weights_all[batch_idx]
            _, _, size, _ = image_512_features.shape
            features_4_full_large_image = np.transpose(np.reshape(image_512_features, (512, size ** 2)), (1, 0))
            _, _, weights_before_sfmx_large_image = mil_model(x=None, H=features_4_full_large_image)
            weights_MIL_image_shaped = np.reshape(weights_before_sfmx_large_image, (size, size)).numpy()
            mil_weights_image_all.append(weights_MIL_image_shaped)

        if data.shape[3] == 1024:
            for batch_idx, features_256 in enumerate(features_256_all):
                features_256.to(DEVICE)
                outputs, weights_after_sfmx, weights_befor_sfmx = mil_model(x=None, H=features_256)

                weights_256_all.append(weights_befor_sfmx.item())
                outputs_256_all.append(outputs)

            # Matching small heatmap imcoloages of 1024 and 256 tiles:
            heatmap_1024 = small_heat_maps_all[0][:, :, :32, :32]
            heatmap_256 = small_heat_maps_256_all[0]

embed_cutout_squares = True
if embed_cutout_squares:
    # Computing per pixel scores using CutOut:
    Embed_Square = utils.EmbedSquare(minibatch_size=1, color='Black')
    cutout_scores_weights = get_cutout_scores(tile_data[0], basic_model=model, MIL_model=mil_model)


'''
# checking the images:
fig = plt.figure()
for idx, tile in enumerate(embedded_squares):
    fig.add_subplot(4, 8, idx + 1)
    plt.imshow(np.transpose(tile.squeeze(0).numpy(), (1, 2, 0)))
    plt.axis('off')
    if idx == 31:
        break
plt.show()

'''

# Saving tile image:
matplotlib.image.imsave('/Users/wasserman/Developer/WSI_MIL/Heatmaps/' + slide.split('.')[0] + '_tile_' + str(tile_num) + '.png', np.transpose(original_tile_data[0].numpy(), (1, 2, 0)))
pd.DataFrame(small_heat_maps_all[0].squeeze().numpy()).to_excel('/Users/wasserman/Developer/WSI_MIL/Heatmaps/heatmap_' + slide.split('.')[0] + '_tile_' + str(tile_num) + '.xlsx')
pd.DataFrame(mil_weights_image_all[0]).to_excel('/Users/wasserman/Developer/WSI_MIL/Heatmaps/MIL_heatmap_' + slide.split('.')[0] + '_tile_' + str(tile_num) + '.xlsx')
# Saving both small Heatmaps to excel files:
'''
if data.shape[3] == 1024:
    pd.DataFrame(heatmap_256.squeeze().numpy()).to_excel('/Users/wasserman/Developer/WSI_MIL/256_' +str(pix) +'_pix_right.xlsx')
    pd.DataFrame(heatmap_1024.squeeze().numpy()).to_excel('/Users/wasserman/Developer/WSI_MIL/1024.xlsx')
'''
# Showing images
f, axarr = plt.subplots(figsize=(11, 4), nrows=1, ncols=3)
idx = 0
axarr[0].imshow(np.transpose(small_heat_maps_all[idx].squeeze(0).numpy(), (1, 2, 0)), cmap='gray')
axarr[1].imshow(np.transpose(original_tile_data[idx].numpy(), (1, 2, 0)))
axarr[2].imshow(mil_weights_image_all[idx], cmap='gray')

#f.suptitle('tile score = {:.4f}, tile MIL weight = {:.4f}'.format(tile_scores[idx], weights_all[idx]))
f.suptitle('tile score = {:.4f}'.format(tile_scores[idx]))
axarr[0].set_title('Score Heatmap')
axarr[1].set_title('Original Tile')
axarr[2].set_title('MIL Weights Heatmap')

if data.shape[3] == 1024:
    f, axarr = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)
    idx = 0
    axarr[0].imshow(np.transpose(small_heat_maps_256_all[idx].squeeze(0).numpy(), (1, 2, 0)), cmap='gray')
    axarr[1].imshow(np.transpose(original_256_all[idx].numpy(), (1, 2, 0)))
    axarr[2].imshow(np.transpose(large_heat_maps_256_all[idx].squeeze(0).numpy(), (1, 2, 0)), cmap='gray')
    axarr[1].set_title('tile score = {:.4f}, tile MIL weight = {:.4f}'.format(tile_scores_256[idx], weights_256_all[idx]))
print('Done')

