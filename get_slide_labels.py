import os
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Slide inspector')
parser.add_argument('--in_dir', default=r'C:\ran_data\Carmel_Slides_examples\folds_labels', type=str, help='input dir')
args = parser.parse_args()

in_dir = args.in_dir
batches = np.arange(1, 9, 1)

for batch in batches:

    meta_data_file = os.path.join(in_dir, 'slides_data_Carmel' + str(batch) + '.xlsx')
    meta_data_DF = pd.read_excel(meta_data_file)

    labels_data_file = os.path.join(in_dir, 'Carmel_annotations_04-07-2021.xlsx')
    label_data_DF = pd.read_excel(labels_data_file)
    data_field = ['PR status', 'ER status', 'Her2 status', 'TissueType', 'PatientIndex']

    #for ind, slide in enumerate(tqdm(meta_data_DF['patient barcode'])):
    for ind, slide in enumerate(meta_data_DF['patient barcode']):
        slide_tissue_id = slide.split('_')[0] + '/' + slide.split('_')[1]
        slide_block_id = slide.split('_')[2]
        slide_label_data = label_data_DF.loc[label_data_DF['TissueID'] == slide_tissue_id]

        if len(slide_label_data) == 0:
            #no matching tissue id
            print('1. Batch ' + str(batch) + ': could not find tissue_id ' + str(slide_tissue_id) + ' in annotations file!')

        elif len(slide_label_data) == 1:
            #one matching tissue id, make sure block id is empty or matching
            BlockID = slide_label_data['BlockID'].item()
            if np.isnan(BlockID) or str(BlockID) == slide_block_id:
                for field in data_field:
                    meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
            else:
                print('2. Batch ' + str(batch) + ': one matching tissue_id for ', str(slide_tissue_id), ', could not find matching blockID ' + str(slide_block_id) + ' in annotations file!')

        elif len(slide_label_data) > 1:
            slide_label_data_block = slide_label_data[slide_label_data['BlockID'] == int(slide_block_id)]
            if len(slide_label_data_block) == 0:
                print('3: Batch ' + str(batch) + ': ', str(len(slide_label_data)), ' matching tissue_id for ',
                      str(slide_tissue_id), ', could not find matching blockID ' + slide_block_id + ' in annotations file!')
            elif len(slide_label_data_block) > 1:
                print('4: Batch ' + str(batch) + ': ', str(len(slide_label_data)), ' matching tissue_id for ',
                      str(slide_tissue_id), ', found more than one matching blockID ' + slide_block_id + ' in annotations file!')
            else:
                for field in data_field:
                    meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]

    #replace empty cells with "missing data", and 0,1 with "Positive", "Negative"
    for field in data_field:
        meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)
        meta_data_DF[field] = meta_data_DF[field].replace(1, 'Positive', regex=True)
        meta_data_DF[field] = meta_data_DF[field].replace(0, 'Negative', regex=True)




    meta_data_DF.to_excel(os.path.join(in_dir, 'slides_data_CARMEL'+ str(batch)+ '_labeled.xlsx'))
print('finished')