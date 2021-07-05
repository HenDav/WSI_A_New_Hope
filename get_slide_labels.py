import os
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Slide inspector')
#parser.add_argument('--in_dir', default=r'\\gipnetappa\public\sgils\BCF scans\Carmel Slides\Batch_1', type=str, help='input dir')
parser.add_argument('--in_dir', default=r'C:\ran_data\Carmel_Slides_examples\folds_labels', type=str, help='input dir')
args = parser.parse_args()

in_dir = args.in_dir

meta_data_file = os.path.join(in_dir, 'slides_data.xlsx')
meta_data_DF = pd.read_excel(meta_data_file)

labels_data_file = os.path.join(os.path.dirname(in_dir), 'Carmel_annotations_6_1_2021.xlsx')
label_data_DF = pd.read_excel(labels_data_file)
data_field = ['PR status', 'ER status', 'Her2 status', 'TissueType', 'PatientIndex']

for ind, slide in enumerate(tqdm(meta_data_DF['patient barcode'])):
    slide_tissue_id = slide.split('_')[0] + '/' + slide.split('_')[1]
    slide_label_data = label_data_DF.loc[label_data_DF['TissueID'] == slide_tissue_id]

    if len(slide_label_data) == 0:
        print('could not find tissue_id ' + str(slide_tissue_id) + ' in annotations file!')
    elif len(slide_label_data) == 1:
        for field in data_field:
            meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]

    elif len(slide_label_data) > 1:
        #edit slide_label_data block_id into list of blocks
        ind = 0
        block_id_lists = [[] for i in range(len(slide_label_data))]
        for block_id in slide_label_data['BlockID'].values:
            range_split = block_id.split('-')
            if len(range_split) > 2:
                print('wrong format for block ' + block_id + ' in slide ' + slide)
                continue
            elif len(range_split) == 2:
                block_id_lists[ind] = [str(block) for block in range(int(range_split[0]), int(range_split[1])+1)] #handle range, should be str??
            elif len(range_split) == 1:
                block_id_lists[ind] = block_id.replace(' ', '').split(',') #handle comma/single value
            ind += 1

        # Search for matches with the appropriate block
        found = False
        slide_block_id = slide.split('_')[2]
        for ii, block_id_list in enumerate(block_id_lists):
            if slide_block_id in block_id_list:
                found = True
                break #assuming there is only one, taking the first

        if found:
            for field in data_field:
                meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[ii]
        else:
            #unknown block, take only consensus labels
            for field in data_field:
                if np.all(slide_label_data[field].values == slide_label_data[field].values[0]):
                    meta_data_DF.loc[meta_data_DF['patient barcode'] == slide, field] = slide_label_data[field].values[0]
                else:
                    print('could not determine ' + field + ' for slide ' + slide)

#replace empty cells with "missing data"
for field in data_field:
    meta_data_DF[field] = meta_data_DF[field].replace(np.nan, 'Missing Data', regex=True)

meta_data_DF.to_excel(os.path.join(in_dir, 'slides_data_labeled.xlsx'))
print('finished')