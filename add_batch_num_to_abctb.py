from glob import glob
import pandas as pd
import os
from tqdm import tqdm

original_path = r'/Volumes/McKinley/ABCTB'
slide_data_path = r'/Users/wasserman/Developer/WSI_MIL/Data from gipdeep/'
slide_data_filename = r'slides_data_ABCTB.xlsx'
new_slide_data_filename = r'slides_data_ABCTB_with_batch_num.xlsx'
batch_Dict = {'1': 'Batch1',
              '2': 'Batch2',
              '3': 'Batch3',
              '4': 'Batch4',
              '5': 'Batch5',
              '6': 'Batch6',
              '7': 'Batch7',
              '8': 'Batch8',
              '9': 'Batch9',
              '10': 'Batch10',
              '11': 'Batch11',
              '12': 'Batch12',
              '13': 'Batch13'
              }

slide_data_DF = pd.read_excel(os.path.join(slide_data_path, slide_data_filename))
num_slides = 0
for batch in tqdm(batch_Dict):
    files = glob(os.path.join(original_path, batch_Dict[batch], '*.ndpi'))
    for file in files:
        row_index = slide_data_DF[slide_data_DF['file'] == file.split('/')[-1]].index
        slide_data_DF.loc[row_index, 'Batch'] = batch_Dict[batch]
        num_slides += 1

slide_data_DF.to_excel(os.path.join(slide_data_path, new_slide_data_filename))
print('Processed {} slides'.format(num_slides))





