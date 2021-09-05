import pandas as pd
import os
import numpy as np

dn = r'C:\ran_data\Carmel_Slides_examples\boxes58_10b2 fix'
dat_file = r'the_correct_boxes_RanS 310821.xlsx'
slides_data_file9 = r'Slides_data_CARMEL9.xlsx'
slides_data_file10 = r'Slides_data_CARMEL10.xlsx'

s9 = pd.read_excel(os.path.join(dn, slides_data_file9))
s10 = pd.read_excel(os.path.join(dn, slides_data_file10))
dat = pd.read_excel(os.path.join(dn, dat_file))
currently_in_9 = []
currently_in_10 = []
for row in dat.iterrows():
    print(row)
    #row[1]['currently_in'] = 50
    slide_name = row[1]['slide_rename']
    is_in_9 = slide_name in np.array(s9['patient barcode'])
    is_in_10 = slide_name in np.array(s10['patient barcode'])

    currently_in_9.append(is_in_9)
    currently_in_10.append(is_in_10)
dat['currently_in_9'] = currently_in_9
dat['currently_in_10'] = currently_in_10

dat.to_excel(os.path.join(dn, 'the_correct_boxes_RanS 310821 fixed.xlsx'))

print('aa')