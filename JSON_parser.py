import json
import pandas as pd
import numpy as np

fn = r'C:\ran_data\TCGA_lung\metadata.cart.2017-03-02T00_36_30.276824.json'
f = open(fn)
data = json.load(f)

slide = np.zeros(len(data), dtype=object)
project_id = np.zeros(len(data), dtype=object)
sample_type = np.zeros(len(data), dtype=object)

# Iterating through the json file
for ii, i_data in enumerate(data):
    slide[ii] = i_data['file_name']
    cases = i_data['cases']
    if len(cases) == 1:
        project_id[ii] = i_data['cases'][0]['project']['project_id']
        if len(data[ii]['cases'][0]['samples']) == 1:
            sample_type[ii] = i_data['cases'][0]['samples'][0]['sample_type']
        else:
            IOError('error2! in file ' + slide[ii])
    else:
        IOError('error1! in file ' + slide[ii])

# Closing file
f.close()

df = pd.DataFrame((slide, project_id, sample_type))
df.to_excel(r'C:\ran_data\TCGA_lung\labels_out.xlsx')

print('Finished')