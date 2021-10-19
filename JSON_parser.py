import json
import pandas as pd
import numpy as np

fn = r'C:\ran_data\TCGA_lung\metadata.cart.2017-03-02T00_36_30.276824.json'
f = open(fn)
data = json.load(f)

slide = np.zeros(len(data), dtype=object)
project_id = np.zeros(len(data), dtype=object)
sample_type = np.zeros(len(data), dtype=object)
tumor_percent = np.zeros(len(data), dtype=object)

tumor_percent_field_name = 'percent_tumor_nuclei'

# Iterating through the json file
for ii, i_data in enumerate(data):
    slide[ii] = i_data['file_name']
    cases = i_data['cases']
    if len(cases) == 1:
        project_id[ii] = i_data['cases'][0]['project']['project_id']
        if len(data[ii]['cases'][0]['samples']) == 1:
            sample_type[ii] = i_data['cases'][0]['samples'][0]['sample_type']
            if len(i_data['cases'][0]['samples'][0]['portions']) == 1:
                if len(data[ii]['cases'][0]['samples'][0]['portions'][0]['slides']) == 1:
                    tumor_percent[ii] = i_data['cases'][0]['samples'][0]['portions'][0]['slides'][0][tumor_percent_field_name]
                else:
                    #more than 1 slide, match filename to slidename
                    slidenames = [slide_data['submitter_id'] for slide_data in data[ii]['cases'][0]['samples'][0]['portions'][0]['slides']]
                    slide_ind = np.where(np.array(slidenames) == slide[ii].split('.')[0])[0].item()
                    tumor_percent[ii] = i_data['cases'][0]['samples'][0]['portions'][0]['slides'][slide_ind][tumor_percent_field_name]
            else:
                IOError('error3! in file ' + slide[ii])
        else:
            IOError('error2! in file ' + slide[ii])
    else:
        IOError('error1! in file ' + slide[ii])

# Closing file
f.close()

df = pd.DataFrame((slide, project_id, sample_type, tumor_percent))
df.to_excel(r'C:\ran_data\TCGA_lung\labels_out_2.xlsx')

print('Finished')