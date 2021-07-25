import pandas as pd
import openslide
import os

slides_file = r'/home/womer/project/All Data/slides_data.xlsx'
out_file = r'/home/rschley/code/WSI_MIL/general_try4/slides_data_with_date.xlsx'

slides_df = pd.read_excel(slides_file)

slides_df['Date'] = 'not extracted from metadata'
for index, slide in slides_df.iterrows():
    try:
        file_path = os.path.join(r'/home/womer/project/All Data', slide['id'], slide['file'])
        img = openslide.open_slide(file_path)
        slides_df.loc[index, 'Date'] = img.properties['aperio.Date']
    except:
        pass

slides_df.to_excel(out_file)
print('finished')