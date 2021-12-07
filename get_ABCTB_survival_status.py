import pandas as pd
import os
import numpy as np

dn = r'C:\ran_data\ABCTB'
#dat_file = r'ABCTB_FollowUp_Data_Ran.xlsx'
dat_file = r'ABCTB_FollowUp_Data_28-11-21.xlsx'
slides_data_file = r'slides_data_ABCTB.xlsx'

s = pd.read_excel(os.path.join(dn, slides_data_file))
dat = pd.read_excel(os.path.join(dn, dat_file))

surv = []

for row in dat.iterrows():
    barcode = row[1]['Identifier']
    fu_status = row[1]['Follow-up Status']
    surv_status = fu_status[:5]
    dt = row[1]['Follow-up Months Since Diagnosis']
    exclude = row[1]['Exclude for time prediction?']
    #if fu_status[:5] == 'Alive' and dt > 60:
    if exclude == 'Exclude':
        surv.append('Missing Data')
    elif (fu_status[:5] == 'Alive' or fu_status == 'Lost to Follow up') and dt > 58: #RanS 1.12.21
        surv.append('Positive')
    #elif fu_status == 'Died from disease' and dt < 60:
    elif fu_status == 'Died from disease' and dt < 63: #RanS 1.12.21
        surv.append('Negative')
    else:
        #ignore all other cases:
        #other/unknown death reasons
        #survived with dt < 60
        #lost to follow up/ No follow-up data
        surv.append('Missing Data')

dat['survival status'] = surv

s = s.merge(right=dat, left_on='patient barcode', right_on='Identifier', how='outer')
#dat.to_excel(os.path.join(dn, 'ABCTB_FollowUp_Data_Ran_survival.xlsx'))
s.to_excel(os.path.join(dn, 'slides_data_ABCTB_011221.xlsx'))
print('Finished')