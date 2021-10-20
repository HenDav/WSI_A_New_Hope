import pandas as pd
import os
import numpy as np

dn = r'C:\ran_data\ABCTB'
dat_file = r'ABCTB_FollowUp_Data_Ran.xlsx'
slides_data_file = r'slides_data_ABCTB.xlsx'

s = pd.read_excel(os.path.join(dn, slides_data_file))
dat = pd.read_excel(os.path.join(dn, dat_file))

surv = []

for row in dat.iterrows():
    barcode = row[1]['Identifier']
    fu_status = row[1]['Follow-up Status']
    surv_status = fu_status[:5]
    dt = row[1]['Follow-up Months Since Diagnosis']
    if fu_status[:5] == 'Alive' and dt > 60:
        surv.append('Positive')
    elif fu_status == 'Died from disease' and dt < 60:
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
s.to_excel(os.path.join(dn, 'slides_data_ABCTB_w_surv.xlsx'))
print('aa')