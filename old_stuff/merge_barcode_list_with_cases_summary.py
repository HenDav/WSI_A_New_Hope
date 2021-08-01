import pandas as pd
import os

dn = r'C:\ran_data\BoneMarrow'
file1 = r'ALL_cases_summary_16-06-21_boxes2-4.xlsx'
file2 = r'barcode_list_box1.xlsx'

df1 = pd.read_excel(os.path.join(dn,file1), sheet_name=1)
df2 = pd.read_excel(os.path.join(dn,file2))
df1['merge_name'] = df1['merge_name'].astype(str)
df2['merge_name'] = df2['merge_name'].astype(str)

df_m = pd.merge(df1, df2,how='outer',on='merge_name')
df_m.to_excel(os.path.join(dn, 'out.xlsx'))

print('finished')