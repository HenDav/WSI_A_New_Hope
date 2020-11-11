import numpy as np
from random import shuffle
import csv

folds = ['test']*107
for ii in np.arange(1,6):
    folds.extend(list(np.ones(120)*ii))
shuffle(folds)

with open(r'C:\Users\User\Dropbox\Technion work 2020\Code\Omers segmentation\WSI_MIL\All Data\folds.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(folds)