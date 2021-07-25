# This file is the main run file to create the backgroud tile grid to be used in the segmentation net

from utils_data_managment import make_background_grid

make_background_grid(DataSet='TCGA')
make_background_grid(DataSet='HEROHE')
