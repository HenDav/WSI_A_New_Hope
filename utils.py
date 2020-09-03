import cv2 as cv
import numpy as np
from PIL import Image
import os
import openslide
import pandas as pd
import shutil
import glob
from tqdm import tqdm
from typing import List, Tuple
import pickle
from random import sample
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms


def get_normalization_values(data_path: 'str'='tcga-data/') -> tuple:
    """
    This function tuns over a set of images and compute mean and variance of each channel
    :return:
    """

    # get a list of all directories with images:
    dirs = _get_tcga_id_list(data_path)
    stats_list =[]
    print('Computing image set Mean and Variance...')
    # gather tissue image values from thumbnail image using the segmentation map:
    for idx, dir in enumerate(dirs):
        image_stats = {}
        thumb = np.array(Image.open(os.path.join(data_path, dir, 'thumb.png')))
        segMap = np.array(Image.open(os.path.join(data_path, dir, 'segMap.png')))
        tissue = thumb.transpose(2, 0, 1) * segMap
        tissue_pixels = (tissue[0] != 0).sum()
        tissue_matter = np.where(tissue[0] != 0)
        values = tissue[:, tissue_matter[0], tissue_matter[1]]
        image_stats['Pixels'] = tissue_pixels
        image_stats['Mean'] = values.mean(axis=1)
        image_stats['Var'] = values.var(axis=1)
        stats_list.append(image_stats)

    # Save data to file:
    with open(os.path.join(data_path, 'ImageStatData.data'), 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(stats_list, filehandle)

    # Compute total mean and var:
    N = 0
    running_mean = 0
    running_mean_squared = 0
    running_var = 0
    for _, item in enumerate(stats_list):
        n = item['Pixels']
        N += n
        running_mean += item['Mean'] * n
        running_mean_squared += (item['Mean'] ** 2) * n
        running_var += item['Var'] * n

    total_mean = running_mean / N
    total_var = (running_mean_squared + running_var) / N - total_mean ** 2
    print('Finished computing statistical data')
    print('Mean: {}'.format(total_mean))
    print('Variance: {}'.format(total_var))
    return total_mean, total_var

def _choose_data(file_name: str, how_many: int = 50) -> np.ndarray:
    """
    This function choose and returns data to be held by DataSet
    :param file_name:
    :param how_many: how_many describes how many tiles to pick from the whole image
    :return:
    """

    # open grid list:
    grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], 'grid_tlsz256.data')
    with open(grid_file, 'rb') as filehandle:
        # read the data as binary data stream
        grid_list = pickle.load(filehandle)

    # Choose locations from the grid:
    loc_num = len(grid_list)
    idxs = sample(range(loc_num), how_many)
    locs = [grid_list[idx] for idx in idxs]

    image_tiles = _get_tiles(file_name, locs, 256)

    return image_tiles


def _get_tiles(file_name: str, locations: List[Tuple], tile_sz: int) -> np.ndarray:
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """
    # open the .svs file:
    img = openslide.open_slide(file_name)
    tiles_num = len(locations)
    tiles = np.zeros((tiles_num, 3, tile_sz, tile_sz), dtype=int)
    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        tiles[idx, :, :, :] = np.array(img.read_region((loc[1], loc[0]), 0, (256, 256)).convert('RGB')).transpose(2, 0, 1)

    return tiles


def make_grid(data_file: str = 'tcga-data/slides_data.xlsx', tile_sz: int = 256):
    """
    This function creates a location for all top left corners of the grid
    :param data_file: name of main excel data file containing size of images (this file is created by function :"make_slides_xl_file")
    :param tile_sz: size of tiles to be created
    :return:
    """


    basic_DF = pd.read_excel(data_file)
    files = list(basic_DF['file'])
    basic_DF.set_index('file', inplace=True)
    tile_nums = []
    total_tiles =[]
    print('Starting Grid production...')
    #for _, file in enumerate(files):
    for i in tqdm(range(len(files))):
        file = files[i]
        data_dict = {}
        height = basic_DF.loc[file, 'Height']
        width  = basic_DF.loc[file, 'Width']

        id = basic_DF.loc[file, 'id']

        basic_grid = [(row, col) for row in range(0, height, tile_sz) for col in range(0, width, tile_sz)]
        total_tiles.append((len(basic_grid)))

        # We now have to check, which tiles of this grid are legitimate, meaning they contain enough tissue material.
        legit_grid = _legit_grid(os.path.join(data_file.split('/')[0], id, 'segMap.png'),
                                 basic_grid,
                                 tile_sz,
                                 (height, width))

        # create a list with number of tiles in each file
        tile_nums.append(len(legit_grid))

        # Save the grid to file:
        file_name = os.path.join(data_file.split('/')[0], id, 'grid_tlsz' + str(tile_sz) + '.data')
        with open(file_name, 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(legit_grid, filehandle)

    # Adding the number of tiles to the excel file:
    basic_DF['Legitimate tiles 256'] = tile_nums
    basic_DF['Total tiles 256'] = total_tiles
    basic_DF['Slide tile usage'] = list(np.array(tile_nums) / np.array(total_tiles))
    basic_DF.to_excel(data_file)

    print('Finished Grid production phase !')


def _legit_grid(image_file_name: str, grid: List[Tuple], tile_size: int, size: tuple, coverage: int = 0.5) -> List[Tuple]:
    """
    This function gets a .svs file name, a basic grid and tile size and returns a list of legitimate grid locations.
    :param image_file_name: .svs file name
    :param grid: basic grid
    :param tile_size: tile size
    :param size: size of original image (height, width)
    :param coverage: Coverage of tissue to make the slide legitimate
    :return:
    """

    # Check if coverage is a number in the range (0, 1]
    if not (coverage > 0 and coverage <= 1):
        raise ValueError('Coverage Parameter should be in the range (0,1]')

    # open the segmentation map image from which the coverage will be calculated:
    segMap = np.array(Image.open(image_file_name))
    rows = size[0] / segMap.shape[0]
    cols = size[1] / segMap.shape[1]

    # the complicated next line only rounds up the numbers
    small_tile = (int(-(-tile_size//rows)), int(-(-tile_size//cols)))
    # computing the compatible grid for the small segmenatation map:
    idx_to_remove =[]
    for idx, (row, col) in enumerate(grid):
        new_row = int(-(-(row // rows)))
        new_col = int(-(-(col // cols)))

        # collect the data from the segMap:
        tile = segMap[new_row : new_row + small_tile[0], new_col : new_col + small_tile[1]]
        tile_pixels = small_tile[0] * small_tile[1]
        tissue_coverage = tile.sum() / tile_pixels
        if tissue_coverage < coverage:
            idx_to_remove.append(idx)

    # We'll now remove items from the grid. starting from the end to the beginning in order to keep the indices correct:
    for idx in reversed(idx_to_remove):
        grid.pop(idx)

    return grid


def make_slides_xl_file(path: str = 'tcga-data'):
    """
    This function goes over all directories and makes a table with slides data:
    (1) id
    (2) file name
    (3) ER, PR, Her2 status
    (4) size of image
    (5) MPP (Microns Per Pixel)
    It also erases all 'log' subdirectories
    :return:
    """

    TCGA_BRCA_DF = pd.read_excel(os.path.join(path, 'TCGA_BRCA.xlsx'))
    TCGA_BRCA_DF.set_index('bcr_patient_barcode', inplace=True)
    try:
        slides_data = pd.read_excel(os.path.join(path, 'slides_data.xlsx'))
        create_file = False
    except:
        create_file = True
        print('slides data file is not found')
        print('Creating a new file in path: {}'.format(path))

    id_list = []

    for idx, (root, dirs, files) in enumerate(tqdm(os.walk(path))):
        id_dict = {}
        if idx is 0:
            all_dirs = dirs
        else:
            if 'logs' in dirs:
                # shutil.rmtree(os.path.join(root, 'logs'))  # Erase all 'logs' path
                pass

            # get all *.svs files in the directory:
            files = glob.glob(os.path.join(root, '*.svs'))
            for _, file in enumerate(files):
                # Create a dictionary to the files and id's:
                id_dict['patient barcode'] = '-'.join(file.split('/')[-1].split('-')[0:3])
                id_dict['id'] = root.split('/')[-1]
                id_dict['file'] = file.split('/')[-1]

                # Get some basic data about the image like MPP (Microns Per Pixel) and size:
                img = openslide.open_slide(file)
                id_dict['MPP'] = img.properties['aperio.MPP']
                id_dict['Width'] = img.dimensions[0]
                id_dict['Height'] = img.dimensions[1]
                id_dict['Objective Power'] = img.properties['openslide.objective-power']
                id_dict['Scan Date'] = img.properties['aperio.Date']
                img.close()

                # Get data from 'TCGA_BRCA.xlsx' and add to the dictionary ER_status, PR_status, Her2_status
                id_dict['ER status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['ER_status']].values[0][0]
                id_dict['PR status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['PR_status']].values[0][0]
                id_dict['Her2 status'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['Her2_status']].values[0][0]
                id_dict['test fold idx'] = TCGA_BRCA_DF.loc[[id_dict['patient barcode']], ['Test_fold_idx']].values[0][0]


                id_list.append(id_dict)

    if create_file:
        slides_data = pd.DataFrame(id_list)
        slides_data.to_excel(os.path.join(path, 'slides_data.xlsx'))

    print('Finished data preparation')


def make_segmentations(path: str = 'tcga-data/', magnification: int = 5):
    print('Starting in making Segmentation Maps for each .svs file...')
    dirs = _get_tcga_id_list(path)
    #for _, dir in enumerate(dirs):
    for i in tqdm(range(len(dirs))):  # In order to get rid of tqdm, just erase this line and un-comment the line above
        dir = dirs[i]
        slide = _get_slide(os.path.join(path, dir))
        #slide = _get_slide(glob.glob(os.path.join(path, '*.svs')))
        if slide is not None:
            # Get a thunmbnail image to create the segmentation for:
            objective_pwr = int(slide.properties['openslide.objective-power'])
            height = slide.dimensions[1]
            width = slide.dimensions[0]
            thumb = slide.get_thumbnail((width / (objective_pwr / magnification), height / (objective_pwr / magnification)))
            thmb_seg_map, thmb_seg_image = _make_segmentation_for_image(thumb, magnification)

            # Saving segmentation map, segmentation image and thumbnail:
            thumb.save(os.path.join(path, dir, 'thumb.png'))
            thmb_seg_map.save(os.path.join(path, dir, 'segMap.png'))
            thmb_seg_image.save(os.path.join(path, dir, 'segImage.png'))

        else:
            # TODO: implement a case for a slide that cannot be opened.
            pass
    print('Segmentation Process finished !')


def _get_slide(path: 'str') -> openslide.OpenSlide:
    """
    This function returns an OpenSlide object from the file within the directory
    :param path:
    :return:
    """

    # file = next(os.walk(path))[2]  # TODO: this line can be erased since we dont use file. also check the except part...
    #if '.DS_Store' in file: file.remove('.DS_Store')
    slide = None
    try:
        #slide = openslide.open_slide(os.path.join(path, file[0]))
        slide = openslide.open_slide(glob.glob(os.path.join(path, '*.svs'))[0])
    except:
        print('Cannot open slide at location: {}'.format(path))

    return slide


def _get_tcga_id_list(path: str = 'tcga-data'):
    """
    This function returns the id of all images in the TCGA data directory given by 'path'
    :return:
    """
    return next(os.walk(path))[1]


def _make_segmentation_for_image(image: Image, magnification: int) -> (Image, Image):
    """
    This function creates a segmentation map for an Image
    :param magnification:
    :return:
    """
    # Converting the image from RGBA to HSV and to a numpy array (from PIL):
    image_array = np.array(image.convert('HSV'))
    # otsu Thresholding:
    _, seg_map = cv.threshold(image_array[:, :, 1], 0, 255, cv.THRESH_OTSU)

    # Smoothing the tissue segmentation imaqe:
    size = 30 * magnification
    kernel_smooth = np.ones((size, size), dtype=np.float32) / size ** 2
    seg_map = cv.filter2D(seg_map, -1, kernel_smooth)

    th_val = 5
    seg_map[seg_map > th_val] = 255
    seg_map[seg_map <= th_val] = 0
    seg_map_PIL = Image.fromarray(seg_map)

    edge_image = cv.Canny(seg_map, 1, 254)
    # Make the edge thicker by dilating:
    kernel_dilation = np.ones((3, 3))  #cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edge_image = Image.fromarray(cv.dilate(edge_image, kernel_dilation, iterations=10)).convert('RGB')
    seg_image = Image.blend(image, edge_image, 0.5)

    return seg_map_PIL, seg_image


def get_transform():

    transform = transforms.Compose([ transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.8998, 0.8253, 0.9357), (0.1125, 0.1751, 0.0787))
                                     ])
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((48.9135746, 75.1674335, 59.00008673),
                                                         (2216.86003428, 5263.29276891, 3241.14047416))
                                    ])
    return None

class WSI_MILdataset(Dataset):
    def __init__(self,
                 root_path: str ='tcga-data',
                 tile_size: int=256,
                 num_of_tiles: int=50,
                 target_kind: str='ER',
                 transform=None):
        if target_kind not in ['ER', 'PR', 'Her2']:
            raise ValueError('target should be one of: ER, PR, Her2')

        excel_data = os.path.join(root_path, 'slides_data.xlsx')
        self.target ={}
        excel_DF = pd.read_excel(excel_data)
        self.tile_size = tile_size
        self.root_path = root_path
        self.target_kind = target_kind
        self.image_file_names = list(excel_DF['file'])
        self.image_path_names = list(excel_DF['id'])
        self.target['ER'] = list(excel_DF['ER status'])
        self.target['PR'] = list(excel_DF['PR status'])
        self.target['Her2'] = list(excel_DF['Her2 status'])
        self.in_fold = list(excel_DF['test fold idx'])
        self.tiles_total = list(excel_DF['Tiles 256'])
        self.num_of_tiles = num_of_tiles
        self.transform = transform


    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root_path, self.image_path_names[idx], self.image_file_names[idx])
        tiles = _choose_data(file_name, self.num_of_tiles)
        label = [1] if self.target[self.target_kind][idx] == 'Positive' else [0]
        label = torch.LongTensor(label)

        N = tiles.shape[0]
        shape = tiles.shape
        X = torch.zeros(shape)
        if not self.transform:
            self.transform = transforms.Compose([transforms.ToTensor()])

        tiles = tiles.transpose(0, 2, 3, 1)
        for i in range(N):
            X[i] = self.transform(tiles[i])

        return X, label







"""
class MnistMILdataset(Dataset):
    def __init__(self, all_image_tiles, labels, transform=None):
        self.all_image_tiles = all_image_tiles
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tiles = self.all_image_tiles[idx]
        x, instance_locations = _make_bag(tiles)
        y = self.labels[idx]

        shape = x.shape
        X = torch.zeros(shape)
        if self.transform:
            x = x.transpose(0, 2, 3, 1)

            for i in range(x.shape[0]):
                X[i] = self.transform(x[i])

        return X, y, instance_locations
"""