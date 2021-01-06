import numpy as np
from PIL import Image
import os
import openslide
import pandas as pd
import glob
import pickle
from random import sample
import random
import torch
from torchvision import transforms
import sys
import time
from typing import List, Tuple
from xlrd.biffh import XLRDError
from zipfile import BadZipFile
from HED_space import HED_color_jitter
from skimage.util import random_noise
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None
from nets_mil import ResNet34_GN_GatedAttention, ResNet50_GN_GatedAttention, ReceptorNet




def chunks(list: List, length: int):
    new_list = [ list[i * length:(i + 1) * length] for i in range((len(list) + length - 1) // length )]
    return new_list


def make_dir(dirname):
    if not dirname in next(os.walk(os.getcwd()))[1]:
        try:
            os.mkdir(dirname)
        except OSError:
            print('Creation of directory ', dirname, ' failed...')
            raise


def _choose_data(grid_file: str, image_file: str, how_many: int, magnification: int = 20, tile_size: int = 256, print_timing: bool = False):
    """
    This function choose and returns data to be held by DataSet
    :param file_name:
    :param how_many: how_many describes how many tiles to pick from the whole image
    :return:
    """
    BASIC_OBJ_POWER = 20
    #adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    adjusted_tile_size = int(tile_size * (magnification / BASIC_OBJ_POWER))  # RanS 22.12.20
    # open grid list:
    with open(grid_file, 'rb') as filehandle:
        grid_list = pickle.load(filehandle)

    # Choose locations from the grid:
    loc_num = len(grid_list)

    #temp try RanS 4.1.21
    try:
        idxs = sample(range(loc_num), how_many)
    except:
        print('image_file:', image_file)
        print('how_many:', str(how_many))
        print('loc_num:', str(loc_num))

    locs = [grid_list[idx] for idx in idxs]

    image_tiles, time_list = _get_tiles(image_file, locs, adjusted_tile_size, print_timing=print_timing)

    return image_tiles, time_list


def _get_tiles(file_name: str, locations: List[Tuple], tile_sz: int, print_timing: bool = False):
    """
    This function returns an array of tiles
    :param file_name:
    :param locations:
    :param tile_sz:
    :return:
    """

    # open the .svs file:
    start_openslide = time.time()
    img = openslide.open_slide(file_name)
    end_openslide = time.time()

    tiles_num = len(locations)

    #RanS 20.12.20 - plot thumbnail with tile locations
    temp = False
    if temp:
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        level_1 = img.level_count - 5
        ld = int(img.level_downsamples[level_1]) #level downsample
        thumb = (img.read_region(location=(0, 0), level=level_1, size=img.level_dimensions[level_1])).convert('RGB')
        fig, ax = plt.subplots()
        plt.imshow(thumb)
        for idx, loc in enumerate(locations):
            print((loc[1]/ld, loc[0]/ld))
            rect = Rectangle((loc[1]/ld, loc[0]/ld), tile_sz / ld, tile_sz / ld, color='r', linewidth=3, fill=False)
            ax.add_patch(rect)
            #rect = Rectangle((loc[1] / ld, loc[0] / ld), tile_sz / ld, tile_sz / ld, color='g', linewidth=3, fill=False)
            #ax.add_patch(rect)

        patch1 = img.read_region((loc[1], loc[0]), 0, (600, 600)).convert('RGB')
        plt.figure()
        plt.imshow(patch1)

        patch2 = img.read_region((loc[1], loc[0]), 0, (2000, 2000)).convert('RGB')
        plt.figure()
        plt.imshow(patch2)

        plt.show()

    tiles_PIL = []

    start_gettiles = time.time()
    for idx, loc in enumerate(locations):
        # When reading from OpenSlide the locations is as follows (col, row) which is opposite of what we did
        try:
            image = img.read_region((loc[1], loc[0]), 0, (tile_sz, tile_sz)).convert('RGB')
        except:
            print('failed to read slide ' + file_name + ' in location ' + str(loc[1]) + ',' + str(loc[0])) # debug errors in reading slides
            print('taking blank patch instead')
            image = Image.fromarray(np.zeros([tile_sz, tile_sz, 3], dtype=np.uint8))
        tiles_PIL.append(image)

    end_gettiles = time.time()


    if print_timing:
        time_list = [end_openslide - start_openslide, (end_gettiles - start_gettiles) / tiles_num]
    else:
        time_list = [0]

    return tiles_PIL, time_list


def _get_grid_list(file_name: str, magnification: int = 20, tile_size: int = 256):
    """
    This function returns the grid location of tile for a specific slide.
    :param file_name:
    :return:
    """

    BASIC_OBJ_POWER = 20
    adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    basic_grid_file_name = 'grid_tlsz' + str(adjusted_tile_size) + '.data'

    # open grid list:
    grid_file = os.path.join(file_name.split('/')[0], file_name.split('/')[1], basic_grid_file_name)
    with open(grid_file, 'rb') as filehandle:
        # read the data as binary data stream
        grid_list = pickle.load(filehandle)

        return grid_list


def _get_slide(path: 'str', data_format: str = 'TCGA') -> openslide.OpenSlide:
    """
    This function returns an OpenSlide object from the file within the directory
    :param path:
    :return:
    """
    if data_format == 'TCGA':
        # file = next(os.walk(path))[2]  # TODO: this line can be erased since we dont use file. also check the except part...
        # if '.DS_Store' in file: file.remove('.DS_Store')
        slide = None
        try:
            # slide = openslide.open_slide(os.path.join(path, file[0]))
            slide = openslide.open_slide(glob.glob(os.path.join(path, '*.svs'))[0])
        except:
            print('Cannot open slide at location: {}'.format(path))
    elif data_format == 'ABCTB' or data_format == 'MIRAX':
        slide = None
        try:
            slide = openslide.open_slide(path)
        except:
            print('Cannot open slide at location: {}'.format(path))

    return slide


def _get_tcga_id_list(path: str = 'tcga-data'):
    """
    This function returns the id of all images in the TCGA data directory given by 'path'
    :return:
    """
    return next(os.walk(path))[1]


def device_gpu_cpu():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using cpu')

    return device


def get_cpu():
    platform = sys.platform
    if platform == 'linux':
        cpu = len(os.sched_getaffinity(0))
    elif platform == 'darwin':
        cpu = 2
        platform = 'MacOs'
    else:
        cpu = 1
        platform = 'Unrecognized'


    #cpu = 20
    print('Running on {} with {} workers'.format(platform, cpu))
    return cpu


def run_data(experiment: str = None, test_fold: int = 1, transform_type: str = 'none',#transformations: bool = False,
             tile_size: int = 256, tiles_per_bag: int = 50, num_bags: int = 1, DX: bool = False, DataSet: str = 'TCGA',
             epoch: int = None, model: str = None, transformation_string: str = None, Receptor: str = None,
             MultiSlide: bool = False):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :return:
    """

    run_file_name = 'runs/run_data.xlsx'
    if os.path.isfile(run_file_name):
        try:
            run_DF = pd.read_excel(run_file_name)
        except (XLRDError, BadZipFile):
            print('Couldn\'t open file {}'.format(run_file_name))
            return
        try:
            run_DF.drop(labels='Unnamed: 0', axis='columns',  inplace=True)
        except KeyError:
            pass

        run_DF_exp = run_DF.set_index('Experiment', inplace=False)
    else:
        run_DF = pd.DataFrame()

    # If a new experiment is conducted:
    if experiment is None:
        if os.path.isfile(run_file_name):
            experiment = run_DF_exp.index.values.max() + 1
        else:
            experiment = 1

        location = 'runs/Exp_' + str(experiment) + '-TestFold_' + str(test_fold)
        run_dict = {'Experiment': experiment,
                    'Test Fold': test_fold,
                    #'Transformations': transformations,
                    'Transformations': transform_type,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'MultiSlide Per Bag': MultiSlide,
                    'No. of Bags': num_bags,
                    'Location': location,
                    'DX': DX,
                    'DataSet': DataSet,
                    'Receptor': Receptor,
                    'Model': 'None',
                    'Last Epoch': 0,
                    'Transformation String': 'None'
                    }
        run_DF = run_DF.append([run_dict], ignore_index=True)
        if not os.path.isdir('runs'):
            os.mkdir('runs')

        run_DF.to_excel(run_file_name)
        print('Created a new Experiment (number {}). It will be saved at location: {}'.format(experiment, location))

        return location, experiment

    elif experiment is not None and epoch is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Last Epoch'] = epoch
        run_DF.to_excel(run_file_name)
    elif experiment is not None and model is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Model'] = model
        run_DF.to_excel(run_file_name)
    elif experiment is not None and transformation_string is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Transformation String'] = transformation_string
        run_DF.to_excel(run_file_name)


    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        #transformations = bool(run_DF_exp.loc[[experiment], ['Transformations']].values[0][0])
        transformations = run_DF_exp.loc[[experiment], ['Transformations']].values[0][0] #RanS 9.12.20
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])
        DX = bool(run_DF_exp.loc[[experiment], ['DX']].values[0][0])
        DataSet = str(run_DF_exp.loc[[experiment], ['DataSet']].values[0][0])
        Receptor = str(run_DF_exp.loc[[experiment], ['Receptor']].values[0][0])
        MultiSlide = str(run_DF_exp.loc[[experiment], ['MultiSlide Per Bag']].values[0][0])

        return location, test_fold, transformations, tile_size, tiles_per_bag, DX, DataSet, Receptor, MultiSlide


def get_concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = torch.randint(low=0, high=h, size=(1,)).numpy()[0]
            x = torch.randint(low=0, high=w, size=(1,)).numpy()[0]
            '''
            # Numpy random numbers will produce the same numbers in every epoch - I changed the random number producer
            # to torch.random to overcome this issue. 
            y = np.random.randint(h)            
            x = np.random.randint(w)
            '''


            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class MyRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class MyCropTransform:
    """crop the image at upper left."""

    def __init__(self, tile_size):
        self.tile_size = tile_size

    def __call__(self, x):
        #x = transforms.functional.crop(img=x, top=0, left=0, height=self.tile_size, width=self.tile_size)
        x = transforms.functional.crop(img=x, top=x.size[0] - self.tile_size, left=x.size[1] - self.tile_size, height=self.tile_size, width=self.tile_size)
        return x


class MyGaussianNoiseTransform:
    """add gaussian noise."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        #x += torch.normal(mean=np.zeros_like(x), std=self.sigma)
        stdev = self.sigma[0]+(self.sigma[1]-self.sigma[0])*np.random.rand()
        # convert PIL Image to ndarray
        x_arr = np.asarray(x)

        # random_noise() method will convert image in [0, 255] to [0, 1.0],
        # inherently it use np.random.normal() to create normal distribution
        # and adds the generated noised back to image
        noise_img = random_noise(x_arr, mode='gaussian', var=stdev ** 2)
        noise_img = (255 * noise_img).astype(np.uint8)

        x = Image.fromarray(noise_img)
        return x


class HEDColorJitter:
    """Jitter colors in HED color space rather than RGB color space."""
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x):
        x_arr = np.array(x)
        x2 = HED_color_jitter(x_arr, self.sigma)
        x2 = Image.fromarray(x2)
        return x2


def define_transformations(transform_type, train, MEAN, STD, tile_size, c_param=0.1):

    # Setting the transformation:
    final_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                              std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))])
    scale_factor = 0
    # if self.transform and self.train:
    if transform_type != 'none' and train:
        # TODO: Consider using - torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        if transform_type == 'flip':
            transform1 = \
                transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip()])
        elif transform_type == 'rvf': #rotate, vertical flip
            transform1 = \
                transforms.Compose([MyRotation(angles=[0, 90, 180, 270]),
                                     transforms.RandomVerticalFlip()])
        elif transform_type in ['cbnfrsc', 'cbnfrs']:  # color, blur, noise, flip, rotate, scale, +-cutout
            scale_factor = 0.2
            transform1 = \
                transforms.Compose([
                    # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type in ['pcbnfrsc', 'pcbnfrs']:  # parameterized color, blur, noise, flip, rotate, scale, +-cutout
        #elif transform_type == 'c_0_05_bnfrsc' or 'c_0_05_bnfrs':  # color 0.1, blur, noise, flip, rotate, scale, +-cutout
            scale_factor = 0.2
            #c_param = 0.05
            transform1 = \
                transforms.Compose([
                    # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                    #transforms.ColorJitter(brightness=(1-c_param*1, 1+c_param*1), contrast=(1-c_param*2, 1+c_param*2),  # RanS 2.12.20
                    #                       saturation=c_param, hue=(-c_param, c_param)),
                    transforms.ColorJitter(brightness=c_param, contrast=c_param * 2, saturation=c_param, hue=c_param),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type == 'cbnfr':  # color, blur, noise, flip, rotate
            scale_factor = 0
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type in ['bnfrsc', 'bnfrs']:  # blur, noise, flip, rotate, scale, +-cutout
            scale_factor = 0.2
            transform1 = \
                transforms.Compose([
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type == 'frs':  # flip, rotate, scale
            scale_factor = 0.2
            transform1 = \
                transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type == 'hedcfrs':  # HED color, flip, rotate, scale
            scale_factor = 0.2
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25)),
                    HEDColorJitter(sigma=0.05),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.CenterCrop(tile_size),  # fix boundary when scaling<1
                    #transforms.functional.crop(top=0, left=0, height=tile_size, width=tile_size)
                    # fix boundary when scaling<1
                ])

        transform = transforms.Compose([transform1, final_transform])
    else:
        transform = final_transform

    if transform_type in ['cbnfrsc', 'bnfrsc', 'c_0_05_bnfrsc', 'pcbnfrsc']:
        transform.transforms.append(Cutout(n_holes=1, length=100)) #RanS 24.12.20

    return transform, scale_factor


def define_data_root(DataSet):
    # Define data root:
    if sys.platform == 'linux': #GIPdeep
        if DataSet == 'LUNG':
            ROOT_PATH = r'/home/rschley/All_Data/LUNG'
        else:
            ROOT_PATH = r'/home/womer/project/All Data'

    elif sys.platform == 'win32': #Ran local
        if DataSet == 'HEROHE':
            ROOT_PATH = r'C:\ran_data\HEROHE_examples'
        elif DataSet == 'TCGA':
            ROOT_PATH = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat'
        elif DataSet == 'LUNG':
            ROOT_PATH = r'C:\ran_data\Lung_examples'
        elif DataSet == 'RedSquares':
            ROOT_PATH = r'C:\ran_data\RedSquares'
        else:
            print('Error - no ROOT_PATH defined')
    else: #Omer local
        if DataSet == 'LUNG':
            ROOT_PATH = 'All Data/LUNG'
        else:
            ROOT_PATH = r'All Data'

    return ROOT_PATH

def assert_dataset_target(DataSet,target_kind):
    if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
        raise ValueError('target should be one of: PDL1, EGFR')
    elif ((DataSet == 'HEROHE') or (DataSet == 'TCGA')) and target_kind not in ['ER', 'PR', 'Her2']:
        raise ValueError('target should be one of: ER, PR, Her2')
    elif (DataSet == 'RedSquares') and target_kind != 'RedSquares':
        raise ValueError('target should be: RedSquares')

def show_patches_and_transformations(X, images, tiles, scale_factor, tile_size):
    fig1, fig2, fig3, fig4, fig5 = plt.figure(), plt.figure(), plt.figure(), plt.figure(), plt.figure()
    fig1.set_size_inches(32, 18)
    fig2.set_size_inches(32, 18)
    fig3.set_size_inches(32, 18)
    fig4.set_size_inches(32, 18)
    fig5.set_size_inches(32, 18)
    grid1 = ImageGrid(fig1, 111, nrows_ncols=(2, 5), axes_pad=0)
    grid2 = ImageGrid(fig2, 111, nrows_ncols=(2, 5), axes_pad=0)
    grid3 = ImageGrid(fig3, 111, nrows_ncols=(2, 5), axes_pad=0)
    grid4 = ImageGrid(fig4, 111, nrows_ncols=(2, 5), axes_pad=0)
    grid5 = ImageGrid(fig5, 111, nrows_ncols=(2, 5), axes_pad=0)

    for ii in range(10):
        img1 = np.squeeze(images[ii, :, :, :])
        grid1[ii].imshow(np.transpose(img1, axes=(1, 2, 0)))

        img2 = np.squeeze(X[ii, :, :, :])
        grid2[ii].imshow(np.transpose(img2, axes=(1, 2, 0)))

        trans_no_norm = \
            transforms.Compose([
                transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=0.1,
                                       hue=(-0.1, 0.1)),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                MyRotation(angles=[0, 90, 180, 270]),
                transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                transforms.CenterCrop(tile_size),  # fix boundary when scaling<1
                transforms.ToTensor()
            ])

        img3 = trans_no_norm(tiles[ii])
        grid3[ii].imshow(np.transpose(img3, axes=(1, 2, 0)))

        trans0 = transforms.ToTensor()
        img4 = trans0(tiles[ii])
        grid4[ii].imshow(np.transpose(img4, axes=(1, 2, 0)))

        color_trans = transforms.Compose([
            transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                   saturation=0.1, hue=(-0.1, 0.1)),
            transforms.ToTensor()])

        blur_trans = transforms.Compose([
            transforms.GaussianBlur(5, sigma=0.1),  # RanS 23.12.20
            transforms.ToTensor()])

        noise_trans = transforms.Compose([
            MyGaussianNoiseTransform(sigma=(0.05, 0.05)),  # RanS 23.12.20
            transforms.ToTensor()])

        cutout_trans = transforms.Compose([
            transforms.ToTensor(),
            Cutout(n_holes=1, length=100)])  # RanS 24.12.20]

        # img5 = color_trans(tiles[ii])
        # img5 = blur_trans(tiles[ii])
        # img5 = noise_trans(tiles[ii])
        img5 = cutout_trans(tiles[ii])
        grid5[ii].imshow(np.transpose(img5, axes=(1, 2, 0)))

    fig1.suptitle('original patches', fontsize=14)
    fig2.suptitle('final patches', fontsize=14)
    fig3.suptitle('all trans before norm', fontsize=14)
    fig4.suptitle('original patches, before crop', fontsize=14)
    fig5.suptitle('color transform only', fontsize=14)

    plt.show()


def get_model(model_name):
    if model_name == 'resnet50_gn':
        model = ResNet50_GN_GatedAttention()
    elif model_name == 'receptornet':
        model = ReceptorNet()
    return model