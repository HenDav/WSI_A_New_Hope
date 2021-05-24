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
from nets_mil import ResNet34_GN_GatedAttention, ResNet50_GN_GatedAttention, ReceptorNet
import nets
from math import isclose
from argparse import Namespace as argsNamespace
from shutil import copy2
from datetime import date
import platform

Image.MAX_IMAGE_PIXELS = None


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


def get_optimal_slide_level(slide, magnification, desired_mag, tile_size):
    desired_downsample = magnification / desired_mag  # downsample needed for each dimension (reflected by level_downsamples property)

    level, best_next_level = -1, -1
    for index, downsample in enumerate(slide.level_downsamples):
        if isclose(desired_downsample, downsample, rel_tol=1e-3):
            level = index
            level_downsample = 1
            break

        elif downsample < desired_downsample:
            best_next_level = index
            level_downsample = int(desired_downsample / slide.level_downsamples[best_next_level])

    adjusted_tile_size = tile_size * level_downsample
    best_slide_level = level if level > best_next_level else best_next_level
    level_0_tile_size = int(desired_downsample) * tile_size
    return best_slide_level, adjusted_tile_size, level_0_tile_size


def _choose_data(grid_list: list,
                 slide: openslide.OpenSlide,
                 how_many: int,
                 magnification: int,
                 tile_size: int = 256,
                 print_timing: bool = False,
                 desired_mag: int = 20):
    """
    This function choose and returns data to be held by DataSet.
    The function is in the PreLoad Version. It works with slides already loaded to memory.

    :param grid_list: A list of all grids for this specific slide
    :param slide: An OpenSlide object of the slide.
    :param how_many: how_many tiles to return from the slide.
    :param magnification: The magnification of level 0 of the slide
    :param tile_size: Desired tile size from the slide at the desired magnification
    :param print_timing: Do or don't collect timing for this procedure
    :param desired_mag: Desired Magnification of the tiles/slide.
    :return:
    """

    best_slide_level, adjusted_tile_size, level_0_tile_size = get_optimal_slide_level(slide, magnification, desired_mag, tile_size)

    # Choose locations from the grid list:
    loc_num = len(grid_list)
    # FIXME: The problem of not enough tiles should disappear when we'll work with fixed tile locations + random vertiacl/horizontal movement
    try:
        idxs = sample(range(loc_num), how_many)
    except ValueError:
        raise ValueError('Requested more tiles than available by the grid list')

    locs = [grid_list[idx] for idx in idxs]
    image_tiles, time_list = _get_tiles(slide=slide,
                                        locations=locs,
                                        tile_size_level_0=level_0_tile_size,
                                        adjusted_tile_sz=adjusted_tile_size,
                                        output_tile_sz=tile_size,
                                        best_slide_level=best_slide_level,
                                        print_timing=print_timing,
                                        random_shift=True)

    return image_tiles, time_list


def _get_tiles(slide: openslide.OpenSlide,
               locations: List[Tuple],
               tile_size_level_0: int,
               adjusted_tile_sz: int,
               output_tile_sz: int,
               best_slide_level: int,
               print_timing: bool = False,
               random_shift: bool = False,
               oversized_HC_tiles: bool = False):
    """
    This function extract tiles from the slide.
    :param slide: OpenSlide object containing a slide
    :param locations: locations of te tiles to be extracted
    :param tile_size_level_0: tile size adjusted for level 0
    :param adjusted_tile_sz: tile size adjusted for best_level magnification
    :param output_tile_sz: output tile size needed
    :param best_slide_level: best slide level to get tiles from
    :param print_timing: collect time profiling data ?
    :return:
    """

    #RanS 20.12.20 - plot thumbnail with tile locations
    temp = False
    if temp:
        from matplotlib.patches import Rectangle
        import matplotlib.pyplot as plt
        level_1 = slide.level_count - 5
        ld = int(slide.level_downsamples[level_1]) #level downsample
        thumb = (slide.read_region(location=(0, 0), level=level_1, size=slide.level_dimensions[level_1])).convert('RGB')
        fig, ax = plt.subplots()
        plt.imshow(thumb)
        for idx, loc in enumerate(locations):
            print((loc[1]/ld, loc[0]/ld))
            rect = Rectangle((loc[1]/ld, loc[0]/ld), adjusted_tile_sz / ld, adjusted_tile_sz / ld, color='r', linewidth=3, fill=False)
            ax.add_patch(rect)
            #rect = Rectangle((loc[1] / ld, loc[0] / ld), tile_sz / ld, tile_sz / ld, color='g', linewidth=3, fill=False)
            #ax.add_patch(rect)

        patch1 = slide.read_region((loc[1], loc[0]), 0, (600, 600)).convert('RGB')
        plt.figure()
        plt.imshow(patch1)

        patch2 = slide.read_region((loc[1], loc[0]), 0, (2000, 2000)).convert('RGB')
        plt.figure()
        plt.imshow(patch2)

        plt.show()

    #tiles_PIL = []

    #RanS 28.4.21, preallocate list of images
    empty_image = Image.fromarray(np.uint8(np.zeros((output_tile_sz,output_tile_sz,3))))
    tiles_PIL = [empty_image] * len(locations)

    start_gettiles = time.time()

    if oversized_HC_tiles:
        adjusted_tile_sz *= 2
        output_tile_sz *= 2
        tile_shifting = (tile_size_level_0 // 2, tile_size_level_0 // 2)

    for idx, loc in enumerate(locations):
        if random_shift:
            tile_shifting = sample(range(-tile_size_level_0 // 2, tile_size_level_0 // 2), 2)
        #elif oversized_HC_tiles:
        #    tile_shifting = (tile_size_level_0 // 2, tile_size_level_0 // 2)

        if random_shift or oversized_HC_tiles:
            new_loc_init = {'Top': loc[0] - tile_shifting[0],
                            'Left': loc[1] - tile_shifting[1]}
            new_loc_end = {'Bottom': new_loc_init['Top'] + tile_size_level_0,
                           'Right': new_loc_init['Left'] + tile_size_level_0}
            if new_loc_init['Top'] < 0:
                new_loc_init['Top'] += abs(new_loc_init['Top'])
            if new_loc_init['Left'] < 0:
                new_loc_init['Left'] += abs(new_loc_init['Left'])
            if new_loc_end['Bottom'] > slide.dimensions[1]:
                delta_Height = new_loc_end['Bottom'] - slide.dimensions[1]
                new_loc_init['Top'] -= delta_Height
            if new_loc_end['Right'] > slide.dimensions[0]:
                delta_Width = new_loc_end['Right'] - slide.dimensions[0]
                new_loc_init['Left'] -= delta_Width
        else:
            new_loc_init = {'Top': loc[0],
                            'Left': loc[1]}

        try:
            # When reading from OpenSlide the locations is as follows (col, row)
            image = slide.read_region((new_loc_init['Left'], new_loc_init['Top']), best_slide_level, (adjusted_tile_sz, adjusted_tile_sz)).convert('RGB')
        except:
            print('failed to read slide ' + slide._filename + ' in location ' + str(loc[1]) + ',' + str(loc[0]))
            print('taking blank patch instead')
            image = Image.fromarray(np.zeros([adjusted_tile_sz, adjusted_tile_sz, 3], dtype=np.uint8))

        if adjusted_tile_sz != output_tile_sz:
            image = image.resize((output_tile_sz, output_tile_sz))

        tiles_PIL[idx] = image

    end_gettiles = time.time()

    if print_timing:
        time_list = [0, (end_gettiles - start_gettiles) / len(locations)]
    else:
        time_list = [0]

    return tiles_PIL, time_list


def _get_grid_list(file_name: str, magnification: int = 20, tile_size: int = 256, desired_mag: int = 20):
    """
    This function returns the grid location of tile for a specific slide.
    :param file_name:
    :return:
    """

    #BASIC_OBJ_POWER = 20
    #adjusted_tile_size = tile_size * (magnification // BASIC_OBJ_POWER)
    adjusted_tile_size = int(tile_size * (magnification / desired_mag))  # RanS 8.2.21
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
    print('Running on {} with {} CPUs'.format(platform, cpu))
    return cpu


def run_data(experiment: str = None,
             test_fold: int = 1,
             transform_type: str = 'none',
             tile_size: int = 256,
             tiles_per_bag: int = 50,
             num_bags: int = 1,
             DX: bool = False,
             DataSet_name: list = ['TCGA'],
             DataSet_size: tuple = None,
             DataSet_Slide_magnification = None,
             epoch: int = None,
             model: str = None,
             transformation_string: str = None,
             Receptor: str = None,
             MultiSlide: bool = False,
             test_mean_auc: float = None):
    """
    This function writes the run data to file
    :param experiment:
    :param from_epoch:
    :param MultiSlide: Describes if tiles from different slides with same class are mixed in the same bag
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

        location = 'runs/Exp_' + str(experiment) + '-' + Receptor + '-TestFold_' + str(test_fold)
        if type(DataSet_name) is not list:
            DataSet_name = [DataSet_name]

        run_dict = {'Experiment': experiment,
                    'Start Date': str(date.today()),
                    'Test Fold': test_fold,
                    'Transformations': transform_type,
                    'Tile Size': tile_size,
                    'Tiles Per Bag': tiles_per_bag,
                    'MultiSlide Per Bag': MultiSlide,
                    'No. of Bags': num_bags,
                    'Location': location,
                    'DX': DX,
                    'DataSet': ' / '.join(DataSet_name),
                    'Receptor': Receptor,
                    'Model': 'None',
                    'Last Epoch': 0,
                    'Transformation String': 'None',
                    'Desired Slide Magnification': DataSet_Slide_magnification
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

    elif experiment is not None and DataSet_size is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Train DataSet Size'] = DataSet_size[0]
        run_DF.at[index, 'Test DataSet Size'] = DataSet_size[1]
        run_DF.to_excel(run_file_name)

    elif experiment is not None and DataSet_Slide_magnification is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'Desired Slide Magnification'] = DataSet_Slide_magnification
        run_DF.to_excel(run_file_name)

    elif experiment is not None and test_mean_auc is not None:
        index = run_DF[run_DF['Experiment'] == experiment].index.values[0]
        run_DF.at[index, 'TestSet Mean AUC'] = test_mean_auc
        run_DF.to_excel(run_file_name)


    # In case we want to continue from a previous training session
    else:
        location = run_DF_exp.loc[[experiment], ['Location']].values[0][0]
        test_fold = int(run_DF_exp.loc[[experiment], ['Test Fold']].values[0][0])
        transformations = run_DF_exp.loc[[experiment], ['Transformations']].values[0][0] #RanS 9.12.20
        tile_size = int(run_DF_exp.loc[[experiment], ['Tile Size']].values[0][0])
        tiles_per_bag = int(run_DF_exp.loc[[experiment], ['Tiles Per Bag']].values[0][0])
        num_bags = int(run_DF_exp.loc[[experiment], ['No. of Bags']].values[0][0])
        DX = bool(run_DF_exp.loc[[experiment], ['DX']].values[0][0])
        DataSet_name = str(run_DF_exp.loc[[experiment], ['DataSet']].values[0][0])
        Receptor = str(run_DF_exp.loc[[experiment], ['Receptor']].values[0][0])
        MultiSlide = str(run_DF_exp.loc[[experiment], ['MultiSlide Per Bag']].values[0][0])
        model_name = str(run_DF_exp.loc[[experiment], ['Model']].values[0][0])
        desired_magnification = str(run_DF_exp.loc[[experiment], ['Desired Slide Magnification']].values[0][0])

        return location, test_fold, transformations, tile_size, tiles_per_bag, \
               num_bags, DX, DataSet_name, Receptor, MultiSlide, model_name, desired_magnification


def run_data_multi_model(experiments: List[str] = None, models: List[str] = None,
                         epoch: int = None,  transformation_string: str = None):
    num_experiments = len(experiments)
    if experiments is not None and transformation_string is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], transformation_string=transformation_string)
    elif experiments is not None and models is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], model=models[index])
    elif experiments is not None and epoch is not None:
        for index in range(num_experiments):
            run_data(experiment=experiments[index], epoch=epoch)




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

class MyMeanPixelRegularization:
    """replace patch with single pixel value"""

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if np.random.rand() < self.p:
            x = torch.zeros_like(x) + torch.tensor([[[0.87316266]], [[0.79902739]], [[0.84941472]]])
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


def define_transformations(transform_type, train, tile_size, color_param=0.1):

    MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
            'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
            'Ron': [0.8998, 0.8253, 0.9357]
            }

    STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
           'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
           'Ron': [0.1125, 0.1751, 0.0787]
           }

    # Setting the transformation:
    if transform_type == 'aug_receptornet':
        final_transform = transforms.Compose([transforms.Normalize(
                                                  mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                  std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))])
    else:
        final_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(
                                                  mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                  std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2]))
                                              ])
    scale_factor = 0.2
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
            transform1 = \
                transforms.Compose([
                    # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), #RanS 18.4.21, avoid the need to keep larger patches
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type in ['pcbnfrsc', 'pcbnfrs']:  # parameterized color, blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                    #transforms.ColorJitter(brightness=(1-c_param*1, 1+c_param*1), contrast=(1-c_param*2, 1+c_param*2),  # RanS 2.12.20
                    #                       saturation=c_param, hue=(-c_param, c_param)),
                    transforms.ColorJitter(brightness=color_param, contrast=color_param * 2, saturation=color_param, hue=color_param),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    #transforms.RandomAffine(degrees=0, scale=(1 - scale_factor, 1 + scale_factor)),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), #RanS 18.4.21, avoid the need to keep larger patches
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])

        elif transform_type == 'aug_receptornet':  #
        #elif transform_type == 'c_0_05_bnfrsc' or 'c_0_05_bnfrs':  # color 0.1, blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=64.0/255, contrast=0.75, saturation=0.25, hue=0.04),
                    transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                    #Mean Pixel Regularization
                    transforms.ToTensor(),
                    Cutout(n_holes=1, length=100),  # RanS 24.12.20
                    MyMeanPixelRegularization(p=0.75)
                ])

        elif transform_type == 'cbnfr':  # color, blur, noise, flip, rotate
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                           saturation=0.1, hue=(-0.1, 0.1)),
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type in ['bnfrsc', 'bnfrs']:  # blur, noise, flip, rotate, scale, +-cutout
            transform1 = \
                transforms.Compose([
                    transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)), #RanS 23.12.20
                    MyGaussianNoiseTransform(sigma=(0, 0.05)),  #RanS 23.12.20
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), #RanS 18.4.21, avoid the need to keep larger patches
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type == 'frs':  # flip, rotate, scale
            transform1 = \
                transforms.Compose([
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), #RanS 18.4.21, avoid the need to keep larger patches
                    transforms.CenterCrop(tile_size),  #fix boundary when scaling<1
                ])
        elif transform_type == 'hedcfrs':  # HED color, flip, rotate, scale
            transform1 = \
                transforms.Compose([
                    transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25)),
                    HEDColorJitter(sigma=0.05),
                    transforms.RandomVerticalFlip(),
                    #transforms.RandomHorizontalFlip(),
                    MyRotation(angles=[0, 90, 180, 270]),
                    transforms.RandomAffine(degrees=0, scale=(1, 1 + scale_factor)), #RanS 18.4.21, avoid the need to keep larger patches
                    transforms.CenterCrop(tile_size),  # fix boundary when scaling<1
                    #transforms.functional.crop(top=0, left=0, height=tile_size, width=tile_size)
                    # fix boundary when scaling<1
                ])

        transform = transforms.Compose([transform1, final_transform])
    else:
        transform = final_transform

    if transform_type in ['cbnfrsc', 'bnfrsc', 'c_0_05_bnfrsc', 'pcbnfrsc']:
        transform.transforms.append(Cutout(n_holes=1, length=100)) #RanS 24.12.20

    #RanS 14.1.21 - mean pixel regularization
    #if transform_type == 'aug_receptornet':
    #    transform.transforms.append(MyMeanPixelRegularization(p=0.75))
        #transform.transforms.append(transforms.RandomApply(torch.nn.ModuleList([MyMeanPixelRegularization]), p=0.75))

    return transform


def get_datasets_dir_dict(Dataset: str):
    dir_dict = {}
    TCGA_gipdeep_path = r'/home/womer/project/All Data/TCGA'
    ABCTB_gipdeep_path = r'/mnt/gipnetapp_public/sgils/Breast/ABCTB/ABCTB'
    HEROHE_gipdeep_path = r'/home/womer/project/All Data/HEROHE'
    SHEBA_gipdeep_path = r'/mnt/gipnetapp_public/sgils/Breast/Sheba/SHEBA'

    TCGA_gipdeep3_path = r'/mnt/hdd/All_Data/TCGA'
    HEROHE_gipdeep3_path = r'/mnt/hdd/All_Data/HEROHE'
    ABCTB_gipdeep3_path = r'/mnt/hdd/All_Data/ABCTB'

    TCGA_ran_path = r'C:\ran_data\TCGA_example_slides\TCGA_examples_131020_flat\TCGA'
    HEROHE_ran_path = r'C:\ran_data\HEROHE_examples'
    ABCTB_ran_path = r'C:\ran_data\ABCTB\ABCTB_examples\ABCTB'

    TCGA_omer_path = r'All Data/TCGA'
    HEROHE_omer_path = r'All Data/HEROHE'

    if Dataset == 'ABCTB_TCGA':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['TCGA'] = TCGA_gipdeep_path
            dir_dict['ABCTB'] = ABCTB_gipdeep_path
        elif sys.platform == 'win32':  # GIPdeep
            dir_dict['TCGA'] = TCGA_ran_path
            dir_dict['ABCTB'] = ABCTB_ran_path
    elif Dataset == 'Breast':
        if sys.platform == 'linux':  # GIPdeep
            for ii in np.arange(1, 4):
                dir_dict['CARMEL' + str(ii)] = r'/mnt/gipnetapp_public/sgils/BCF scans/Carmel Slides/Batch_' + str(ii)

            if platform.node() == 'gipdeep3':  # Run from local files
                dir_dict['TCGA'] = TCGA_gipdeep3_path
                dir_dict['HEROHE'] = HEROHE_gipdeep3_path
                dir_dict['ABCTB'] = ABCTB_gipdeep_path
            else:
                dir_dict['TCGA'] = TCGA_gipdeep_path
                dir_dict['HEROHE'] = HEROHE_gipdeep_path

        elif sys.platform == 'win32':  #Ran local
            dir_dict['TCGA'] = TCGA_ran_path
            dir_dict['HEROHE'] = HEROHE_ran_path

        elif sys.platform == 'darwin':   #Omer local
            dir_dict['TCGA'] = TCGA_omer_path
            dir_dict['HEROHE'] = HEROHE_omer_path

        else:
            raise Exception('Unrecognized platform')

    elif Dataset == 'TCGA':
        if sys.platform == 'linux':  # GIPdeep
            '''if platform.node() == 'gipdeep3':  # Run from local files
                dir_dict['TCGA'] = TCGA_gipdeep3_path
            else:  # Run from netapp'''
            dir_dict['TCGA'] = TCGA_gipdeep_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['TCGA'] = TCGA_ran_path

        elif sys.platform == 'darwin':  # Omer local
            dir_dict['TCGA'] = TCGA_omer_path

        else:
            raise Exception('Unrecognized platform')

    elif Dataset == 'HEROHE':
        if sys.platform == 'linux':  # GIPdeep
            '''if platform.node() == 'gipdeep3':  # Run from local files
                dir_dict['HEROHE'] = HEROHE_gipdeep3_path
            else:  # Run from netapp'''
            dir_dict['HEROHE'] = HEROHE_gipdeep_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['HEROHE'] = HEROHE_ran_path

        elif sys.platform == 'darwin':  # Omer local
            dir_dict['HEROHE'] = HEROHE_omer_path

    elif Dataset == 'ABCTB_TIF':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ABCTB_TIF'] = r'/home/womer/project/All Data/ABCTB_TIF'
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['ABCTB_TIF'] = r'All Data/ABCTB_TIF'

    elif Dataset == 'ABCTB_TILES':
        if sys.platform == 'linux':  # GIPdeep
            dir_dict['ABCTB_TILES'] = r'/home/womer/project/All Data/ABCTB_TILES'
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['ABCTB_TILES'] = r'All Data/ABCTB_TILES'

    elif Dataset == 'ABCTB':
        if sys.platform == 'linux':  # GIPdeep Run from local files
            dir_dict['ABCTB'] = ABCTB_gipdeep_path #non local, temp RanS 28.4.21
            #dir_dict['ABCTB'] = ABCTB_gipdeep3_path

        elif sys.platform == 'win32':  # Ran local
            dir_dict['ABCTB'] = ABCTB_ran_path
        #else:
        #    raise Exception('ABCTB can be used only on gipdeep3')

    elif Dataset == 'SHEBA':
        if sys.platform == 'linux':
            dir_dict['SHEBA'] = SHEBA_gipdeep_path

    elif Dataset == 'LUNG':
        if sys.platform == 'linux':
            dir_dict['LUNG'] = r'/home/rschley/All_Data/LUNG/LUNG'
        elif sys.platform == 'win32':  # Ran local
            dir_dict['LUNG'] = r'C:\ran_data\Lung_examples\LUNG'
        elif sys.platform == 'darwin':  # Omer local
            dir_dict['LUNG'] = 'All Data/LUNG'

    elif Dataset == 'PDL1':
        if sys.platform == 'linux':
            dir_dict['PDL1'] = r'/mnt/gipnetapp_public/sgils/LUNG/PDL1'
        elif sys.platform == 'win32':  # Ran local
            dir_dict['PDL1'] = r'C:\ran_data\IHC_examples\PDL1'

    return dir_dict


def assert_dataset_target(DataSet, target_kind):
    if DataSet == 'LUNG' and target_kind not in ['PDL1', 'EGFR']:
        raise ValueError('For LUNG DataSet, target should be one of: PDL1, EGFR')
    elif DataSet == 'HEROHE' and target_kind != 'Her2':
        raise ValueError('for HEROHE DataSet, target should be Her2')
    elif (DataSet == 'TCGA' or DataSet[:6] == 'CARMEL' or DataSet == 'Breast') and target_kind not in ['ER', 'PR', 'Her2']:
        raise ValueError('target should be one of: ER, PR, Her2')
    elif (DataSet == 'RedSquares') and target_kind != 'RedSquares':
        raise ValueError('target should be: RedSquares')
    elif DataSet == 'Breast' and target_kind != 'Her2':
        raise ValueError('HEROHE is part of DataSet Breast so target must be Her2 ')
    elif DataSet == 'SHEBA' and target_kind != 'Onco':
        raise ValueError('for SHEBA DataSet, target should be Onco')

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
            #transforms.ColorJitter(brightness=64.0/255, contrast=0.75, saturation=0.25, hue=0.04),  # RanS 13.1.21
            transforms.ToTensor()])

        '''blur_trans = transforms.Compose([
            transforms.GaussianBlur(5, sigma=0.1),  # RanS 23.12.20
            transforms.ToTensor()])

        noise_trans = transforms.Compose([
            MyGaussianNoiseTransform(sigma=(0.05, 0.05)),  # RanS 23.12.20
            transforms.ToTensor()])

        cutout_trans = transforms.Compose([
            transforms.ToTensor(),
            Cutout(n_holes=1, length=100)])  # RanS 24.12.20]'''

        img5 = color_trans(tiles[ii])
        # img5 = blur_trans(tiles[ii])
        # img5 = noise_trans(tiles[ii])
        #img5 = cutout_trans(tiles[ii])
        grid5[ii].imshow(np.transpose(img5, axes=(1, 2, 0)))

    fig1.suptitle('original patches', fontsize=14)
    fig2.suptitle('final patches', fontsize=14)
    fig3.suptitle('all trans before norm', fontsize=14)
    fig4.suptitle('original patches, before crop', fontsize=14)
    fig5.suptitle('color transform only', fontsize=14)

    plt.show()


def get_model(model_name, saved_model_path='none'):
    #if train_type == 'MIL':
    # MIL models
    if model_name == 'resnet50_gn':
        model = ResNet50_GN_GatedAttention()
    elif model_name == 'receptornet':
        model = ReceptorNet('resnet50_2FC', saved_model_path)
    elif model_name == 'receptornet_preact_resnet50':
        model = ReceptorNet('preact_resnet50', saved_model_path)
    #elif train_type == 'REG':

    #REG models
    elif model_name == 'resnet50_3FC':
        model = nets.resnet50_with_3FC()
    elif model_name == 'preact_resnet50':
        model = nets.PreActResNet50()
    elif model_name == 'resnet50_gn':
        model = nets.ResNet50_GN()
    elif model_name == 'resnet18':
        model = nets.ResNet_18()
    elif model_name == 'resnet50':
        model = nets.ResNet_50()
    else:
        print('model not defined!')
    return model


def save_code_files(args: argsNamespace, train_DataSet):
    """
    This function saves the code files and argparse data to a Code directory within the run path.
    :param args: argsparse Namespace of the run.
    :return:
    """
    code_files_path = os.path.join(args.output_dir, 'Code')
    args.run_file = os.path.basename(__file__)
    args_dict = vars(args)

    # Add Grid Data:
    data_dict = args_dict
    # grid_meta_data_file = os.path.join(train_DataSet.ROOT_PATH, train_DataSet.DataSet, 'Grids', 'production_meta_data.xlsx')
    for _, key in enumerate(train_DataSet.dir_dict):
        grid_meta_data_file = os.path.join(train_DataSet.dir_dict[key], 'Grids', 'production_meta_data.xlsx')
        if os.path.isfile(grid_meta_data_file):
            grid_data_DF = pd.read_excel(grid_meta_data_file)
            grid_dict = grid_data_DF.to_dict('split')
            grid_dict['dataset'] = key
            grid_dict.pop('index')
            grid_dict.pop('columns')
            data_dict[key + '_grid'] = grid_dict
            #data_dict = {**args_dict, **grid_dict}
    #else:
    #    data_dict = args_dict

    data_DF = pd.DataFrame([data_dict]).transpose()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        os.mkdir(code_files_path)
    data_DF.to_excel(os.path.join(code_files_path, 'run_arguments.xlsx'))
    # Get all .py files in the code path:
    py_files = glob.glob('*.py')
    for _, file in enumerate(py_files):
        copy2(file, code_files_path)
