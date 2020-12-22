from torchvision import transforms
import random
import torch
import numpy as np

"""
This file defines all transforms to be used by datasets.py
"""


class MyRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


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


MEAN = {'TCGA': [58.2069073 / 255, 96.22645279 / 255, 70.26442606 / 255],
        'HEROHE': [224.46091564 / 255, 190.67338568 / 255, 218.47883547 / 255],
        'Ron': [0.8998, 0.8253, 0.9357]
        }

STD = {'TCGA': [40.40400300279664 / 255, 58.90625962739444 / 255, 45.09334057330417 / 255],
       'HEROHE': [np.sqrt(1110.25292532) / 255, np.sqrt(2950.9804851) / 255, np.sqrt(1027.10911208) / 255],
       'Ron': [0.1125, 0.1751, 0.0787]
       }


BASIC_TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize( mean=(MEAN['Ron'][0], MEAN['Ron'][1], MEAN['Ron'][2]),
                                                            std=(STD['Ron'][0], STD['Ron'][1], STD['Ron'][2])
                                                            )
                                      ])

FLIP_1 = transforms.Compose([ transforms.RandomHorizontalFlip() ])

FLIP_2 = transforms.Compose([transforms.RandomHorizontalFlip(),
                             transforms.RandomVerticalFlip
                             ])

FLIP_ROTATE = transforms.Compose([transforms.RandomHorizontalFlip(),
                                  MyRotation(angles=[0, 90, 180, 270])
                             ])


            if transform_type == 'flip':
                self.scale_factor = 0
                transform1 = \
                    transforms.Compose([ transforms.RandomVerticalFlip(),
                                         transforms.RandomHorizontalFlip()])
            elif transform_type == 'wcfrs': #weak color, flip, rotate, scale
                self.scale_factor = 0.2
                transform1 = \
                    transforms.Compose([
                        # transforms.ColorJitter(brightness=(0.65, 1.35), contrast=(0.5, 1.5),
                        transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25),  # RanS 2.12.20
                                               saturation=0.1, hue=(-0.1, 0.1)),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        MyRotation(angles=[0, 90, 180, 270]),
                        transforms.RandomAffine(degrees=0, scale=(1 - self.scale_factor, 1 + self.scale_factor)),
                        #transforms.CenterCrop(self.tile_size),  #fix boundary when scaling<1
                        transforms.functional.crop(top=0, left=0, height=self.tile_size, width=self.tile_size)  # fix boundary when scaling<1
                    ])
            elif transform_type == 'hedcfrs':  # HED color, flip, rotate, scale
                self.scale_factor = 0.2
                transform1 = \
                    transforms.Compose([
                        transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25)),
                        HEDColorJitter(sigma=0.05),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomHorizontalFlip(),
                        MyRotation(angles=[0, 90, 180, 270]),
                        transforms.RandomAffine(degrees=0, scale=(1 - self.scale_factor, 1 + self.scale_factor)),
                        #transforms.CenterCrop(self.tile_size),  # fix boundary when scaling<1
                        transforms.functional.crop(top=0, left=0, height=self.tile_size, width=self.tile_size)  # fix boundary when scaling<1
                    ])


            self.transform = transforms.Compose([transform1,
                                                 final_transform])
        else:
            self.scale_factor = 0
            self.transform = final_transform
def get_transform(transform_name: str = 'flip'):
    if transform_name == 'flip':
        full_transform = transforms.Compose([ FLIP_2,
                                              BASIC_TRANSFORM
                                              ])
    pass





