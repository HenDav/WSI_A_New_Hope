import numpy as np
from PIL import Image
import os
import pandas as pd
from typing import List, Tuple
import openslide
import torch
import sys
import os
import multiprocessing
Image.MAX_IMAGE_PIXELS = None


def chunks(list: List, length: int):
    new_list = [ list[i * length:(i + 1) * length] for i in range((len(list) + length - 1) // length )]
    return new_list


def list_subdirectories(base_dir='.'):
    result = []
    for current_sub_dir in os.listdir(base_dir):
        full_sub_dir_path = os.path.join(base_dir, current_sub_dir)
        if os.path.isdir(full_sub_dir_path):
            result.append(full_sub_dir_path)

    return result


def get_latest_subdirectory(base_dir='.'):
    subdirectories = list_subdirectories(base_dir=base_dir)
    return os.path.normpath(max(subdirectories, key=os.path.getmtime))


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def get_cpu_count():
    return multiprocessing.cpu_count()


def calculate_box_aspect_ratio(top_left, bottom_right):
    box = (bottom_right - top_left) + 1
    return box[0] / box[1]


def to_int(x):
    try:
        return int(x)
    except ValueError:
        return 0


def add_to_patient_dict(patient_dict, patient_barcode, x):
    if patient_barcode not in patient_dict:
        patient_dict[patient_barcode] = [x]
    else:
        patient_dict[patient_barcode].append(x)
