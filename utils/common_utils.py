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


def split(items, n):
    k, m = divmod(len(items), n)
    split_items = list(items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return split_items


def chunks(items, length):
    chucked_items = []
    for i in range(0, len(items), length):
        chucked_items.append(items[i:i + length])
    return chucked_items


def even_chunks(items, length):
    chucked_items = [items[i * length:(i + 1) * length] for i in range((len(items) + length - 1) // length)]
    return chucked_items


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


def save_object_dict(obj, file_path):
    dict = {key: value for key, value in obj.__dict__.items() if isinstance(value, str) or isinstance(value, int) or isinstance(value, float)}
    with open(file_path, "w") as text_file:
        for key, value in dict.items():
            text_file.write(f'{key}: {value}\n')


def magnification_to_mpp(magnification):
    return (40 / magnification) * 0.25