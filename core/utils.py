# python peripherals
import logging
from typing import List
import os
import multiprocessing

# torch
import torch

# numpy
import numpy


#################################
#           Logging             #
#################################
def create_log_file_path(file_name: str, results_dir_path: str) -> str:
    return os.path.normpath(os.path.join(results_dir_path, f'{file_name}.log'))


# https://stackoverflow.com/questions/22934616/multi-line-logging-in-python
def create_logger(log_file_path: str, name: str, level: int) -> logging.Logger:
    fmt = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    datefmt = '%m-%d %H:%M'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler = logging.FileHandler(filename=log_file_path)
    handler.setFormatter(fmt=formatter)

    console = logging.StreamHandler()
    console.setLevel(level=level)
    console.setFormatter(fmt=formatter)

    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    logger.addHandler(hdlr=handler)
    logger.addHandler(hdlr=console)

    return logger


###########################################
#           Directory Listing             #
###########################################
def list_subdirectories(base_dir: str = '.') -> List[str]:
    result = []
    for current_sub_dir in os.listdir(base_dir):
        full_sub_dir_path = os.path.join(base_dir, current_sub_dir)
        if os.path.isdir(full_sub_dir_path):
            result.append(full_sub_dir_path)

    return result


def get_latest_subdirectory(base_dir: str = '.') -> str:
    subdirectories = list_subdirectories(base_dir=base_dir)
    return os.path.normpath(max(subdirectories, key=os.path.getmtime))


#######################################
#           Text Formatting           #
#######################################
def generate_title_text(text: str) -> str:
    text_len = len(text)
    decoration_len = text_len * 2
    space_len = decoration_len - 2 - text_len
    if space_len % 2 != 0:
        decoration_len = decoration_len + 1
        space_len = space_len + 1

    half_space_len = space_len // 2
    title = '#' * decoration_len
    title = title + f"{'#'}{' ' * half_space_len}{text.capitalize()}{' ' * half_space_len}{'#'}"
    title = title + '#' * decoration_len
    return title


def generate_bullet_text(text: str, indentation: int) -> str:
    tab = '\t'
    return f"{tab * indentation} - {text}"


def generate_captioned_bullet_text(text: str, value: object, indentation: int, padding: int) -> str:
    tab = '\t'
    text = text + ':'
    return f'{tab * indentation} - {text.capitalize():{" "}{"<"}{padding}}{value}'


def generate_serialized_object_text(text: str, obj: object) -> str:
    return generate_title_text(text=text) + '\n' + f'{obj}'


def generate_batch_loss_text(epoch_index: int, batch_index: int, batch_loss: float, average_batch_loss: float, index_padding: int, loss_padding: int, batch_count: int, batch_duration: float, indentation: int):
    tab = '\t'
    fill = " "
    align = "<"
    return f'{tab*indentation} - [Epoch {epoch_index:{fill}{align}{index_padding}} | Batch {batch_index:{fill}{align}{index_padding}} / {batch_count}]: Batch Loss = {batch_loss:{fill}{align}{loss_padding}}, Avg. Batch Loss = {average_batch_loss:{fill}{align}{loss_padding}}, Batch Duration: {batch_duration} sec.'


#######################################
#           Miscellaneous             #
#######################################
def calculate_batches_per_epoch(dataset_size: int, batch_size: int) -> int:
    return int(numpy.ceil(dataset_size / batch_size))

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def get_cpu_count() -> int:
    return multiprocessing.cpu_count()


def to_int(x: str) -> int:
    try:
        return int(x)
    except ValueError:
        return 0


def magnification_to_mpp(magnification: int) -> float:
    return (40 / magnification) * 0.25


def save_object_dict(obj: object, file_path: str):
    object_dict = {key: value for key, value in obj.__dict__.items() if isinstance(value, str) or isinstance(value, int) or isinstance(value, float)}
    with open(file_path, "w") as text_file:
        for key, value in object_dict.items():
            text_file.write(f'{key}: {value}\n')