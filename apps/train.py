# python core
import argparse

# wsi
import utils
from nn import datasets
from nn import trainers
from nn import losses

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch
import torchvision

parser = argparse.ArgumentParser(description='WSI Train')
# parser.add_argument('--test_fold', type=int, default=1)
# parser.add_argument('--folds_count', type=int, default=6)
# parser.add_argument('--epochs', type=int, default=400)
# parser.add_argument('--dataset_name', type=str, default='TCGA')
# parser.add_argument('--tile_size', type=int, default=256)
# parser.add_argument('--desired_magnification', type=int, default=10)
# parser.add_argument('--minimal_tiles_count', type=int, default=10)
# parser.add_argument('--lr', type=float, default=1e-5)
# parser.add_argument('--batch_size', default=48, type=int)
# parser.add_argument('--train_dataset_size', default=10000, type=int)
# parser.add_argument('--train_buffer_size', default=10000, type=int)
# parser.add_argument('--train_max_size', default=5000, type=int)
# parser.add_argument('--validation_dataset_size', default=5000, type=int)
# parser.add_argument('--validation_buffer_size', default=5000, type=int)
# parser.add_argument('--validation_max_size', default=5000, type=int)
# parser.add_argument('--datasets_base_dir_path', type=str, default=None)
# parser.add_argument('--results_base_dir_path', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = datasets.WSITuplesDataset(dir_path='./tuples')
    for item in train_dataset:
        h = 5
