# python core
import argparse
import os

# numpy
import numpy as np

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

parser = argparse.ArgumentParser(description='WSI Tuples Generator')
parser.add_argument('--inner-radius', type=int, default=2)
parser.add_argument('--outer-radius', type=int, default=10)
parser.add_argument('--test-fold', type=int, default=0)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--tile-size', type=int, default=256)
parser.add_argument('--desired-magnification', type=int, default=10)
parser.add_argument('--metadata-file-path', type=str, default=None)
parser.add_argument('--metadata-enhancement-dir-path', type=str, default=None)
parser.add_argument('--datasets-base-dir-path', type=str, default=None)
parser.add_argument('--dataset-ids', nargs='+', default=['TCGA'])
parser.add_argument('--minimal-tiles-count', type=int, default=10)
parser.add_argument('--folds-count', type=int, default=6)
parser.add_argument('--num-workers', type=int, default=5)
parser.add_argument('--negative-examples-count', type=int, default=2)
parser.add_argument('--tuples-count', type=int, default=1000)
parser.add_argument('--tuples-dir-path', type=str)
parser.add_argument('--dump-dir-path', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    np.random.seed(30)

    train_tuples_generator = datasets.WSITuplesGenerator(
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        metadata_file_path=args.metadata_file_path,
        datasets_base_dir_path=args.datasets_base_dir_path,
        dataset_ids=args.dataset_ids,
        minimal_tiles_count=args.minimal_tiles_count,
        metadata_enhancement_dir_path=args.metadata_enhancement_dir_path,
        folds_count=args.folds_count)

    folds = list(range(train_tuples_generator.get_folds_count()))
    train_folds = [fold for fold in folds if fold != args.test_fold]
    test_folds = [args.test_fold]
    train_tuples_generator.create_tuples(
        tuples_count=args.tuples_count,
        negative_examples_count=args.negative_examples_count,
        folds=train_folds,
        dir_path=args.tuples_dir_path,
        num_workers=args.num_workers,
        dump_dir_path=args.dump_dir_path)
