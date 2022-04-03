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

parser = argparse.ArgumentParser(description='WSI Project')
parser.add_argument('--test_fold', type=int, default=1)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--dataset_name', type=str, default='TCGA')
parser.add_argument('--tile_size', type=int, default=256)
parser.add_argument('--desired_magnification', type=int, default=10)
parser.add_argument('--minimal_tiles_count', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', default=48, type=int)
parser.add_argument('--train_dataset_size', default=10000, type=int)
parser.add_argument('--train_buffer_size', default=10000, type=int)
parser.add_argument('--train_max_size', default=5000, type=int)
parser.add_argument('--validation_dataset_size', default=5000, type=int)
parser.add_argument('--validation_buffer_size', default=5000, type=int)
parser.add_argument('--validation_max_size', default=5000, type=int)
parser.add_argument('--datasets_base_dir_path', type=str, default=None)
parser.add_argument('--results_base_dir_path', type=str, default=None)
args = parser.parse_args()

if __name__ == '__main__':
    device = utils.get_device()
    cpu_count = utils.get_cpu_count()

    train_dataset = datasets.WSIDistanceDataset(
        dataset_size=args.train_dataset_size,
        buffer_size=args.train_buffer_size,
        max_size=args.train_max_size,
        replace=True,
        num_workers=10,
        dataset_name=args.dataset_name,
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        minimal_tiles_count=args.minimal_tiles_count,
        test_fold=args.test_fold,
        train=True,
        datasets_base_dir_path=args.datasets_base_dir_path,
        inner_radius=2,
        outer_radius=10)

    validation_dataset = datasets.WSIDistanceDataset(
        dataset_size=args.validation_dataset_size,
        buffer_size=args.validation_buffer_size,
        max_size=args.validation_max_size,
        replace=False,
        num_workers=10,
        dataset_name=args.dataset_name,
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        minimal_tiles_count=args.minimal_tiles_count,
        test_fold=args.test_fold,
        train=False,
        datasets_base_dir_path=args.datasets_base_dir_path,
        inner_radius=2,
        outer_radius=10)

    validation_dataset.start(load_buffer=False)
    validation_dataset.stop()
    train_dataset.start(load_buffer=False)
    train_dataset.stop()

    # model = torchvision.models.resnet50()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # loss_fn = losses.TupletLoss()
    # model_trainer = trainers.WSIDistanceModelTrainer(model=model, loss_functions=[loss_fn], optimizer=optimizer)
    # model_trainer.fit(
    #     train_dataset=train_dataset,
    #     validation_dataset=validation_dataset,
    #     epochs=args.epochs,
    #     train_batch_size=args.batch_size,
    #     validation_batch_size=args.batch_size,
    #     validation_split=None,
    #     results_base_dir_path=args.results_base_dir_path)
    #
    # train_dataset.stop()

