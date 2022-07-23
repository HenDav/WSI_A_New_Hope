# python core
import argparse

# wsi
import utils
from nn import datasets
from nn import trainers
from nn import losses
from nn import networks

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch
import torchvision

# numpy
import numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inner-radius', type=int, default=2)
    parser.add_argument('--outer-radius', type=int, default=10)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--desired-magnification', type=int, default=10)
    parser.add_argument('--metadata-file-path', type=str, default=None)
    parser.add_argument('--metadata-enhancement-dir-path', type=str, default=None)
    parser.add_argument('--datasets-base-dir-path', type=str, default=None)
    parser.add_argument('--results-base-dir-path', type=str, default=None)
    parser.add_argument('--dataset-ids', nargs='+', default=['TCGA'])
    parser.add_argument('--minimal-tiles-count', type=int, default=10)
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--workers-count', type=int, default=5)
    parser.add_argument('--negative-examples-count', type=int, default=2)

    parser.add_argument('--validation-dataset-size', type=int, default=1000)
    parser.add_argument('--validation-queue-size', type=int, default=100)
    parser.add_argument('--train-dataset-size', type=int, default=1000)
    parser.add_argument('--train-queue-size', type=int, default=100)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=int, default=0.0001)

    parser.add_argument('--tuplets-dir-path', type=str)
    parser.add_argument('--dump-dir-path', type=str, default=None)
    args = parser.parse_args()

    train_tuplets_generator = datasets.WSITupletsGenerator(
        folds=args.folds,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        dataset_size=args.train_dataset_size,
        dataset_ids=args.dataset_ids,
        datasets_base_dir_path=args.datasets_base_dir_path,
        dump_dir_path=args.dump_dir_path,
        metadata_enhancement_dir_path=args.metadata_enhancement_dir_path,
        minimal_tiles_count=args.minimal_tiles_count)

    validation_tuplets_generator = datasets.WSITupletsGenerator(
        folds=args.folds,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        tile_size=args.tile_size,
        desired_magnification=args.desired_magnification,
        dataset_size=args.validation_dataset_size,
        dataset_ids=args.dataset_ids,
        datasets_base_dir_path=args.datasets_base_dir_path,
        dump_dir_path=args.dump_dir_path,
        metadata_enhancement_dir_path=args.metadata_enhancement_dir_path,
        minimal_tiles_count=args.minimal_tiles_count)

    validation_tuplets_generator.start_tuplets_creation(
        negative_examples_count=args.negative_examples_count,
        workers_count=args.workers_count,
        queue_size=args.validation_queue_size)

    validation_tuplets_generator.stop_tuplets_creation()

    train_tuplets_generator.start_tuplets_creation(
        negative_examples_count=args.negative_examples_count,
        workers_count=args.workers_count,
        queue_size=args.train_queue_size)

    train_dataset = datasets.WSITuplesOnlineDataset(tuplets_generator=train_tuplets_generator, replace=True)
    validation_dataset = datasets.WSITuplesOnlineDataset(tuplets_generator=validation_tuplets_generator, replace=False)

    model = networks.WSIBYOL()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = losses.BYOLLoss()

    model_trainer = trainers.WSIModelTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer)

    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        results_base_dir_path=args.results_base_dir_path)

