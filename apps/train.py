# python core
import argparse
import os
from datetime import datetime
from pathlib import Path
from distutils.dir_util import copy_tree
import shutil
import multiprocessing
import logging

# wsi
from nn import datasets
from nn import trainers
from nn import losses
from nn import networks
from utils import common_utils

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# pytorch
import torch
import torchvision

# numpy
import numpy


def create_results_dir(results_base_dir_path):
    results_dir_path = os.path.normpath(os.path.join(results_base_dir_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    Path(results_dir_path).mkdir(parents=True, exist_ok=True)
    return results_dir_path


def setup_logging(results_dir_path):
    logging_file_path = os.path.normpath(os.path.join(results_dir_path, 'log.txt'))

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logging_file_path,
                        filemode='w')

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')

    # tell the handler to use this format
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def backup_codebase(results_dir_path, logger):
    codebase_dest_dir_path = os.path.normpath(os.path.join(results_dir_path, 'code'))
    common_utils.save_object_dict(obj=args, file_path=os.path.join(results_dir_path, 'args.txt'))
    shutil.copytree(src=args.codebase_dir_path, dst=codebase_dest_dir_path, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))
    logger.info(f'Codebase copied to: {codebase_dest_dir_path}')


def create_tuplets_generator(args, validation_fold, logger):
    train_folds = [train_fold for train_fold in args.folds if train_fold != validation_fold]
    validation_folds = [validation_fold]

    logger.info('Creating train tuplets generator')
    train_tuplets_generator = datasets.WSITupletsGenerator(
        folds=train_folds,
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

    logger.info('Creating validation tuplets generator')
    validation_tuplets_generator = datasets.WSITupletsGenerator(
        folds=validation_folds,
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

    return train_tuplets_generator, validation_tuplets_generator


def generate_validation_data(args, validation_tuplets_generator, logger):
    logger.info('Starting generation of validation tuplets')
    validation_tuplets_generator.start_tuplets_creation(
        negative_examples_count=args.negative_examples_count,
        workers_count=args.workers_count,
        queue_size=args.validation_queue_size)

    logger.info('Stopping generation of validation tuplets')
    validation_tuplets_generator.stop_tuplets_creation()


def generate_train_data(args, train_tuplets_generator, logger):
    logger.info('Starting generation of train tuplets')
    train_tuplets_generator.start_tuplets_creation(
        negative_examples_count=args.negative_examples_count,
        workers_count=args.workers_count,
        queue_size=args.train_queue_size)


def create_datasets(args, train_tuplets_generator, validation_tuplets_generator, logger):
    logger.info('Creating train and validation datasets')
    train_dataset = datasets.WSITuplesOnlineDataset(tuplets_generator=train_tuplets_generator, replace=True)
    validation_dataset = datasets.WSITuplesOnlineDataset(tuplets_generator=validation_tuplets_generator, replace=False)
    return train_dataset, validation_dataset


def create_model(args, logger):
    logger.info('Creating model')
    model = networks.WSIBYOL()
    if torch.cuda.device_count() > 1:
        logger.info(f'Wrapping model with torch.nn.DataParallel (torch.cuda.device_count(): {torch.cuda.device_count()})')
        model = torch.nn.DataParallel(model)
    return model


def create_optimizer(args, logger):
    logger.info(f'Creating optimizer (learning rate: {args.learning_rate}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    return optimizer


def create_loss_function(args, logger):
    logger.info(f'Creating loss function')
    loss_fn = losses.BYOLLoss()
    return loss_fn


def create_model_trainer(args, model, optimizer, loss_fn, logger):
    logger.info(f'Creating model trainer')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_trainer = trainers.WSIModelTrainer(
        model=model,
        loss_function=loss_fn,
        optimizer=optimizer,
        validation_rate=args.validation_rate,
        device=device)

    return model_trainer


def train_folds(args, validation_fold, logger):
    train_tuplets_generator, validation_tuplets_generator = create_tuplets_generator(
        args=args,
        validation_fold=validation_fold,
        logger=logger)

    generate_validation_data(
        args=args,
        validation_tuplets_generator=validation_tuplets_generator,
        logger=logger)

    generate_train_data(
        args=args,
        train_tuplets_generator=train_tuplets_generator,
        logger=logger)

    train_dataset, validation_dataset = create_datasets(
        args=args,
        train_tuplets_generator=train_tuplets_generator,
        validation_tuplets_generator=validation_tuplets_generator,
        logger=logger)

    model = create_model(
        args=args,
        logger=logger)

    optimizer = create_optimizer(
        args=args,
        logger=logger)

    loss_fn = create_loss_function(
        args=args,
        logger=logger)

    model_trainer = create_model_trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        logger=logger)

    model_trainer.plot_samples(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        batch_size=args.batch_size)

    model_trainer.fit(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        results_dir_path=fold_results_dir_path)


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--inner-radius', type=int, default=2)
    parser.add_argument('--outer-radius', type=int, default=10)
    parser.add_argument('--tile-size', type=int, default=256)
    parser.add_argument('--desired-magnification', type=int, default=10)
    parser.add_argument('--dataset-ids', nargs='+', default=['TCGA'])
    parser.add_argument('--minimal-tiles-count', type=int, default=10)
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--workers-count', type=int, default=5)
    parser.add_argument('--negative-examples-count', type=int, default=2)

    parser.add_argument('--metadata-file-path', type=str, default=None)
    parser.add_argument('--metadata-enhancement-dir-path', type=str, default=None)
    parser.add_argument('--datasets-base-dir-path', type=str, default=None)
    parser.add_argument('--results-base-dir-path', type=str, default=None)
    parser.add_argument('--codebase-dir-path', type=str, default=None)

    parser.add_argument('--validation-dataset-size', type=int, default=1000)
    parser.add_argument('--validation-queue-size', type=int, default=100)
    parser.add_argument('--train-dataset-size', type=int, default=1000)
    parser.add_argument('--train-queue-size', type=int, default=100)

    parser.add_argument('--validation-rate', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.0001)

    parser.add_argument('--tuplets-dir-path', type=str)
    parser.add_argument('--dump-dir-path', type=str, default=None)
    args = parser.parse_args()

    results_dir_path = create_results_dir(results_base_dir_path=args.results_base_dir_path)
    setup_logging(results_dir_path=results_dir_path)
    logger = logging.getLogger('train.py')
    backup_codebase(results_dir_path=results_dir_path, logger=logger)

    logger.info(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    logger.info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    logger.info(f'torch.version.cuda: {torch.version.cuda}')

    for validation_fold in args.folds:
        train_folds(
            args=args,
            validation_fold=validation_fold,
            logger=logger)
