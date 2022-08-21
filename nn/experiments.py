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
from nn.networks import *
from nn.losses import *
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


class Experiment:
    def __init__(self, args):
        self._args = args
        self._results_dir_path = self._create_results_dir_path()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._setup_logging()
        self._logger = logging.getLogger(name=self.__class__.__name__)
        self._logger.info(f'results_dir_path: {self._results_dir_path}')
        self._backup_codebase()

        self._logger.info(f'Initializing train data:')
        self._train_tuplets_generator = self._create_tuplets_generator()
        self._train_tuplets_dataset = self._create_tuplets_dataset(tuplets_generator=self._train_tuplets_generator)

        self._logger.info(f'Initializing validation data:')
        self._validation_tuplets_generator = self._create_tuplets_generator()
        self._validation_tuplets_dataset = self._create_tuplets_dataset(tuplets_generator=self._validation_tuplets_generator)

        self._logger.info(f'Initializing pytorch environment:')
        self._model = self._create_model()
        self._optimizer = self._create_optimizer()
        self._loss_fn = self._create_loss_function()
        self._model_trainer = self._create_model_trainer()

    def _create_results_dir_path(self):
        results_dir_path = os.path.normpath(os.path.join(self._args.results_base_dir_path, self._args.experiment_name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        return results_dir_path

    def _setup_logging(self):
        logging_file_path = os.path.normpath(os.path.join(self._results_dir_path, 'log.txt'))

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

    def _backup_codebase(self):
        codebase_dest_dir_path = os.path.normpath(os.path.join(self._results_dir_path, 'code'))
        self._logger.info(f'codebase_dest_dir_path: {codebase_dest_dir_path}')
        common_utils.save_object_dict(obj=self._args, file_path=os.path.join(self._results_dir_path, 'args.txt'))
        shutil.copytree(src=self._args.codebase_dir_path, dst=codebase_dest_dir_path, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))

    def _create_tuplets_generator(self):
        self._logger.info('      - Creating tuplets generator')
        tuplets_generator = datasets.WSITupletsGenerator(
            inner_radius_mm=self._args.inner_radius,
            outer_radius_mm=self._args.outer_radius,
            tile_size=self._args.tile_size,
            desired_magnification=self._args.desired_magnification,
            dataset_size=self._args.train_dataset_size,
            dataset_ids=self._args.dataset_ids,
            datasets_base_dir_path=self._args.datasets_base_dir_path,
            dump_dir_path=self._args.dump_dir_path,
            metadata_enhancement_dir_path=self._args.metadata_enhancement_dir_path,
            minimal_tiles_count=self._args.minimal_tiles_count)
        return tuplets_generator

    def _generate_validation_data(self, folds):
        self._logger.info('Starting generation of validation tuplets')
        self._validation_tuplets_generator.start_tuplets_creation(
            negative_examples_count=self._args.negative_examples_count,
            folds=folds,
            workers_count=self._args.workers_count,
            queue_size=self._args.validation_queue_size)

        self._logger.info('Stopping generation of validation tuplets')
        self._validation_tuplets_generator.stop_tuplets_creation()

    def _generate_train_data(self, folds):
        self._logger.info('Starting generation of train tuplets')
        self._train_tuplets_generator.start_tuplets_creation(
            negative_examples_count=self._args.negative_examples_count,
            folds=folds,
            workers_count=self._args.workers_count,
            queue_size=self._args.train_queue_size)

    def _create_tuplets_dataset(self, tuplets_generator):
        self._logger.info('      - Creating tuplets dataset')
        tuplets_dataset = datasets.WSITupletsOnlineDataset(tuplets_generator=tuplets_generator)
        return tuplets_dataset

    def _create_model(self):
        self._logger.info('      - Creating model')

        try:
            FeatureExtractorClass = globals()[self._args.feature_extractor_model_class]
        except Exception:
            FeatureExtractorClass = None

        try:
            ClassifierClass = globals()[self._args.classifier_model_class]
        except Exception:
            ClassifierClass = None

        if FeatureExtractorClass is not None and ClassifierClass is not None:
            model = CompoundClassifier(
                FeatureExtractorClass=FeatureExtractorClass,
                ClassifierClass=ClassifierClass)
        elif FeatureExtractorClass is not None:
            model = FeatureExtractorClass()
        else:
            raise f'Unknown model pairing: {self._args.feature_extractor_model_class}, {self._args.classifier_model_class}'

        if torch.cuda.device_count() > 1:
            self._logger.info(f'      - Wrapping model with torch.nn.DataParallel (torch.cuda.device_count(): {torch.cuda.device_count()})')
            model = torch.nn.DataParallel(model)

        self._logger.info(model)

        return model

    def _create_optimizer(self):
        self._logger.info(f'      - Creating optimizer:')
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._args.learning_rate)
        self._logger.info(optimizer)
        return optimizer

    def _create_loss_function(self):
        self._logger.info(f'      - Creating loss function:')

        try:
            LossFunctionClass = globals()[self._args.loss_function_class]
        except Exception:
            raise f'Unknown loss function: {self._args.loss_function_class}'

        loss_fn = LossFunctionClass()

        self._logger.info(loss_fn)

        return loss_fn

    def _create_model_trainer(self):
        self._logger.info(f'      - Creating model trainer')
        model_trainer = trainers.WSIModelTrainer(
            model=self._model,
            loss_function=self._loss_fn,
            optimizer=self._optimizer,
            train_dataset=self._train_tuplets_dataset,
            validation_dataset=self._validation_tuplets_dataset,
            epochs=self._args.epochs,
            batch_size=self._args.batch_size,
            model_storage_rate=self._args.model_storage_rate,
            results_dir_path=self._results_dir_path,
            device=self._device)

        return model_trainer

    # def train_folds(self, results_dir_path, validation_fold):
    #     train_tuplets_generator, validation_tuplets_generator = create_tuplets_generator(
    #         validation_fold=validation_fold,
    #         logger=logger)
    #
    #     generate_validation_data(
    #         args=args,
    #         validation_tuplets_generator=validation_tuplets_generator,
    #         logger=logger)
    #
    #     generate_train_data(
    #         args=args,
    #         train_tuplets_generator=train_tuplets_generator,
    #         logger=logger)
    #
    #     train_dataset, validation_dataset = create_datasets(
    #         args=args,
    #         train_tuplets_generator=train_tuplets_generator,
    #         validation_tuplets_generator=validation_tuplets_generator,
    #         logger=logger)
    #
    #     model = create_model(
    #         args=args,
    #         logger=logger)
    #
    #     optimizer = create_optimizer(
    #         args=args,
    #         model=model,
    #         logger=logger)
    #
    #     loss_fn = create_loss_function(
    #         args=args,
    #         logger=logger)
    #
    #     model_trainer = create_model_trainer(
    #         args=args,
    #         model=model,
    #         optimizer=optimizer,
    #         loss_fn=loss_fn,
    #         logger=logger)
    #
    #     model_trainer.plot_samples(
    #         train_dataset=train_dataset,
    #         validation_dataset=validation_dataset,
    #         batch_size=args.batch_size)
    #
    #     model_trainer.fit(
    #         train_dataset=train_dataset,
    #         validation_dataset=validation_dataset,
    #         epochs=args.epochs,
    #         experiment_name=f'fold{validation_fold}',
    #         batch_size=args.batch_size)

    def run(self):
        self._logger.info(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
        self._logger.info(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
        self._logger.info(f'torch.version.cuda: {torch.version.cuda}')

        initial_model_file_path = os.path.normpath(os.path.join(self._results_dir_path, 'initial_model.pt'))
        self._logger.info(f'initial_model_file_path: {initial_model_file_path}')
        torch.save(self._model.state_dict(), initial_model_file_path)

        for validation_fold in self._args.folds:
            train_folds = [train_fold for train_fold in self._args.folds if train_fold != validation_fold]
            self._model.load_state_dict(torch.load(initial_model_file_path, map_location=self._device))
            self._generate_validation_data(folds=[validation_fold])
            self._generate_train_data(folds=train_folds)
            self._model_trainer.fit()

            # self._model_trainer.plot_samples()
            # self._model_trainer.fit()


