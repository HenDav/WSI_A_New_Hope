# python core
from abc import ABC, abstractmethod
import os
from datetime import datetime
from pathlib import Path
import shutil
import logging
import sys
from typing import List, TypeVar, Generic, cast

# git
import git

# pytorch
import torch
from torch.utils.data import Dataset

# gipmed
from core import utils
from core.base import LoggerObject, ArgumentsParser
from core.base import FactoryObject
from core.metadata import MetadataManager
from nn.datasets import WSIDataset, SSLDataset
from nn.trainers import ModelTrainer, SSLModelTrainer
from nn.feature_extractors import *

# tap
from tap import Tap


# =================================================
# Experiment Class
# =================================================
class Experiment(LoggerObject):
    def __init__(self, name: str, results_base_dir_path: str, model_trainers: List[ModelTrainer]):
        self._name = name
        self._results_base_dir_path = results_base_dir_path
        self._results_dir_path = self._create_results_dir_path()
        self._model_trainers = model_trainers
        self._log_file_path = utils.create_log_file_path(file_name=self.__class__.__name__, results_dir_path=self._results_dir_path)
        super(LoggerObject, self).__init__(log_file_path=self._log_file_path)

    def _create_results_dir_path(self) -> str:
        results_dir_path = os.path.normpath(os.path.join(self._results_base_dir_path, self._name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        return results_dir_path

    def _backup_codebase(self):
        repo = git.Repo('.', search_parent_directories=True)
        codebase_source_dir_path = repo.working_tree_dir
        codebase_destination_dir_path = os.path.normpath(os.path.join(self._results_dir_path, 'code'))
        shutil.copytree(src=codebase_source_dir_path, dst=codebase_destination_dir_path, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))

    def run(self):
        self._logger.info(msg=utils.generate_title_text(text=f'Running Experiment: {self._name}'))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='sys.argv', value=sys.argv, indentation=1, padding=30))
        self._backup_codebase()

        for model_trainer in self._model_trainers:
            model_trainer.train()


# =================================================
# ExperimentArgumentsParser Class
# =================================================
class ExperimentArgumentsParser(ABC, ArgumentsParser[Experiment]):
    name: str
    results_base_dir_path: str

    @abstractmethod
    def create_experiment(self) -> Experiment:
        pass


# =================================================
# ExperimentArgumentsParser Class
# =================================================
class SSLExperimentArgumentsParser(ExperimentArgumentsParser):
    T = TypeVar('T')

    feature_extractor_epochs: int
    feature_extractor_batch_size: int
    feature_extractor_num_workers: int
    feature_extractor_checkpoint_rate: int
    feature_extractor_folds: List[int]

    classifier_epochs: int
    classifier_batch_size: int
    classifier_num_workers: int
    classifier_checkpoint_rate: int
    classifier_folds: List[int]

    feature_extractor_train_dataset_json: str
    feature_extractor_validation_dataset_json: str

    classifier_train_dataset_json: str
    classifier_validation_dataset_json: str

    feature_extractor_loss_json: str
    feature_extractor_optimizer_json: str
    feature_extractor_model_json: str

    classifier_loss_json: str
    classifier_optimizer_json: str
    classifier_model_json: str

    feature_extractor_train_dataset_arguments_parser_json: str
    feature_extractor_validation_dataset_arguments_parser_name: str

    classifier_train_dataset_arguments_parser_name: str
    classifier_validation_dataset_arguments_parser_name: str

    feature_extractor_loss_arguments_parser_name: str
    feature_extractor_optimizer_arguments_parser_name: str
    feature_extractor_model_arguments_parser_name: str

    classifier_loss_arguments_parser_name: str
    classifier_optimizer_arguments_parser_name: str
    classifier_model_arguments_parser_name: str

    def create(self) -> Experiment:
        train_dataset = utils.argument_parser_type_cast(instance_type=SSLDataset, arguments_parser_name=self.dataset_arguments_parser_name)
        validation_dataset = utils.argument_parser_type_cast(instance_type=SSLDataset, arguments_parser_name=self.dataset_arguments_parser_name)
        feature_extractor_loss = utils.argument_parser_type_cast(instance_type=torch.nn.Module, arguments_parser_name=self.feature_extractor_loss_arguments_parser_name)
        # classifier_loss = utils.argument_parser_type_cast(instance_type=torch.nn.Module, arguments_parser_name=self.classifier_loss_arguments_parser_name)
        optimizer = utils.argument_parser_type_cast(instance_type=torch.nn.Module, arguments_parser_name=self.optimizer_arguments_parser_name)
        feature_extractor = utils.argument_parser_type_cast(instance_type=torch.nn.Module, arguments_parser_name=self.feature_extractor_arguments_parser_name)
        # classifier = utils.argument_parser_type_cast(instance_type=torch.nn.Module, arguments_parser_name=self.classifier_arguments_parser_name)

        feature_extractor_trainer = SSLModelTrainer(
            name='Feature Extractor Trainer',
            model=feature_extractor,
            loss=feature_extractor_loss,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=self.classifier_epochs,
            batch_size=self.feature_extractor_batch_size,
            folds=self.feature_extractor_folds,
            num_workers=self.feature_extractor_num_workers,
            checkpoint_rate=self.feature_extractor_checkpoint_rate,
            results_base_dir_path=self.results_base_dir_path,
            device=torch.device('cuda'))

        return Experiment(name=self.name, results_base_dir_path=self.results_base_dir_path, model_trainers=[feature_extractor_trainer])
