# python core
from abc import ABC, abstractmethod
import os
from datetime import datetime
from pathlib import Path
import shutil
import logging
import sys
from typing import List

# git
import git

# pytorch
import torch
from torch.utils.data import Dataset

# gipmed
from core import utils
from core.base import LoggerObject
from core.metadata import MetadataManager
from nn.datasets import WSIDataset
from nn.trainers import ModelTrainer
from nn.networks import *


class Experiment(ABC, LoggerObject):
    def __init__(self, name: str, results_base_dir_path: str, model_trainers: List[ModelTrainer]):
        self._name = name
        self._results_base_dir_path = results_base_dir_path
        self._results_dir_path = self._create_results_dir_path()
        self._model_trainers = model_trainers
        self._log_file_path = utils.create_log_file_path(file_name=self.__class__.__name__, results_dir_path=self._results_dir_path)
        super(ABC, self).__init__()
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
