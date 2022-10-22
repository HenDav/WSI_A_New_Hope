# python core
from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import List, Dict, Optional, Union, cast
import json

# git
import git

# gipmed
from core import utils
from core.base import OutputObject
from nn.feature_extractors import *
from nn.losses import *
from nn.datasets import *
from nn.trainers import *

# torch
from torch.nn import *
from torch.optim import *
from torchvision.models import *

# tap
from tap import Tap

# jsonpath
from jsonpath_ng import jsonpath, parse


# =================================================
# Experiment Class
# =================================================
class Experiment(OutputObject):
    def __init__(self, name: str, output_dir_path: Path, model_trainers: List[ModelTrainer]):
        self._model_trainers = model_trainers
        super().__init__(name=name, output_dir_path=output_dir_path)

    @property
    def model_trainers(self) -> List[ModelTrainer]:
        return self._model_trainers

    def _backup_codebase(self):
        repo = git.Repo('.', search_parent_directories=True)
        codebase_source_dir_path = repo.working_tree_dir
        codebase_destination_dir_path = os.path.normpath(os.path.join(self._output_dir_path, 'code'))
        shutil.copytree(src=codebase_source_dir_path, dst=codebase_destination_dir_path, ignore=shutil.ignore_patterns('.git', '.idea', '__pycache__'))

    def run(self):
        self._logger.info(msg=utils.generate_title_text(text=f'Running Experiment: {self._name}'))
        self._backup_codebase()

        for model_trainer in self._model_trainers:
            model_trainer.train()

    @staticmethod
    def from_json(json_file_path: Path) -> List[Experiment]:
        json_file = open(file=json_file_path)
        json_str = json_file.read()
        json_data = json.loads(json_str)
        experiments = []
        for experiment_dict in json_data['Experiments']:
            model_trainers = []
            experiment_dict['results_dir_path'] = Experiment._parse_path_list(path_list=experiment_dict['results_dir_path'], root=json_data)
            for model_trainer_dict in experiment_dict["model_trainers"]:
                model_trainer = cast(typ=ModelTrainer, val=Experiment._parse_model_trainer(obj_dict=model_trainer_dict, root=json_data))
                model_trainers.append(model_trainer)

            results_dir_path = Path(experiment_dict["results_dir_path"])
            shutil.copy(src=json_file_path, dst=results_dir_path / json_file_path.name)
            experiment = Experiment(name=experiment_dict["name"], output_dir_path=experiment_dict["results_dir_path"], model_trainers=model_trainers)
            experiments.append(experiment)

        return experiments

    @staticmethod
    def _parse_model_trainer(obj_dict: Dict, root: Dict, level_up_key: Optional[str] = None, model_trainer_root: Optional[Dict] = None) -> object:
        for key, value in obj_dict.items():
            if type(value) is not dict:
                if type(obj_dict[key]) is str:
                    obj_dict[key] = datetime.now().strftime(obj_dict[key])
                if type(obj_dict[key]) is list and key == 'results_dir_path':
                    obj_dict[key] = Experiment._parse_path_list(path_list=obj_dict[key], root=root)
                else:
                    obj_dict[key] = Experiment._try_parse_jsonpath(expression=value, data=root)

        for key, value in obj_dict.items():
            if type(value) is dict:
                obj_dict[key] = Experiment._parse_model_trainer(obj_dict=value, root=root, level_up_key=key, model_trainer_root=obj_dict if model_trainer_root is None else model_trainer_root)

        if "class_name" in obj_dict:
            if level_up_key == "optimizer":
                model = cast(torch.nn.Module, model_trainer_root["class_args"]["model"])
                obj_dict["class_args"]["params"] = model.parameters()

            return globals()[obj_dict["class_name"]](**obj_dict["class_args"])

        return obj_dict

    @staticmethod
    def _try_parse_jsonpath(expression: str, data: Dict) -> str:
        try:
            match = parse(expression).find(data)
            return match[0].value
        except:
            return expression

    @staticmethod
    def _parse_path_list(path_list: Union[str, list], root: Dict) -> Path:
        parsed_path_list = [Experiment._try_parse_jsonpath(expression=path_item, data=root) for path_item in path_list]
        parsed_path = Path(datetime.now().strftime(os.path.normpath(os.path.join(*parsed_path_list))))
        return parsed_path


# =================================================
# ExperimentArgumentsParser Class
# =================================================
class ExperimentArgumentsParser(Tap):
    json_file_path: Path
