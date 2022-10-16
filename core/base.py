# python peripherals
import os.path
from typing import Union
import logging
from pathlib import Path
import inspect
import types
from typing import Dict, Generic, TypeVar
from abc import ABC, abstractmethod

# numpy
import numpy

# gipmed
from core import utils


class SeedableObject:
    seed: Union[None, int] = None

    def __init__(self, **kw):
        self._rng = numpy.random.default_rng(seed=SeedableObject.seed)
        super(SeedableObject, self).__init__(**kw)

    @property
    def rng(self) -> numpy.random.Generator:
        return self._rng


class LoggerObject:
    def __init__(self, log_file_path: Path, level: int = logging.DEBUG, **kw):
        self._logger = utils.create_logger(log_file_path=log_file_path, name=self.__class__.__name__, level=level)
        super(LoggerObject, self).__init__(**kw)
