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

# tap
from tap import Tap

T = TypeVar("T")

# https://github.com/dabeaz/python-cookbook/blob/master/src/9/multiple_dispatch_with_function_annotations/example1.py
class MultiMethod:
    '''
    Represents a single multimethod.
    '''

    def __init__(self, name):
        self._methods = {}
        self.__name__ = name

    def register(self, meth):
        '''
        Register a new method as a multimethod
        '''
        sig = inspect.signature(meth)

        # Build a type-signature from the method's annotations
        types = []
        for name, parm in sig.parameters.items():
            if name == 'self':
                continue
            if parm.annotation is inspect.Parameter.empty:
                raise TypeError(
                    'Argument {} must be annotated with a type'.format(name)
                )
            if not isinstance(parm.annotation, type):
                raise TypeError(
                    'Argument {} annotation must be a type'.format(name)
                )
            if parm.default is not inspect.Parameter.empty:
                self._methods[tuple(types)] = meth
            types.append(parm.annotation)

        self._methods[tuple(types)] = meth

    def __call__(self, *args):
        '''
        Call a method based on type signature of the arguments
        '''
        types = tuple(type(arg) for arg in args[1:])
        meth = self._methods.get(types, None)
        if meth:
            return meth(*args)
        else:
            raise TypeError('No matching method for types {}'.format(types))

    def __get__(self, instance, cls):
        '''
        Descriptor method needed to make calls work in a class
        '''
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self


class MultiDict(dict):
    '''
    Special dictionary to build multimethods in a metaclass
    '''

    def __setitem__(self, key, value):
        if key in self:
            # If key already exists, it must be a multimethod or callable
            current_value = self[key]
            if isinstance(current_value, MultiMethod):
                current_value.register(value)
            else:
                mvalue = MultiMethod(key)
                mvalue.register(current_value)
                mvalue.register(value)
                super().__setitem__(key, mvalue)
        else:
            super().__setitem__(key, value)


class MultipleMeta(type):
    '''
    Metaclass that allows multiple dispatch of methods
    '''

    def __new__(cls, clsname, bases, clsdict):
        return type.__new__(cls, clsname, bases, dict(clsdict))

    @classmethod
    def __prepare__(cls, clsname, bases):
        return MultiDict()


class SeedableObject:
    seed: Union[None, int] = None

    def __init__(self):
        self._rng = numpy.random.default_rng(seed=SeedableObject.seed)

    @property
    def rng(self) -> numpy.random.Generator:
        return self._rng


class LoggerObject:
    def __init__(self, log_file_path: Path, level: int = logging.DEBUG):
        self._logger = utils.create_logger(log_file_path=log_file_path, name=self.__class__.__name__, level=level)


class FactoryObject(ABC):
    @staticmethod
    @abstractmethod
    def from_tap():
        pass


class ArgumentsParser(ABC, Tap, Generic[T]):
    @abstractmethod
    def create(self) -> T:
        pass


class JSONDecoderObject(ABC, Generic[T]):
    @staticmethod
    @abstractmethod
    def from_json(self) -> T:
        pass
