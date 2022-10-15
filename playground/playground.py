# python peripherals
from __future__ import annotations
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

T = TypeVar('T')


class FactoryObject(ABC, Generic[T]):
    # T = TypeVar('T', bound=Tap)

    @staticmethod
    @abstractmethod
    def from_tap() -> int:
        pass


class TestClass(FactoryObject):
    @staticmethod
    def from_tap():
        return TestClass()


if __name__ == '__main__':
    bla = TestClass()
    bla2 = TestClass.from_tap()

    # print('asdfy' * 10)
    # title = 'Hello'
    # value = 'Yoyo'
    # print(f' - {title:{" "}{"<"}{20}}{value}')
    #
    #
    # title = 'Hesdfgsdfgsdfgllo'
    # value = 'Yoyo'
    # print(f' - {title:{" "}{"<"}{20}} {value:{" "}{">"}{10}}')
    #
    # bla = '#####################\n'
    # bla = bla + '#                    #\n'
    # bla = bla + '#####################\n'
    # print(bla)

    # """main.py"""
    #
    # from tap import Tap
    #
    # class SimpleArgumentParser(Tap):
    #     name: str  # Your name
    #     language: str = 'Python'  # Programming language
    #
    #
    # class SimpleArgumentParser2(Tap):
    #     package: str = 'Tap'  # Package name
    #     stars: int  # Number of stars
    #     max_stars: int = 5  # Maximum stars
    #
    #
    # class SimpleArgumentParser3(SimpleArgumentParser, SimpleArgumentParser2):
    #     pass
    #
    #
    # args = SimpleArgumentParser3().parse_args()
    #
    # print(f'My name is {args.name} and I give the {args.language} package '
    #       f'{args.package} {args.stars}/{args.max_stars} stars!')

