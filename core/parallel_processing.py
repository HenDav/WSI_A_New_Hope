# python peripherals
import os
from multiprocessing import Process, Queue
import queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union, Generic, TypeVar
import traceback
from enum import Enum, auto

# numpy
import numpy

# gipmed
from core.base import OutputObject


# =================================================
# ParallelProcessorBase Class
# =================================================
class ParallelProcessorBase(ABC):
    def __init__(self, num_workers: int, **kw: object):
        self._num_workers = num_workers
        self._workers = []
        super(ParallelProcessorBase, self).__init__(**kw)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_workers']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def process(self):
        num_workers_digits = len(str(self._num_workers))

        print('Running Pre-Process')
        self._pre_process()

        self._workers = [Process(target=self._worker_func, args=tuple([worker_id],)) for worker_id in range(self._num_workers)]

        print('')
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1:{" "}{"<"}{num_workers_digits}} / {self._num_workers:{" "}{">"}{num_workers_digits}}', end='')
        print('')

        print('Running Post-Process')
        self._post_process()

    def join(self):
        print('Running Pre-Join')
        self._pre_join()

        print('Joining processes')
        for worker in self._workers:
            worker.join()

        print('Running Post-Join')
        self._post_join()

    @abstractmethod
    def _pre_process(self):
        pass

    @abstractmethod
    def _post_process(self):
        pass

    @abstractmethod
    def _pre_join(self):
        pass

    @abstractmethod
    def _post_join(self):
        pass

    @abstractmethod
    def _worker_func(self, worker_id: int):
        pass


# =================================================
# ParallelProcessorTask Class
# =================================================
class ParallelProcessorTask(ABC):
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def post_process(self):
        pass


# =================================================
# TaskParallelProcessor Class
# =================================================
class TaskParallelProcessor(ParallelProcessorBase, OutputObject):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, **kw)
        self._tasks_queue = Queue()
        self._completed_tasks_queue = Queue()
        self._tasks = self._generate_tasks()
        self._completed_tasks = []

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    def _pre_process(self):
        for task in self._tasks:
            self._tasks_queue.put(obj=task)

        for _ in range(self._num_workers):
            self._tasks_queue.put(obj=None)

    def _post_process(self):
        total_tasks_count = self.tasks_count + self._num_workers
        last_remaining_tasks_count = numpy.inf
        total_tasks_count_digits = len(str(total_tasks_count))
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            if last_remaining_tasks_count > remaining_tasks_count:

                print(f'\rRemaining Tasks {remaining_tasks_count:{" "}{"<"}{total_tasks_count_digits}} / {total_tasks_count:{" "}{">"}{total_tasks_count_digits}}', end='')
                last_remaining_tasks_count = remaining_tasks_count

            if remaining_tasks_count == 0:
                break

        print('')

        print('Draining Queue')
        sentinels_count = 0
        while True:
            completed_task = self._completed_tasks_queue.get()
            if completed_task is None:
                sentinels_count = sentinels_count + 1
            else:
                self._completed_tasks.append(completed_task)

            if sentinels_count == self._num_workers:
                break

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        pass

    def _worker_func(self, worker_id: int):
        while True:
            task = self._tasks_queue.get()
            if task is None:
                self._completed_tasks_queue.put(None)
                return

            try:
                task.process()
                task.post_process()
            except:
                print()
                traceback.print_exc()

            self._completed_tasks_queue.put(task)


# =================================================
# BioMarker Class
# =================================================
class GetItemPolicy(Enum):
    Replace = auto()
    TryReplace = auto()


# =================================================
# BufferedParallelProcessor Class
# =================================================
class BufferedParallelProcessor(ParallelProcessorBase, OutputObject):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, queue_maxsize: int, buffer_size: int, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, **kw)
        self._queue_maxsize = queue_maxsize
        self._buffer_size = buffer_size
        self._items_queue = Queue(maxsize=queue_maxsize)
        self._sentinels_queue = Queue()
        self._items_buffer = []

    def stop(self):
        for worker_id in range(self._num_workers):
            self._sentinels_queue.put(obj=None)

    def get_item(self, index: int, get_item_policy: GetItemPolicy) -> object:
        mod_index = numpy.mod(index, self._buffer_size)
        item = self._items_buffer[mod_index]

        new_item = None
        if get_item_policy == GetItemPolicy.TryReplace:
            try:
                new_item = self._items_queue.get_nowait()
            except queue.Empty:
                pass
        elif get_item_policy == GetItemPolicy.Replace:
            new_item = self._items_queue.get()

        if new_item is not None:
            rand_index = int(numpy.random.randint(self._buffer_size, size=1))
            self._items_buffer[rand_index] = new_item

        return item

    def _pre_process(self):
        pass

    def _post_process(self):
        while len(self._items_buffer) < self._buffer_size:
            self._items_buffer.append(self._items_queue.get())
            print(f'\rBuffer Populated with {len(self._items_buffer)} Items', end='')
        print('')

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_item(self) -> object:
        pass

    def _worker_func(self, worker_id: int):
        while True:
            try:
                self._sentinels_queue.get_nowait()
                break
            except:
                pass

            try:
                item = self._generate_item()
                self._items_queue.put(obj=item)
            except:
                print()
                traceback.print_exc()
