# python peripherals
import os
from multiprocessing import Process, Queue
import queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union, Generic, TypeVar

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
    @abstractmethod
    def process(self):
        print('Running Pre-Process')
        self._pre_process()

        self._workers = [Process(target=self._worker_func, args=tuple([worker_id],)) for worker_id in range(self._num_workers)]

        print('')
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {self._num_workers}', end='')
        print('')

        print('Running Pre-Join')
        self._pre_join()

        print('Joining processes')
        for worker in self._workers:
            worker.join()

        print('Running Post-Join')
        self._post_join()

        print('Processing Done!')

    @abstractmethod
    def _pre_process(self):
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
# ParallelProcessor Class
# =================================================
class ParallelProcessor(ABC, ParallelProcessorBase, OutputObject):
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

    def _pre_join(self):
        total_tasks_count = self.tasks_count + self._num_workers
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            print(f'\rRemaining Tasks {remaining_tasks_count} / {total_tasks_count}', end='')
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
                pass

            self._completed_tasks_queue.put(task)


# =================================================
# BufferedParallelProcessorTask Class
# =================================================
class BufferedParallelProcessorTask(ABC):
    @abstractmethod
    def process(self):
        pass


# =================================================
# BufferedParallelProcessor Class
# =================================================
class BufferedParallelProcessor(ABC, ParallelProcessorBase, OutputObject):
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

    def _pre_process(self):
        pass

    def _pre_join(self):
        while len(self._items_buffer) < self._buffer_size:
            self._items_buffer.append(self._items_queue.get())
            print(f'\rBuffer Populated with {len(self._items_buffer)} Items', end='')
        print('')

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_item(self) -> object:
        pass

    def get_item(self, index: int) -> object:
        mod_index = numpy.mod(index, self._buffer_size)
        item = self._buffer[mod_index]

        try:
            new_item = self._items_queue.get_nowait()
            rand_index = int(numpy.random.randint(self._buffer_size, size=1))
            self._buffer[rand_index] = new_item
        except queue.Empty:
            pass

        return item

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
                pass
