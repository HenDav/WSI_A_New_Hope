# python peripherals
import os
from multiprocessing import Process, Queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union

# numpy
import numpy

# gipmed
from core.base import OutputObject


class ParallelProcessorTask(ABC):
    @abstractmethod
    def process(self):
        pass

    @abstractmethod
    def post_process(self):
        pass


class ParallelProcessor(ABC, OutputObject):
    def __init__(self, name: str, output_dir_path: Path, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, **kw)
        self._tasks_queue = Queue()
        self._completed_tasks_queue = Queue()
        self._tasks = self._generate_tasks()
        self._completed_tasks = []

    def process(self, num_workers: int):
        for task in self._tasks:
            self._tasks_queue.put(obj=task)

        for _ in range(num_workers):
            self._tasks_queue.put(obj=None)

        workers = [Process(target=self._worker_func, args=tuple([worker_id],)) for worker_id in range(num_workers)]

        print('')

        for i, worker in enumerate(workers):
            worker.start()
            print(f'\rWorker Started {i+1} / {num_workers}', end='')

        print('')

        total_tasks_count = self.tasks_count + num_workers
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            print(f'\rRemaining Tasks {remaining_tasks_count} / {total_tasks_count}', end='')
            if remaining_tasks_count == 0:
                break

        print('')

        print('Draining queue')
        sentinels_count = 0
        while True:
            completed_task = self._completed_tasks_queue.get()
            if completed_task is None:
                sentinels_count = sentinels_count + 1
            else:
                self._completed_tasks.append(completed_task)

            if sentinels_count == num_workers:
                break

        print('Joining processes')
        for worker in workers:
            worker.join()

        print('Running post-process')
        self._post_process()

        print('Done!')

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    @abstractmethod
    def _post_process(self):
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
