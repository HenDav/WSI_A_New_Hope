# python peripherals
import os
import numpy
import itertools
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import logging
import typing
from typing import Union, List, Tuple, Dict, Optional, Callable
from abc import ABC, abstractmethod
from typing import Protocol

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines

# gipmed
from nn.datasets import WSIDataset
from core.base import OutputObject
from core import utils


class EpochProcessor(Protocol):
    def __call__(self, epoch_index: int, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        ...


class ModelTrainer(OutputObject):
    def __init__(
            self,
            name: str,
            output_dir_path: Path,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            validation_dataset: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int,
            num_workers: int,
            checkpoint_rate: int,
            device: torch.device):
        super().__init__(name=name, output_dir_path=output_dir_path)
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._epochs = epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._checkpoint_rate = checkpoint_rate
        # self._results_dir_path = self._create_results_dir_path()
        self._model_file_path = self._create_model_file_path(epoch_index=None)
        self._train_dataset_size = len(self._train_dataset)
        self._validation_dataset_size = len(self._validation_dataset)
        self._train_indices = list(range(self._train_dataset_size))
        self._validation_indices = list(range(self._validation_dataset_size))
        self._train_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._train_dataset_size, batch_size=self._batch_size)
        self._validation_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._validation_dataset_size, batch_size=self._batch_size)
        self._device = device
        self._model.to(device)

    def train(self):
        self._logger.info(msg=utils.generate_title_text(text=f'Model Trainer: {self._name}'))

        self._logger.info(msg=utils.generate_bullet_text(text='Train Parameters', indentation=1))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Epochs', value=self._epochs, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Batch Size', value=self._batch_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Dataset Size', value=self._train_dataset_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Dataset Size', value=self._validation_dataset_size, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Batches per Epoch', value=self._train_batches_per_epoch, indentation=2, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Batches per Epoch', value=self._validation_batches_per_epoch, indentation=2, padding=30))

        self._logger.info(msg=utils.generate_bullet_text(text='Train Objects', indentation=1))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Model', obj=self._model))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Optimizer', obj=self._optimizer))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Loss', obj=self._loss))

        self._train()

    def plot_train_samples(self, batch_size: int, figsize: Tuple[int, int], fontsize: int):
        self._plot_samples(dataset=self._train_dataset, indices=self._train_indices, batch_size=batch_size, figsize=figsize, fontsize=fontsize)

    def plot_validation_samples(self, batch_size: int, figsize: Tuple[int, int], fontsize: int):
        self._plot_samples(dataset=self._validation_dataset, indices=self._validation_indices, batch_size=batch_size, figsize=figsize, fontsize=fontsize)

    def _plot_samples(self, dataset: Dataset, indices: List[int], batch_size: int, figsize: Tuple[int, int], fontsize: int):
        sampler = SequentialSampler(indices)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_data_aug = self._preprocess_batch(batch_data)
            for sample_index in range(batch_size):
                image_count = batch_data_aug.shape[1]
                fig, axes = plt.subplots(nrows=1, ncols=image_count, figsize=figsize)
                for image_index in range(image_count):
                    x = batch_data_aug[sample_index, image_index, :, :, :]
                    image = transforms.ToPILImage()(x)
                    axes[image_index].imshow(X=image)
                    axes[image_index].axis('off')
                    axes[image_index].set_title(f'Image #{image_index}', fontsize=fontsize)
                plt.show()

            if batch_index == 0:
                break

    def _train(self):
        trainer_results_dir_path = os.path.normpath(os.path.join(self._output_dir_path, self._name))
        Path(trainer_results_dir_path).mkdir(parents=True, exist_ok=True)
        self._train_epochs(results_dir_path=trainer_results_dir_path)

    def _process_epoch(self, epoch_index: int, data_loader: DataLoader, epoch_name: str, epoch_processor: EpochProcessor):
        self._logger.info(msg=utils.generate_bullet_text(text=f'{epoch_name} Epoch #{epoch_index}', indentation=1))
        epoch_processor(epoch_index=epoch_index, data_loader=data_loader)

    def _train_epochs(self, results_dir_path: str):
        best_validation_average_loss = None
        train_loss_array = numpy.array([])
        validation_loss_array = numpy.array([])
        train_sampler = SubsetRandomSampler(self._train_indices)
        validation_sampler = SubsetRandomSampler(self._validation_indices)
        train_data_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, sampler=train_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
        validation_data_loader = DataLoader(self._validation_dataset, batch_size=self._batch_size, sampler=validation_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
        loss_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss.npy'))
        for epoch_index in range(1, self._epochs + 1):

            self._process_epoch(epoch_index=epoch_index, data_loader=train_data_loader, epoch_name='Train', epoch_processor=self._train_epoch)
            self._process_epoch(epoch_index=epoch_index, data_loader=validation_data_loader, epoch_name='Validation', epoch_processor=self._validation_epoch)

            # self._logger.info(msg=utils.generate_bullet_text(text=f'Training Epoch #{epoch_index}', indentation=1))
            # train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            # train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])
            #
            # self._logger.info(msg=utils.generate_bullet_text(text=f'Validation Epoch #{epoch_index}', indentation=1))
            # validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            # validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])
            #
            # if best_validation_average_loss is None:
            #     torch.save(self._model.state_dict(), self._model_file_path)
            #     best_validation_average_loss = numpy.mean(validation_loss)
            # else:
            #     validation_average_loss = numpy.mean(validation_loss)
            #     if validation_average_loss < best_validation_average_loss:
            #         torch.save(self._model.state_dict(), self._model_file_path)
            #         best_validation_average_loss = validation_average_loss
            #
            # if epoch_index % self._checkpoint_rate == 0:
            #     lastest_model_path = self._create_model_file_path(epoch_index=epoch_index)
            #     torch.save(self._model.state_dict(), lastest_model_path)
            #
            # loss_data = {
            #     'train_loss': train_loss_array,
            #     'validation_loss': validation_loss_array,
            # }
            #
            # numpy.save(file=loss_file_path, arr=loss_data, allow_pickle=True)
            #
            # if (self._epochs is not None) and (epoch_index + 1 == self._epochs):
            #     break

    def _train_epoch(self, epoch_index, data_loader):
        self._model.train()
        return self._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._train_batch)

    def _validation_epoch(self, epoch_index, data_loader) -> float:
        self._model.eval()
        with torch.no_grad():
            return self._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._validation_batch)

    def _preprocess_batch(self, batch_data):
        return batch_data

    def _train_batch(self, batch_data):
        def closure():
            self._optimizer.zero_grad()
            loss = self._evaluate_loss(batch_data=batch_data)
            loss.backward()
            return loss

        final_loss = self._optimizer.step(closure)
        return final_loss.item()

    def _validation_batch(self, batch_data):
        loss = self._evaluate_loss(batch_data=batch_data)
        return loss.item()

    def _evaluate_loss(self, batch_data):
        in_features = self._preprocess_batch(batch_data=batch_data)
        out_features = self._model(in_features)
        return self._loss(out_features)

    def _epoch(self, epoch_index, data_loader, process_batch_fn):
        loss_array = numpy.array([])
        start = timer()
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_loss = process_batch_fn(batch_data)
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()

            batch_loss_text = utils.generate_batch_loss_text(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=float(numpy.mean(loss_array)),
                index_padding=8,
                loss_padding=25,
                batch_count=len(data_loader),
                batch_duration=end-start,
                indentation=1)

            self._logger.info(msg=batch_loss_text)

            start = timer()

        return loss_array

    def _create_model_file_path(self, epoch_index: Union[None, int]) -> str:
        if epoch_index is None:
            model_file_name = f'model_{epoch_index}.pt'
        else:
            model_file_name = f'model.pt'

        return os.path.normpath(os.path.join(self._output_dir_path, model_file_name))

    def _create_results_dir_path(self) -> str:
        results_dir_path = os.path.normpath(os.path.join(self._output_dir_path, self._name))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        return results_dir_path


class WSIModelTrainer(ModelTrainer):
    def __init__(
            self,
            name: str,
            output_dir_path: Path,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.nn.Module,
            train_dataset: WSIDataset,
            validation_dataset: WSIDataset,
            epochs: int,
            batch_size: int,
            num_workers: int,
            checkpoint_rate: int,
            device: torch.device,
            folds: Optional[List[int]],
            augmentations: Optional[torch.nn.Sequential] = None):
        super().__init__(
            name=name,
            model=model,
            loss=loss,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint_rate=checkpoint_rate,
            output_dir_path=output_dir_path,
            device=device)
        self._folds = folds
        # self._augmentations = augmentations
        self._augmentations = torch.nn.Sequential(
            transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=0.1, hue=(-0.1, 0.1)),
            transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip())
            # transforms.CenterCrop(tile_size)

    @property
    def _wsi_train_dataset(self) -> WSIDataset:
        return typing.cast(WSIDataset, self._train_dataset)

    @property
    def _wsi_validation_dataset(self) -> WSIDataset:
        return typing.cast(WSIDataset, self._validation_dataset)

    def _train(self):
        if self._folds is not None:
            for i, validation_fold in enumerate(self._folds):
                self._logger.info(msg=utils.generate_title_text(text=f'Cross Validation #{i+1}'))
                train_folds = [train_fold for train_fold in self._folds if train_fold != validation_fold]
                validation_folds = [validation_fold]
                self._wsi_train_dataset.set_folds(folds=train_folds)
                self._wsi_validation_dataset.set_folds(folds=validation_folds)
                cross_val_results_dir_path = self._create_cross_val_results_dir_path(train_folds=train_folds)
                self._train_epochs(results_dir_path=cross_val_results_dir_path)
        else:
            super()._train()

    def _preprocess_batch(self, batch_data: torch.Tensor):
        augmented_dims = []
        for i in range(batch_data.shape[1]):
            dim = (batch_data[:, i, :, :, :] / 255).to(self._device)
            dim_aug = transforms.Lambda(lambda items: torch.stack([self._augmentations(item) for item in items]))(dim)
            augmented_dims.append(dim_aug)

        return torch.stack(tensors=augmented_dims, dim=1)

    def _create_cross_val_results_dir_path(self, train_folds: List[int]) -> str:
        folds_dir_name = f"folds_{''.join(map(str, train_folds))}"
        cross_val_results_dir_path = os.path.normpath(os.path.join(self._output_dir_path, folds_dir_name))
        Path(cross_val_results_dir_path).mkdir(parents=True, exist_ok=True)
        return cross_val_results_dir_path
