# python peripherals
import os
import numpy
import itertools
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer
import logging
import typing
from typing import Union, List

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
from core.base import LoggerObject
from core import utils


class ModelTrainer(LoggerObject):
    def __init__(
            self,
            name: str,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            validation_dataset: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int,
            num_workers: int,
            checkpoint_rate: int,
            results_base_dir_path: str,
            device: torch.device):
        self._name = name
        self._model = model
        self._loss = loss
        self._optimizer = optimizer
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._epochs = epochs
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._checkpoint_rate = checkpoint_rate
        self._results_base_dir_path = results_base_dir_path
        self._results_dir_path = self._create_results_dir_path()
        self._model_file_path = self._create_model_file_path(epoch_index=None)
        self._train_dataset_size = len(self._train_dataset)
        self._validation_dataset_size = len(self._validation_dataset)
        self._train_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._train_dataset_size, batch_size=self._batch_size)
        self._validation_batches_per_epoch = utils.calculate_batches_per_epoch(dataset_size=self._validation_dataset_size, batch_size=self._batch_size)
        self._device = device
        self._model.to(device)
        super(LoggerObject, self).__init__(log_file_base_dir=results_base_dir_path, log_file_name=self._name)

    def train(self):
        self._logger.info(msg=utils.generate_title_text(text='Train Parameters'))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Epochs', value=self._epochs, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Batch Size', value=self._batch_size, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Dataset Size', value=self._train_dataset_size, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Dataset Size', value=self._validation_dataset_size, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Training Batches per Epoch', value=self._train_batches_per_epoch, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_captioned_bullet_text(text='Validation Batches per Epoch', value=self._validation_batches_per_epoch, indentation=1, padding=30))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Model', obj=self._model))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Optimizer', obj=self._optimizer))
        self._logger.info(msg=utils.generate_serialized_object_text(text='Loss', obj=self._loss))
        self._train()

    def _train(self):
        results_dir_path = os.path.normpath(os.path.join(self._results_base_dir_path, self._name))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        self._train_epochs(results_dir_path=results_dir_path)

    def _train_epochs(self, results_dir_path: str):
        best_validation_average_loss = None
        train_loss_array = numpy.array([])
        validation_loss_array = numpy.array([])
        train_indices = list(range(self._train_dataset_size))
        validation_indices = list(range(self._validation_dataset_size))
        train_sampler = SubsetRandomSampler(train_indices)
        validation_sampler = SubsetRandomSampler(validation_indices)
        train_data_loader = DataLoader(self._train_dataset, batch_size=self._batch_size, sampler=train_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
        validation_data_loader = DataLoader(self._validation_dataset, batch_size=self._batch_size, sampler=validation_sampler, pin_memory=True, drop_last=False, num_workers=self._num_workers)
        loss_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss.npy'))
        for epoch_index in range(1, self._epochs + 1):

            self._logger.info(msg=utils.generate_bullet_text(text=f'Training Epoch #{epoch_index}', indentation=1))
            train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])

            self._logger.info(msg=utils.generate_bullet_text(text=f'Validation Epoch #{epoch_index}', indentation=1))
            validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])

            if best_validation_average_loss is None:
                torch.save(self._model.state_dict(), self._model_file_path)
                best_validation_average_loss = numpy.mean(validation_loss)
            else:
                validation_average_loss = numpy.mean(validation_loss)
                if validation_average_loss < best_validation_average_loss:
                    torch.save(self._model.state_dict(), self._model_file_path)
                    best_validation_average_loss = validation_average_loss

            if epoch_index % self._checkpoint_rate == 0:
                lastest_model_path = self._create_model_file_path(epoch_index=epoch_index)
                torch.save(self._model.state_dict(), lastest_model_path)

            loss_data = {
                'train_loss': train_loss_array,
                'validation_loss': validation_loss_array,
            }

            numpy.save(file=loss_file_path, arr=loss_data, allow_pickle=True)

            if (self._epochs is not None) and (epoch_index + 1 == self._epochs):
                break

    def _train_epoch(self, epoch_index, data_loader):
        self._model.train()
        return self._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._train_batch)

    def _validation_epoch(self, epoch_index, data_loader):
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

        return os.path.normpath(os.path.join(self._results_dir_path, model_file_name))

    def _create_results_dir_path(self) -> str:
        results_dir_path = os.path.normpath(os.path.join(self._results_base_dir_path, self._name))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)
        return results_dir_path


class CrossValidationModelTrainer(ModelTrainer):
    def __init__(
            self,
            name: str,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.nn.Module,
            train_dataset: WSIDataset,
            validation_dataset: WSIDataset,
            epochs: int,
            batch_size: int,
            folds: List[int],
            num_workers: int,
            checkpoint_rate: int,
            results_base_dir_path: str,
            device: torch.device):
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
            results_base_dir_path=results_base_dir_path,
            device=device)
        self._folds = folds

    @property
    def _wsi_train_dataset(self) -> WSIDataset:
        return typing.cast(WSIDataset, self._train_dataset)

    @property
    def _wsi_validation_dataset(self) -> WSIDataset:
        return typing.cast(WSIDataset, self._validation_dataset)

    def _train(self):
        for i, validation_fold in enumerate(self._folds):
            self._logger.info(msg=utils.generate_title_text(text=f'Cross Validation #{i+1}'))
            train_folds = [train_fold for train_fold in self._folds if train_fold != validation_fold]
            validation_folds = [validation_fold]
            self._wsi_train_dataset.set_folds(folds=train_folds)
            self._wsi_validation_dataset.set_folds(folds=validation_folds)
            cross_val_results_dir_path = self._create_cross_val_results_dir_path(train_folds=train_folds)
            self._train_epochs(results_dir_path=cross_val_results_dir_path)

    def _create_cross_val_results_dir_path(self, train_folds: List[int]) -> str:
        folds_dir_name = f"folds_{''.join(map(str, train_folds))}"
        cross_val_results_dir_path = os.path.normpath(os.path.join(self._results_dir_path, folds_dir_name))
        Path(cross_val_results_dir_path).mkdir(parents=True, exist_ok=True)
        return cross_val_results_dir_path


class SSLModelTrainer(CrossValidationModelTrainer):
    def __init__(
            self,
            name: str,
            model: torch.nn.Module,
            loss: torch.nn.Module,
            optimizer: torch.nn.Module,
            train_dataset: WSIDataset,
            validation_dataset: WSIDataset,
            epochs: int,
            batch_size: int,
            folds: List[int],
            num_workers: int,
            checkpoint_rate: int,
            results_base_dir_path: str,
            device: torch.device):

        super().__init__(
            name=name,
            model=model,
            loss=loss,
            optimizer=optimizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=epochs,
            batch_size=batch_size,
            folds=folds,
            num_workers=num_workers,
            checkpoint_rate=checkpoint_rate,
            results_base_dir_path=results_base_dir_path,
            device=device)

        self._transform = torch.nn.Sequential(
            transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=0.1, hue=(-0.1, 0.1)),
            transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip())
            # transforms.CenterCrop(tile_size)

    def _preprocess_batch(self, batch_data):
        x0 = (batch_data[:, 0, :, :, :] / 255).to(self._device)
        x1 = (batch_data[:, 1, :, :, :] / 255).to(self._device)
        x0_aug = transforms.Lambda(lambda x: torch.stack([self._transform(x_) for x_ in x]))(x0)
        x1_aug = transforms.Lambda(lambda x: torch.stack([self._transform(x_) for x_ in x]))(x1)
        return torch.stack((x0_aug, x1_aug), dim=1)

    def plot_samples(self, train_dataset, validation_dataset, batch_size):
        train_dataset_size = len(train_dataset)
        validation_dataset_size = len(validation_dataset)
        train_indices = list(range(train_dataset_size))
        validation_indices = list(range(validation_dataset_size))

        train_sampler = SequentialSampler(train_indices)
        validation_sampler = SequentialSampler(validation_indices)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, drop_last=False, num_workers=0)
        validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler, pin_memory=True, drop_last=False, num_workers=0)

        for batch_index, batch_data in enumerate(train_data_loader, 0):
            batch_data_aug = self._preprocess_batch(batch_data)
            for i in range(batch_size):
                x0 = batch_data_aug[i, 0, :, :, :]
                x1 = batch_data_aug[i, 1, :, :, :]
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))

                anchor_pic = transforms.ToPILImage()(x0)
                positive_pic = transforms.ToPILImage()(x1)

                axes[0].imshow(anchor_pic)
                axes[0].axis('off')
                axes[0].set_title('Anchor Tile')
                axes[1].imshow(positive_pic)
                axes[1].axis('off')
                axes[1].set_title('Positive Tile')

                plt.show()

            if batch_index == 0:
                break


