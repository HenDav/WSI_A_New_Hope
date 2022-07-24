# python peripherals
import os
import numpy
import itertools
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

# torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torchvision import transforms

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines


class ModelTrainer:
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        self._model = model
        self._loss_function = loss_function
        self._optimizer = optimizer
        self._device = device
        self._model.to(device)

    def fit(self, train_dataset, validation_dataset, epochs, batch_size, results_dir_path, shuffle_dataset=True):
        train_dataset_size = len(train_dataset)
        validation_dataset_size = len(validation_dataset)
        train_indices = list(range(train_dataset_size))
        validation_indices = list(range(validation_dataset_size))

        if shuffle_dataset is True:
            train_sampler = SubsetRandomSampler(train_indices)
            validation_sampler = SubsetRandomSampler(validation_indices)
        else:
            train_sampler = SequentialSampler(train_indices)
            validation_sampler = SequentialSampler(validation_indices)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False, num_workers=0)
        validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=False, num_workers=0)

        epochs_text = epochs if epochs is not None else 'infinite'

        ModelTrainer._print_training_configuration('Epochs', epochs_text)
        ModelTrainer._print_training_configuration('Train Batch size', batch_size)
        ModelTrainer._print_training_configuration('Validation Batch size', batch_size)
        ModelTrainer._print_training_configuration('Training dataset length', len(train_indices))
        ModelTrainer._print_training_configuration('Training batches per epoch', int(numpy.ceil(len(train_indices) / batch_size)))
        ModelTrainer._print_training_configuration('Validation dataset length', len(validation_indices))
        ModelTrainer._print_training_configuration('Validation batches per epoch', int(numpy.ceil(len(validation_indices) / batch_size)))

        model_file_path = os.path.normpath(os.path.join(results_dir_path, 'model.pt'))
        results_file_path = os.path.normpath(os.path.join(results_dir_path, 'results.npy'))
        model_architecture_file_path = os.path.normpath(os.path.join(results_dir_path, 'model_arch.txt'))
        loss_function_file_path = os.path.normpath(os.path.join(results_dir_path, 'loss_functions.txt'))
        optimizer_file_path = os.path.normpath(os.path.join(results_dir_path, 'optimizer.txt'))
        trainer_data_file_path = os.path.normpath(os.path.join(results_dir_path, 'trainer_data.txt'))
        Path(results_dir_path).mkdir(parents=True, exist_ok=True)

        with open(model_architecture_file_path, "w") as text_file:
            text_file.write(str(self._model))

        with open(loss_function_file_path, "w") as text_file:
            text_file.write(str(self._loss_function))

        with open(optimizer_file_path, "w") as text_file:
            text_file.write(str(self._optimizer))

        with open(trainer_data_file_path, "w") as text_file:
            text_file.write(f'train_batch_size: {batch_size}\n')
            text_file.write(f'validation_batch_size: {batch_size}\n')
            text_file.write(f'epochs: {epochs_text}\n')
            text_file.write(f'results_dir_path: {results_dir_path}\n')
            text_file.write(f'train_dataset_size: {train_dataset_size}\n')
            text_file.write(f'validation_dataset_size: {validation_dataset_size}\n')

        print(' - Start Training:')
        results = None
        best_validation_average_loss = None
        train_loss_array = numpy.array([])
        validation_loss_array = numpy.array([])
        for epoch_index in itertools.count():
            print(f'    - Training Epoch #{epoch_index+1}:')
            train_loss = self._train_epoch(epoch_index=epoch_index, data_loader=train_data_loader)
            train_loss_array = numpy.append(train_loss_array, [numpy.mean(train_loss)])
            print(f'    - Validation Epoch #{epoch_index+1}:')
            validation_loss = self._validation_epoch(epoch_index=epoch_index, data_loader=validation_data_loader)
            validation_loss_array = numpy.append(validation_loss_array, [numpy.mean(validation_loss)])

            if best_validation_average_loss is None:
                torch.save(self._model.state_dict(), model_file_path)
                best_validation_average_loss = numpy.mean(validation_loss)
            else:
                validation_average_loss = numpy.mean(validation_loss)
                if validation_average_loss < best_validation_average_loss:
                    torch.save(self._model.state_dict(), model_file_path)
                    best_validation_average_loss = validation_average_loss

            # lastest_model_path = os.path.normpath(os.path.join(results_dir_path, f'model_{epoch_index}.pt'))
            # torch.save(self._model.state_dict(), lastest_model_path)

            results = {
                'train_loss_array': train_loss_array,
                'validation_loss_array': validation_loss_array,
                'epochs': epochs_text,
                'train_batch_size': batch_size,
                'validation_batch_size': batch_size,
                'model_file_path': model_file_path,
                'results_file_path': results_file_path
            }

            numpy.save(file=results_file_path, arr=results, allow_pickle=True)

            if (epochs is not None) and (epoch_index + 1 == epochs):
                break

        return results

    def plot_samples(self, train_dataset, validation_dataset, batch_size):
        pass

    def _train_epoch(self, epoch_index, data_loader):
        self._model.train()
        return self._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._train_batch)

    def _validation_epoch(self, epoch_index, data_loader):
        self._model.eval()
        with torch.no_grad():
            return self._epoch(epoch_index=epoch_index, data_loader=data_loader, process_batch_fn=self._validation_batch)

    def _preprocess_batch(self, batch_data):
        return None

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
        return self._loss_function(out_features)

    def _epoch(self, epoch_index, data_loader, process_batch_fn):
        loss_array = numpy.array([])
        start = timer()
        for batch_index, batch_data in enumerate(data_loader, 0):
            batch_loss = process_batch_fn(batch_data)
            loss_array = numpy.append(loss_array, [batch_loss])
            end = timer()
            ModelTrainer._print_batch_loss(
                epoch_index=epoch_index,
                batch_index=batch_index,
                batch_loss=batch_loss,
                average_batch_loss=numpy.mean(loss_array),
                fill=' ',
                align='<',
                index_width=8,
                loss_width=25,
                batch_count=len(data_loader),
                batch_duration=end-start)
            start = timer()

        return loss_array

    @staticmethod
    def _print_training_configuration(title, value):
        print(f' - {title:{" "}{"<"}{30}} {value:{" "}{">"}{10}}')

    @staticmethod
    def _print_batch_loss(epoch_index, batch_index, batch_loss, average_batch_loss, fill, align, index_width, loss_width, batch_count, batch_duration):
        print(f'        - [Epoch {epoch_index+1:{fill}{align}{index_width}} | Batch {batch_index+1:{fill}{align}{index_width}} / {batch_count}]: Batch Loss = {batch_loss:{fill}{align}{loss_width}}, Avg. Batch Loss = {average_batch_loss:{fill}{align}{loss_width}}, Batch Duration: {batch_duration} sec.')


class WSIModelTrainer(ModelTrainer):
    def __init__(self, model, loss_function, optimizer, device='cuda'):
        ModelTrainer.__init__(self, model=model, loss_function=loss_function, optimizer=optimizer, device=device)
        self._transform = torch.nn.Sequential(
            transforms.ColorJitter(brightness=(0.85, 1.15), contrast=(0.75, 1.25), saturation=0.1, hue=(-0.1, 0.1)),
            transforms.GaussianBlur(3, sigma=(1e-7, 1e-1)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip())
            # transforms.CenterCrop(tile_size)

    def _preprocess_batch(self, batch_data):
        x0 = (batch_data[:, 0, :, :, :] / 255).to('cuda')
        x1 = (batch_data[:, 1, :, :, :] / 255).to('cuda')
        x0_aug = transforms.Lambda(lambda x: torch.stack([self._transform(x_) for x_ in x]))(x0)
        x1_aug = transforms.Lambda(lambda x: torch.stack([self._transform(x_) for x_ in x]))(x1)
        return torch.stack((x0_aug, x1_aug))

    def plot_samples(self, train_dataset, validation_dataset, batch_size):
        train_dataset_size = len(train_dataset)
        validation_dataset_size = len(validation_dataset)
        train_indices = list(range(train_dataset_size))
        validation_indices = list(range(validation_dataset_size))

        train_sampler = SequentialSampler(train_indices)
        validation_sampler = SequentialSampler(validation_indices)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False, num_workers=0)
        validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=False, num_workers=0)

        for batch_index, batch_data in enumerate(train_data_loader, 0):
            batch_data_aug = self._preprocess_batch(batch_data)
            for i in range(batch_size):
                x0 = batch_data_aug[0, i, :, :, :]
                x1 = batch_data_aug[1, i, :, :, :]
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


